"""
Fringe visibility analysis for Michelson interferometer scans.

For each laser source, a set of data files records photon counts on
detector "A" as a function of mirror displacement ("Position"). As the
mirror is scanned, interference fringes appear; their contrast
("visibility") falls off with displacement as the two arms' path-length
difference exceeds the source's coherence length. This script:

  1. Loads each scan file for a given laser/filter combination.
  2. Cleans and smooths the raw counts.
  3. Locates fringe peaks/troughs and computes visibility
     V = (I_max - I_min) / (I_max + I_min) for each mirror displacement.
  4. Fits a Gaussian to visibility vs. displacement (the expected shape
     of a coherence envelope) and plots the result.

Run directly (`python visibility_analysis.py`) to reproduce all of the
analyses defined in DATASETS below. No interactive input is required;
missing data directories are skipped with a warning rather than
raising an error, so the script degrades gracefully on any machine.
"""
from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from scipy.signal import find_peaks_cwt
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Width scale used by find_peaks_cwt to detect fringe peaks/troughs.
# Built once at import time (not per file) since it never changes and
# constructing it is one of the more expensive steps in the pipeline.
_PEAK_WIDTHS = np.logspace(0.1, 2, num=200)

# Filenames are expected to encode the mirror displacement as
# "<int_mm>_<hundredths_mm>mm", e.g. "17_00mm_He-laser.txt" -> 17.00 mm.
# This mirrors the convention used across all supplied data files.
_DISTANCE_PATTERN = re.compile(r"(\d+)_(\d+)mm")


# --------------------------------------------------------------------------
# Core signal-processing steps
# --------------------------------------------------------------------------

def load_scan(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load mirror position and detector-A counts from a scan file.

    Only the first two columns (Position, Counts A) are read: the
    supplied data files always have zero-valued "Counts B"/"Counts AB"
    columns, and only Counts A is used anywhere in this analysis, so
    reading the other columns would waste memory for no benefit.
    """
    position, counts_a = np.loadtxt(
        filepath, delimiter="\t", skiprows=1, usecols=(0, 1), unpack=True
    )
    return position, counts_a


def filter_3sigma(y_data: np.ndarray) -> np.ndarray:
    """Drop samples more than 3 standard deviations from the mean.

    Detector electronics occasionally register spurious spikes (visible
    in the raw data as isolated counts several times larger than their
    neighbours). These would otherwise be mistaken for fringe peaks, so
    they are removed before smoothing/peak-finding.
    """
    mean, std = y_data.mean(), y_data.std()
    return y_data[np.abs(y_data - mean) <= 3 * std]


def smooth(y_data: np.ndarray, window: int = 3) -> np.ndarray:
    """Apply a simple moving-average filter to reduce shot noise."""
    return np.convolve(y_data, np.ones(window) / window, mode="valid")


def find_peak_trough_means(y_data: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """Return the mean fringe-peak and fringe-trough heights in y_data.

    Peaks/troughs are located with a continuous-wavelet transform
    (find_peaks_cwt) rather than a simple local-maximum search, since
    fringe spacing varies slightly across a scan and cwt is more
    tolerant of that than a fixed-window method.

    Returns (None, None) if no peaks or troughs are found (e.g. a very
    short or unusually flat scan), so callers can skip that file rather
    than crash on an empty-array mean.
    """
    peak_idx = find_peaks_cwt(y_data, widths=_PEAK_WIDTHS)
    trough_idx = find_peaks_cwt(-y_data, widths=_PEAK_WIDTHS)
    avg_peak = float(y_data[peak_idx].mean()) if len(peak_idx) else None
    avg_trough = float(y_data[trough_idx].mean()) if len(trough_idx) else None
    return avg_peak, avg_trough


def visibility(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """Compute fringe visibility V = (max - min) / (max + min).

    Returns NaN (with a warning) for any entry where max + min == 0,
    rather than letting NumPy raise a hidden divide-by-zero warning or
    propagate an inf/-inf into the downstream curve fit.
    """
    mins = np.asarray(mins, dtype=float)
    maxs = np.asarray(maxs, dtype=float)
    denom = maxs + mins
    if np.any(denom == 0):
        logger.warning("visibility(): %d entries have max+min == 0; set to NaN", np.sum(denom == 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(denom != 0, (maxs - mins) / denom, np.nan)
    return v


def gaussian(x, h, a, x0, sigma):
    """Gaussian coherence-envelope model: h + a * exp(-(x-x0)^2 / (2*sigma^2))."""
    return h + a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


# --------------------------------------------------------------------------
# File discovery and per-dataset processing
# --------------------------------------------------------------------------

def parse_distance_mm(filename: str) -> float:
    """Extract the mirror displacement (mm) encoded in a filename.

    Expects the pattern "<int>_<hundredths>mm" as described in the
    module docstring (e.g. "17_00mm_He-laser.txt" -> 17.00). Raises
    ValueError with the offending filename if the pattern is absent,
    since silently mis-parsing a distance would corrupt the fit.
    """
    match = _DISTANCE_PATTERN.search(filename)
    if not match:
        raise ValueError(
            f"Could not parse mirror displacement from filename: {filename!r} "
            f"(expected a pattern like '17_00mm...')"
        )
    whole, hundredths = match.groups()
    return int(whole) + int(hundredths) / 100


def find_data_files(directory: Path, suffix: str) -> list[Path]:
    """Return all files in `directory` whose name contains `suffix`."""
    return [p for p in directory.iterdir() if suffix in p.name]


def process_file(filepath: Path) -> tuple[Optional[float], Optional[float]]:
    """Run the clean -> smooth -> peak-find pipeline on one scan file.

    Returns (avg_peak, avg_trough); either may be None if no fringes
    were detected in this file (see find_peak_trough_means).
    """
    _, counts = load_scan(filepath)
    cleaned = filter_3sigma(counts)
    smoothed = smooth(cleaned)
    return find_peak_trough_means(smoothed)


def build_dataset(directory: Path, suffix: str) -> tuple[np.ndarray, np.ndarray]:
    """Build (distance, visibility) arrays, sorted by distance, for all
    files in `directory` matching `suffix`.

    Files that fail to parse or yield no detectable fringes are skipped
    with a warning rather than aborting the whole dataset.
    """
    files = find_data_files(directory, suffix)
    if not files:
        raise FileNotFoundError(f"No files matching '{suffix}' found in {directory}")

    distances, mins, maxs = [], [], []
    for filepath in files:
        try:
            distance = parse_distance_mm(filepath.name)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", filepath.name, exc)
            continue

        avg_peak, avg_trough = process_file(filepath)
        if avg_peak is None or avg_trough is None:
            logger.warning("Skipping %s: no fringe peaks/troughs detected", filepath.name)
            continue

        distances.append(distance)
        mins.append(avg_trough)
        maxs.append(avg_peak)

    if not distances:
        raise ValueError(f"No usable data extracted from {directory} (suffix '{suffix}')")

    # Sort by displacement: file-listing order is not guaranteed, and an
    # unsorted x-array would make the fitted-curve plot span the wrong
    # range (matplotlib/np.linspace assume x_data[0]..x_data[-1] is the
    # scan's actual start/end).
    order = np.argsort(distances)
    distance_arr = np.asarray(distances)[order]
    v = visibility(np.asarray(mins)[order], np.asarray(maxs)[order])
    return distance_arr, v


def fit_gaussian(distance: np.ndarray, v: np.ndarray, p0: tuple[float, float, float, float]):
    """Fit the Gaussian coherence-envelope model to visibility data.

    Returns (popt, pcov), or (None, None) with a logged warning if the
    fit fails to converge -- e.g. because the data only cover part of
    the coherence envelope and don't resemble a single Gaussian peak.
    """
    try:
        popt, pcov = curve_fit(gaussian, distance, v, p0=p0, maxfev=10000)
        return popt, pcov
    except RuntimeError as exc:
        logger.warning("Gaussian fit did not converge (p0=%s): %s", p0, exc)
        return None, None


# --------------------------------------------------------------------------
# Reporting and plotting
# --------------------------------------------------------------------------

def print_fit_table(popt: np.ndarray, pcov: np.ndarray, distance: np.ndarray, v: np.ndarray) -> None:
    """Print fitted Gaussian parameters (with % error) and goodness of fit."""
    errs = np.sqrt(np.diag(pcov))
    fit_v = gaussian(distance, *popt)
    r, p = pearsonr(v, fit_v)
    rows = [
        ["H, Height", popt[0], f"{100 * errs[0] / popt[0]:.2f}%" if popt[0] else "n/a"],
        ["A, Amplitude", popt[1], f"{100 * errs[1] / popt[1]:.2f}%" if popt[1] else "n/a"],
        ["x0, Centre", popt[2], f"{100 * errs[2] / popt[2]:.2f}%" if popt[2] else "n/a"],
        ["S, Sigma", popt[3], f"{100 * errs[3] / popt[3]:.2f}%" if popt[3] else "n/a"],
        ["R, Correlation", r, f"p={p:.3g}"],
    ]
    print("\n" + tabulate(rows, headers=["Gaussian Variable", "Value", "Error / Stat"]))


def plot_dataset(
    ax: plt.Axes,
    distance: np.ndarray,
    v: np.ndarray,
    popt: Optional[np.ndarray],
    pcov: Optional[np.ndarray],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Scatter-plot visibility vs. distance, overlaying a Gaussian fit
    curve and printing its parameter table when a fit is available.

    Used for every plot in this script (single- and multi-panel figures
    alike), so the fit-overlay logic exists in exactly one place.
    """
    points = ax.scatter(distance, v, label=title)
    if popt is not None:
        spacing = distance[1] - distance[0] if len(distance) > 1 else 1.0
        extended_x = np.linspace(distance[0] - spacing * 5, distance[-1] + spacing * 5, len(distance) * 100)
        ax.plot(extended_x, gaussian(extended_x, *popt), label="Predicted Gaussian Fit", color=points.get_facecolor()[0])
        print_fit_table(popt, pcov, distance, v)
    else:
        logger.info("No converged fit for '%s'; plotting raw data only.", title)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.1, 1.1)
    ax.legend()


# --------------------------------------------------------------------------
# Dataset configuration and orchestration
# --------------------------------------------------------------------------

@dataclass
class Dataset:
    """One visibility analysis: a directory/suffix of scan files, a
    Gaussian initial guess, and the labels used for its plot."""
    directory: str
    suffix: str
    title: str
    p0: tuple[float, float, float, float]  # (height, amplitude, mean, stdev)
    xlabel: str = "Displacement (mm)"
    ylabel: str = "Visibility"
    group: int = 0        # datasets sharing a group number share one figure
    show_fit: bool = True  # False reproduces the original "full data" scatter-only plots


# Mirrors the six analyses in the original script. p0 guesses are
# preserved unchanged from the original per-block values.
DATASETS: list[Dataset] = [
    Dataset("Single Photon Data", "40.txt", "40 nm Filter Single Photons", (0.12, 0.1, 18.38, 0.8), group=1),
    Dataset("Single Photon Data", "3.txt", "40 nm & 3 nm Filter Single Photons", (0.05, 0.1, 18.4, 0.5), group=1),
    Dataset("Single Photon Data", "10.txt", "40 nm & 10 nm Filter Single Photons", (0, 0.1, 18.41, 0.6), group=1),
    #Dataset("IR Laser Data", "IR-laser.txt", "IR Laser", (-1, 0.7, 18.3, 0.2), group=2),
    Dataset("HeNe Laser Data", "He-laser.txt", "HeNe Laser", (0.5, 0.05, 18.2, 0.6), group=3),
    #Dataset("IR Laser Data", "IR-laser.txt", "Full IR-Laser", (-1, 0.7, 18.3, 0.2), group=4, show_fit=False),
]


def run_dataset(data_root: Path, ds: Dataset, ax: plt.Axes) -> None:
    """Build, fit (if requested), and plot a single Dataset onto `ax`."""
    directory = data_root / ds.directory
    if not directory.is_dir():
        logger.warning("Skipping '%s': directory not found: %s", ds.title, directory)
        return

    try:
        distance, v = build_dataset(directory, ds.suffix)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Skipping '%s': %s", ds.title, exc)
        return

    popt = pcov = None
    if ds.show_fit:
        popt, pcov = fit_gaussian(distance, v, ds.p0)

    plot_dataset(ax, distance, v, popt, pcov, ds.title, ds.xlabel, ds.ylabel)


def main(data_root: Path) -> None:
    """Run every analysis in DATASETS, grouping plots into figures by
    their `group` field (matching the original script's figure layout)."""
    groups: dict[int, list[Dataset]] = {}
    for ds in DATASETS:
        groups.setdefault(ds.group, []).append(ds)

    for group_id in sorted(groups):
        fig, ax = plt.subplots()
        for ds in groups[group_id]:
            run_dataset(data_root, ds, ax)
        plt.show()


def _parse_args() -> argparse.Namespace:
    # The three data folders ("HeNe Laser Data", "IR Laser Data",
    # "Single Photon Data") live alongside this script, mirroring the
    # original hardcoded paths (e.g. "...\QIL Code\Single Photon Data\"),
    # where the code and its data folders were siblings in one project
    # directory. Resolving relative to __file__ (rather than a bare
    # "./data") means this works regardless of the current working
    # directory the script happens to be launched from.
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=script_dir,
        help="Directory containing the per-laser data subfolders "
             "('HeNe Laser Data', 'IR Laser Data', 'Single Photon Data'). "
             f"Default: the script's own directory ({script_dir}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.data_root)