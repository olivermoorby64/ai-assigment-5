import os
import time
import statistics
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = r"H:\University Folder\QIL Code\Single Photon Data"

# Files to benchmark
TARGET = ".txt"

# Number of times each file is loaded by each method
REPEATS = 10


# ============================================================
# METHOD 1 - ORIGINAL
# ============================================================
# Loads ALL columns from the file.
#

def load_original(filepath):

    return np.loadtxt(
        filepath,
        delimiter="\t",
        unpack=True,
        skiprows=1
    )


# ============================================================
# METHOD 2 - ONLY LOAD REQUIRED COLUMNS
# ============================================================
# Loads only columns 0 and 1.
#
# This avoids reading/parsing columns that aren't needed.

def load_selected_columns(filepath):

    position, counts_a = np.loadtxt(
        filepath,
        delimiter="\t",
        skiprows=1,
        usecols=(0, 1),
        unpack=True
    )

    return position, counts_a


# ============================================================
# GET FILES
# ============================================================

def get_files(path, target):

    files = [
        filename
        for filename in os.listdir(path)
        if target in filename
    ]

    return sorted(files)


# ============================================================
# BENCHMARK ONE FILE
# ============================================================

def benchmark_file(filepath, repeats):

    original_times = []
    selected_times = []

    original_data = None
    selected_data = None

    # --------------------------------------------------------
    # ORIGINAL METHOD
    # --------------------------------------------------------

    for _ in range(repeats):

        start = time.perf_counter()

        original_data = load_original(filepath)

        end = time.perf_counter()

        original_times.append(end - start)

    # --------------------------------------------------------
    # SELECTED-COLUMN METHOD
    # --------------------------------------------------------

    for _ in range(repeats):

        start = time.perf_counter()

        selected_data = load_selected_columns(filepath)

        end = time.perf_counter()

        selected_times.append(end - start)

    # --------------------------------------------------------
    # Compare only the data actually used by your program
    #
    # Original:
    #   original_data[0] = position
    #   original_data[1] = counts_a
    #
    # New:
    #   selected_data[0] = position
    #   selected_data[1] = counts_a
    # --------------------------------------------------------

    data_identical = (
        np.array_equal(original_data[0], selected_data[0])
        and
        np.array_equal(original_data[1], selected_data[1])
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    original_mean = statistics.mean(original_times)
    selected_mean = statistics.mean(selected_times)

    original_median = statistics.median(original_times)
    selected_median = statistics.median(selected_times)

    original_min = min(original_times)
    selected_min = min(selected_times)

    original_max = max(original_times)
    selected_max = max(selected_times)

    # --------------------------------------------------------
    # Speed improvement
    # --------------------------------------------------------

    speedup = original_mean / selected_mean

    percentage_faster = (
        (original_mean - selected_mean)
        / original_mean
    ) * 100

    return {
        "Original Mean (ms)": original_mean * 1000,
        "Selected Mean (ms)": selected_mean * 1000,

        "Original Median (ms)": original_median * 1000,
        "Selected Median (ms)": selected_median * 1000,

        "Original Min (ms)": original_min * 1000,
        "Selected Min (ms)": selected_min * 1000,

        "Original Max (ms)": original_max * 1000,
        "Selected Max (ms)": selected_max * 1000,

        "Speedup": speedup,
        "Faster (%)": percentage_faster,

        "Data Identical": data_identical
    }


# ============================================================
# RUN BENCHMARK
# ============================================================

def run_benchmark(path, target=".txt", repeats=10):

    files = get_files(path, target)

    if not files:
        print("No matching files found.")
        return

    print("=" * 100)
    print("NUMPY FILE LOADING BENCHMARK")
    print("=" * 100)

    print(f"Directory : {path}")
    print(f"Files     : {len(files)}")
    print(f"Repeats   : {repeats}")
    print()

    results = []

    # --------------------------------------------------------
    # Benchmark every file
    # --------------------------------------------------------

    for number, filename in enumerate(files, start=1):

        filepath = os.path.join(path, filename)

        file_size = os.path.getsize(filepath)

        result = benchmark_file(
            filepath,
            repeats
        )

        result["File"] = filename
        result["Size (KB)"] = file_size / 1024

        results.append(result)

        print(
            f"{number:4d}/{len(files):4d}  "
            f"{filename:<30} "
            f"Original: {result['Original Mean (ms)']:8.3f} ms  "
            f"Selected: {result['Selected Mean (ms)']:8.3f} ms  "
            f"Speedup: {result['Speedup']:.2f}x"
        )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    df = pd.DataFrame(results)

    # Reorder columns
    df = df[
        [
            "File",
            "Size (KB)",

            "Original Mean (ms)",
            "Selected Mean (ms)",

            "Original Median (ms)",
            "Selected Median (ms)",

            "Original Min (ms)",
            "Selected Min (ms)",

            "Original Max (ms)",
            "Selected Max (ms)",

            "Speedup",
            "Faster (%)",

            "Data Identical"
        ]
    ]

    # ========================================================
    # OVERALL STATISTICS
    # ========================================================

    print("\n")
    print("=" * 100)
    print("OVERALL RESULTS")
    print("=" * 100)

    original_total = df["Original Mean (ms)"].sum()
    selected_total = df["Selected Mean (ms)"].sum()

    original_average = df["Original Mean (ms)"].mean()
    selected_average = df["Selected Mean (ms)"].mean()

    overall_speedup = original_total / selected_total

    overall_improvement = (
        (original_total - selected_total)
        / original_total
    ) * 100

    print()
    print(f"Number of files:              {len(files)}")

    print()
    print(
        f"Original total:               "
        f"{original_total:.3f} ms"
    )

    print(
        f"Selected-columns total:       "
        f"{selected_total:.3f} ms"
    )

    print()
    print(
        f"Original average per file:    "
        f"{original_average:.3f} ms"
    )

    print(
        f"Selected average per file:    "
        f"{selected_average:.3f} ms"
    )

    print()
    print(
        f"Overall speedup:              "
        f"{overall_speedup:.2f}x"
    )

    print(
        f"Overall improvement:          "
        f"{overall_improvement:.2f}%"
    )

    # ========================================================
    # DATA VALIDATION
    # ========================================================

    all_identical = df["Data Identical"].all()

    print()

    if all_identical:
        print("DATA VALIDATION: PASS")
        print(
            "Both methods produced identical position and "
            "counts data for every file."
        )
    else:
        print("DATA VALIDATION: FAIL")
        print(
            "At least one file produced different position "
            "or counts data."
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_file = "numpy_loading_comparison.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print()
    print(
        f"Detailed results saved to: {output_file}"
    )

    # ========================================================
    # PRINT TABLE
    # ========================================================

    print("\n")
    print("=" * 100)
    print("PER-FILE RESULTS")
    print("=" * 100)

    print(
        df[
            [
                "File",
                "Size (KB)",
                "Original Mean (ms)",
                "Selected Mean (ms)",
                "Speedup",
                "Faster (%)",
                "Data Identical"
            ]
        ].to_string(index=False)
    )

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results = run_benchmark(
        DATA_PATH,
        TARGET,
        REPEATS
    )