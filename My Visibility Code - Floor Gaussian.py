import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt
import os
from tabulate import tabulate

# Extracts data from a text file into a numpy array
def extractData(filename, path=None):
    return np.loadtxt(path+filename,delimiter='\t', unpack=True, skiprows=1)

# Return 4 variables pertaining to peak data
def getPeaks(y_data):
    peaks = find_peaks_cwt(y_data, widths = np.logspace(0.1, 2, num=200) )
    troughs = find_peaks_cwt(-y_data, widths = np.logspace(0.1, 2, num=200) )  
    avg_peak = np.mean(y_data[peaks]) if len(peaks) > 0 else None
    avg_trough = np.mean(y_data[troughs]) if len(troughs) > 0 else None
    return peaks, troughs, avg_peak, avg_trough

def gaussian(x, a, x0, sigma):
    return a * np.exp( -(x - x0) ** 2 / (2 * sigma ** 2) )

def visibility(mins, maxs):
    mins = np.asarray(mins)
    maxs = np.asarray(maxs)
    v = (maxs - mins) / (maxs + mins)
    return v

# Filters y data within 3 standard deviations (eliminates vicious anomalies)
def filter_3sigma(y_data):
    lower_limit = np.mean(y_data) - 3 * np.std(y_data)
    upper_limit = np.mean(y_data) + 3 * np.std(y_data)
    filtered_y = y_data[(y_data >= lower_limit) & (y_data <= upper_limit)]
    return filtered_y

# Returns a list of all files with specific extensions in a specific directory
def compileDirectory(path=None, target ='.txt'):
    file_list = os.listdir(path) # Get all file names within the specififed path
    clean_list = []
    for item in file_list:
        if target in item: # If a given file contains the extension, append it
            clean_list.append(item)
    return clean_list

# Returns the minimums, maximums, and distance names arrays from the clean file list
def obtainDataset(clean_list, path=None):
    mins = []
    maxs = []
    distance = []
    for item in clean_list:
        x_data, y_data, _, _ = extractData(item, path)  # Get data
        filtered_y = filter_3sigma(y_data) # Filter data
        smoothed_filtered_y = np.convolve(filtered_y, np.ones(3) / 3, mode='valid')  # Smooth data
        peaks, troughs, avg_peak, avg_trough = getPeaks(smoothed_filtered_y)  # Get peak information
        mins.append(avg_trough)
        maxs.append(avg_peak)
        distance.append(int(str(item[0]) + str(item[1]) + str(item[3]) + str(item[4])) / 100)
    return mins, maxs, distance

def calculateErrors(distance, v, amp, mean, stdev):
    popt, pcov = curve_fit(gaussian, distance, v, p0=[amp, mean, stdev], maxfev=10000) # changing these parameter changes predicted fit lines
    return popt, pcov

# Define the variables needed
def variables(path=None, target=".txt", amp=0.25, mean=18.5, stdev=0.5):
    clean_list = compileDirectory(path, target)
    mins, maxs, distance = obtainDataset(clean_list, path)
    v = visibility(mins, maxs)
    popt, pcov = calculateErrors(distance, v, amp, mean, stdev)
    return distance, v, popt, pcov

# Plot the variables
def customPlotting(x_data, y_data, popt, pcov, title=None, xlabel=None, ylabel=None):
    spacing = x_data[1]-x_data[0]
    extended_x = np.linspace(x_data[0]-spacing*5, x_data[-1]+spacing*5, len(x_data)*100)
    errs = np.sqrt(np.diag(pcov))
    y_fit = gaussian(x_data, *popt)
    FWHM = 2*np.sqrt(2*np.log(2)) * popt[2]
    tc = FWHM/(3*10**11)
    print(*popt)
    print(*pcov)

    dataplot = plt.scatter(x_data, y_data, label=title)
    fitplot = plt.plot(extended_x, gaussian(extended_x, *popt), label="Predicted Gaussian Fit", color=dataplot.get_facecolor())
    print("\n",tabulate( [['A, Amplitude', ''+str(popt[0])+' , '+str(100*errs[0]/popt[0])+'%'],
                    ['x0, Centre', ''+str(popt[1])+' , '+str(100*errs[1]/popt[1])+'%'],
                    ['S, Sigma', ''+str(popt[2])+' , '+str(100*errs[2]/popt[2])+'%'],
                    ['R, Correlation', ''+str(pearsonr(y_data, y_fit)[0])+' , '+str(pearsonr(y_data, y_fit)[1])+'.'],
                    ['FWHM, Coherence Length', ''+str(FWHM)+' , '+str(100*errs[2]/popt[2])+'%']],
                    headers=['Gaussian Variable', 'Predicted Value With Erorr']))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(-0.1, 1.1)
    plt.legend()

distance, v, popt, pcov = variables("H:\\University Folder\\QIL Code\\Single Photon Data\\", "40.txt", 0.1, 18.38, 0.8)
customPlotting(distance, v, popt, pcov, "40 nm Filter Single Photons", "Displacement (mm)", "Visibility")

distance, v, popt, pcov = variables("H:\\University Folder\\QIL Code\\Single Photon Data\\", "3.txt", 0.1, 18.4, 0.5)
customPlotting(distance, v, popt, pcov, "40 nm & 3 nm Filter Single Photons", "Displacement (mm)", "Visibility")

distance, v, popt, pcov = variables("H:\\University Folder\\QIL Code\\Single Photon Data\\", "10.txt", 0.1, 18.41, 0.6)
customPlotting(distance, v, popt, pcov, "40 nm & 10 nm Filter Single Photons", "Displacement (mm)", "Visibility")
plt.show()

distance, v, popt, pcov = variables("H:\\University Folder\\QIL Code\\IR Laser Data\\", "IR-laser.txt", 0.82927, 18.0377, 2.12022)
#customPlotting(distance, v, popt, pcov, "IR Laser", "Displacement (mm)", "Visibility")
#plt.show()

distance, v, popt, pcov = variables("H:\\University Folder\\QIL Code\\HeNe Laser Data\\", "He-laser.txt", 0.05, 18.2, 0.6)
#customPlotting(distance, v, popt, pcov, "HeNe Laser", "Displacement (mm)", "Visibility")
#plt.show()

distance, v, popt, pcov = variables("H:\\University Folder\\QIL Code\\IR Laser Data\\", "IR-laser.txt", 0.7, 18.3, 0.2)
spacing = distance[1]-distance[0]
extended_x = np.linspace(distance[0]-spacing*5, distance[-1]+spacing*5, len(distance)*100)
errs = np.sqrt(np.diag(pcov))

dataplot = plt.scatter(distance, v, label="Full IR-Laser Data")
plt.title("Full IR-Laser")
plt.xlabel("Displacement (mm)")
plt.ylabel("Visibility")
plt.ylim(-0.1, 1.1)
plt.legend()
#plt.show()