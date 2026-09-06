#Using cwt to find peaks, treating the data as a wavelet function.The rest is the same.
#Libraries
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt
import os

# Get data from txt file and return it into 4 length array
def getData(filename):
    return np.loadtxt(filename,delimiter='\t', unpack=True, skiprows=1)

# Return 4 variable pertaining to peak data
def getPeaks(y_data):
    peaks = find_peaks_cwt(y_data, widths = np.logspace(0.1, 2, num=200) )  # Find peaks and troughs,width may alter
    troughs = find_peaks_cwt(-y_data, widths = np.logspace(0.1, 2, num=200) )  
    avg_peak = np.mean(y_data[peaks]) if len(peaks) > 0 else None
    avg_trough = np.mean(y_data[troughs]) if len(troughs) > 0 else None
    return peaks, troughs, avg_peak, avg_trough


# Get all file names with '.txt'
def getNames(target = '.txt'):
    file_list = os.listdir()
    clean_list = []
    for item in file_list:
        if target in item:
            clean_list.append(item)
    return clean_list


clean_list = getNames(target='40.txt') # Get file names, change text in speech marks to change files you target
number_list = []
mins = [] # Store values
maxs = []

for item in clean_list: # For each file...
    x_data, y_data, _, _ = getData(item) # Get data
    smoothed_y = np.convolve(y_data, np.ones(4) / 4, mode='valid') # Smooth data, window may alter
    peaks, troughs, avg_peak, avg_trough = getPeaks(smoothed_y) # Get peak information
    mins.append(avg_trough) # Store average peak information
    maxs.append(avg_peak)

    number_list.append( int(str(item[0])+str(item[1])+str(item[3])+str(item[4]))/100 )

mins = np.asarray(mins)
maxs = np.asarray(maxs)
v = (maxs-mins)/(maxs+mins)
data = pd.DataFrame({'Distance (mm)': number_list, 'Minimum ()': mins, 'Maximum ()': maxs, 'Visibility ()': v}, index=clean_list) # Print data as dataframe

plt.plot(number_list, v, '.')
plt.show()

print(data.sort_values('Visibility ()'))