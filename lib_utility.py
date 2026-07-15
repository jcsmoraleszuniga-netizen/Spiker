import copy
import functools
import gc
import json
import re
import timeit
import warnings
from itertools import pairwise
from pathlib import Path
# from tkinter import filedialog
import numpy as np
from numba import njit
from numpy import diff, exp, array, nanmean, std, arange, ones, stack, trapz, where, zeros, mean, median, \
    percentile, \
    delete, \
    savetxt, gradient, concatenate, ndarray, dtype, signedinteger
from matplotlib import pyplot as plt
from numpy._typing import _32Bit, _64Bit
from scipy.ndimage import find_objects, label
from scipy.signal import fftconvolve, oaconvolve
from scipy.stats import stats
from typing import List, Tuple, Callable, Any, Optional, Iterable
from numpy.typing import NDArray
from scipy import optimize


def timing(function: Callable):
    @functools.wraps(function)
    def elapsed(*args):
        start = timeit.default_timer()
        print(f"{function.__name__:->40}")
        result = function(*args)
        total_time = timeit.default_timer() - start
        print(f"{function.__name__:+>60} {total_time = :.5f}")
        return result

    return elapsed


@njit
def vtp(time_value: float | np.floating, time_increment: float | np.floating) -> int | np.integer:
    """Transform Values To Points, given an increment"""
    return int(time_value / time_increment)


def vtp_relative(time_value: float | np.floating, time_array: NDArray[np.floating]) -> int:
    """Transform Values To Points, optimized for sorted time arrays using binary search."""

    # 1. Find the index where the value would be inserted to maintain order
    idx = np.searchsorted(time_array, time_value)

    # 2. Handle edge cases where the value is out of the array's bounds
    if idx == 0:
        return 0
    if idx == len(time_array):
        return len(time_array) - 1

    # 3. Compare the point before the index and the point at the index
    # We don't need absolute values here because we know the order of the elements
    left_diff = time_value - time_array[idx - 1]
    right_diff = time_array[idx] - time_value

    if left_diff < right_diff:
        return int(idx - 1)
    else:
        return int(idx)


def ptv(value: int | np.integer, value_increment: float | np.floating) -> float | np.floating:
    """Transform Values To Points, given an increment"""
    return value * value_increment


# @timing
@njit
def exp_decay(
        t: NDArray[np.floating], i0: np.floating, pk0: np.floating, t0: np.floating
        ) -> NDArray[np.floating]:
    return i0 + pk0 * exp(t / t0)


# @timing
def exp_rise(
        t: NDArray[np.floating], top: np.floating, bott: np.floating, v50: np.floating, slope: np.floating
        ) -> NDArray[np.floating]:
    return top + (bott - top) / (1 + (v50 - t) / slope)


# @timing
def linear(
        t: NDArray[np.floating], slope: np.floating, intercept: np.floating
        ) -> NDArray[np.floating]:
    return t * slope + intercept


# @timing
# def conv_vector(n_p: int, c_type: str = 'g', sharpness: int = 2) -> NDArray[np.floating]:
#     conv: NDArray[np.floating] = array([])
#     max_diff: float = 0.0006
#     if n_p < 9:
#         raise ValueError(f" Use a value of n_p >= 9 (minimum number of points). Try again.")
#     else:
#         if c_type == 'g':  # Gaussian convolution vector
#             sd: np.floating = std(arange(n_p / sharpness))
#             x = arange(n_p)
#             conv = exp(-(((x - n_p / 2) / sd) ** 2) / 2) / (sd * np.sqrt(2 * np.pi))
#             if np.abs(1 - np.sum(conv)) >= max_diff:
#                 print(f"{n_p = }")
#                 print(f"{sharpness = }")
#                 print(f"{sd = }")
#                 print(f"{x = }")
#                 print(f"{conv = }")
#                 print(f"{np.sum(conv) = }")
#                 raise ValueError(
#                         f"Vector sum is significantly different from 1: {1 - np.sum(conv) = :.6f}. Use a value of n_p >= 9. "
#                         f"\nMaximum difference accepted for the sum is: |1 - np.sum(conv)| < {max_diff}."
#                         )
#         elif c_type == 'f':  # Flat convolution vector
#             conv = ones(n_p) / n_p
#     return conv

def conv_vector(n_p: int, c_type: str = 'g', sharpness: float = 2.0) -> NDArray[np.floating]:
    if n_p < 9:
        raise ValueError("Use a value of n_p >= 9 (minimum number of points). Try again.")

    if c_type == 'g':  # Gaussian convolution vector
        # Calculate standard deviation
        sd: np.floating = np.std(np.arange(n_p / sharpness))

        # FIX 1: Calculate the exact mathematical center to prevent phase shifting
        x = np.arange(n_p)
        center = (n_p - 1) / 2.0

        # Generate the curve
        conv = np.exp(-0.5 * ((x - center) / sd) ** 2)

        # FIX 2: Force perfect normalization instead of crashing
        conv /= np.sum(conv)

        # plt.figure()  # for testing purposes
        # plt.plot(x, conv)
        # plt.title(f"Convolution kernel, delete me after... {n_p=} {sharpness=} {np.sum(conv)}")
        # plt.show()

    elif c_type == 'f':  # Flat (Boxcar) convolution vector
        conv = np.ones(n_p) / n_p

    else:
        raise ValueError(f"Unknown c_type '{c_type}'. Please use 'g' or 'f'.")

    return conv


def crossing_point(arr: NDArray[np.floating]) -> int | None:
    """Returns the first position where the array intersects with 0.0 using vectorization."""
    if arr.size < 2:
        return None

    # Get the signs (-1, 0, or 1)
    sign_array = np.sign(arr)

    # Find where the difference between consecutive elements is non-zero
    # np.diff(sign_array) returns sign_array[i+1] - sign_array[i]
    diffs = np.diff(sign_array)

    # Find indices where the difference is not zero
    changes = np.where(diffs != 0)[0]

    if changes.size > 0:
        # We add 1 because diff shifts the index by one (the change is detected at i+1)
        return int(changes[0] + 1)

    return None


# @timing
def find_over_threshold(
        response: NDArray[np.floating], threshold: NDArray[np.floating], direction: int
        ) -> NDArray[np.floating]:
    # RAM FIX: Replaced memory-heavy np.stack and np.max/min with direct boolean evaluation.
    if direction > 0:
        events_over_threshold = response > threshold
    elif direction < 0:
        events_over_threshold = response < threshold
    else:
        return np.zeros_like(response)

    return np.where(events_over_threshold, 1, 0)


def find_peaks_and_boundaries(
        over_threshold: NDArray[np.floating], response: NDArray[np.floating], direction: int,
        ) -> tuple[NDArray[np.floating], list[tuple[int, int, int]]]:
    """
    Finds peaks, their areas, and their boundaries in a single vectorized pass.

    Returns:
        tuple containing:
        - peaks (NDArray): Array with calculated sector areas located at peak indices.
        - boundaries (list[tuple]): List of (peak_index, start_index, end_index).
    """
    # 1. Initialize outputs
    peaks = np.zeros_like(over_threshold)
    boundaries_and_peaks = []

    # 2. Label the contiguous sectors
    labeled_array, num_features = label(over_threshold > 0)

    if num_features == 0:
        return peaks, boundaries_and_peaks

    # 3. Get the exact slice objects for every sector instantly
    slices = find_objects(labeled_array)

    # 4. Iterate through the slices
    for sl in slices:
        if sl is None:
            continue

        sector_slice = sl[0]
        start_idx = sector_slice.start
        end_idx = sector_slice.stop  # Exclusive upper bound

        sector_response = response[sector_slice]

        # Calculate the area representation of this sector
        sector_area = np.sum(np.abs(over_threshold[sector_slice]))

        # 5. Find the local index of the max/min within this isolated sector
        if direction == 1:
            local_peak_idx = int(np.argmax(sector_response))
        elif direction == -1:
            local_peak_idx = int(np.argmin(sector_response))
        else:
            print("Wrong direction")
            return peaks, []

        # 6. Map to global index
        global_peak_idx = start_idx + local_peak_idx

        # 7. Populate both data structures
        peaks[global_peak_idx] = sector_area
        boundaries_and_peaks.append((global_peak_idx, start_idx, end_idx))

    return peaks, boundaries_and_peaks
# @timing
# @njit
# def find_peaks(
#         over_threshold: NDArray[np.floating], response: NDArray[np.floating], direction: int,
#         ) -> NDArray[np.floating]:
#     # 1. Create the empty peaks array
#     peaks = np.zeros_like(over_threshold)
#
#     # 2. Label the contiguous sectors
#     labeled_array, num_features = label(over_threshold > 0)
#
#     if num_features == 0:
#         return peaks
#
#     # 3. Get the exact slice objects for every sector instantly
#     slices = find_objects(labeled_array)
#
#     # 4. Iterate through the slices and find the peak in each
#     for sl in slices:
#         if sl is None:
#             continue
#
#         sector_slice = sl[0]
#         sector_response = response[sector_slice]
#
#         # Calculate the area representation of this sector
#         # np.abs ensures the area is positive and comparable regardless of polarity
#         sector_area = np.sum(np.abs(over_threshold[sector_slice]))
#
#         # 5. Find the local index of the max/min within this isolated sector
#         if direction == 1:
#             local_peak_idx = np.argmax(sector_response)
#         elif direction == -1:
#             local_peak_idx = np.argmin(sector_response)
#         else:
#             print("Wrong direction")
#             return peaks
#
#         # 6. Map the local index back to the global array and assign the area weight
#         global_peak_idx = sector_slice.start + local_peak_idx
#         peaks[global_peak_idx] = sector_area
#
#     return peaks
#
#
# def find_peak_boundaries(
#         over_threshold: NDArray[np.floating], response: NDArray[np.floating], direction: int,
#         ) -> list[tuple[int, int, int]]:
#     """
#     Finds peaks and the boundaries of the area surrounding them.
#
#     Returns:
#         list[tuple[int, int, int]]: A list containing tuples of
#         (peak_index, start_index, end_index) for each detected sector.
#         Note: end_index is exclusive, perfect for standard Python slicing (e.g., array[start:end]).
#     """
#     boundaries_and_peaks = []
#
#     # 1. Label the contiguous sectors
#     labeled_array, num_features = label(over_threshold > 0)
#
#     if num_features == 0:
#         return boundaries_and_peaks
#
#     # 2. Get the exact slice objects for every sector instantly
#     slices = find_objects(labeled_array)
#
#     # 3. Iterate through the slices and find the peak and boundaries in each
#     for sl in slices:
#         if sl is None:
#             continue
#
#         sector_slice = sl[0]
#         start_idx = sector_slice.start
#         end_idx = sector_slice.stop  # This is the exclusive upper bound
#
#         sector_response = response[sector_slice]
#
#         # 4. Find the local index of the max/min within this isolated sector
#         if direction == 1:
#             local_peak_idx = int(np.argmax(sector_response))
#         elif direction == -1:
#             local_peak_idx = int(np.argmin(sector_response))
#         else:
#             print("Wrong direction")
#             return []
#
#         # 5. Map the local index back to the global array
#         global_peak_idx = start_idx + local_peak_idx
#
#         # 6. Store the peak and its surrounding boundaries
#         boundaries_and_peaks.append((global_peak_idx, start_idx, end_idx))
#
#     return boundaries_and_peaks


# @timing
def get_stats(arr: NDArray[np.floating]) -> Tuple[List[str], List[Any]]:
    try:
        res = stats.normaltest(arr)
        p_value = res.pvalue
    except ValueError:
        print("Normality test will be skipped")
        p_value = "Undetermined"
    name_lst = [
            "Count", "Average", "STD", "Median", "1st percentile", "3rd percentile", "IQR", "Min", "Max",
            "H0 normal: p-value"
            ]
    value_lst = [
            len(arr),
            mean(arr, dtype=np.float64),
            std(arr, dtype=np.float64),
            median(arr),
            (q1 := percentile(arr, 25)),
            (q3 := percentile(arr, 75)),
            q3 - q1,
            np.min(arr),
            np.max(arr),
            p_value,
            ]
    return name_lst, value_lst


# @timing
def find_outliers(arr: NDArray[np.floating]) -> ndarray[Any, dtype[signedinteger[Any] | dtype]]:
    q1: np.floating = percentile(arr, 25)
    q3: np.floating = percentile(arr, 75)
    iqr: np.floating = q3 - q1
    threshold = 1.5 * iqr
    return where((arr < q1 - threshold) | (arr > q3 + threshold))[0]


# @timing
def remove_outlier(array_2d: NDArray[np.floating]) -> NDArray[np.floating]:
    """Identify  the outliers in the 'y' axis of the array and then removes them with their respective 'x' values"""
    return delete(array_2d, find_outliers(array_2d.T[1]), axis=0)


# @timing
# def save(arr: iter, title_="save") -> None:
#     """Pop up a file dialog to save the list of values"""
#     files = [('All Files', '*.*'), ('CSV Files', '*.csv'), ('Text Document', '*.txt')]
#     file_name = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=files, title=title_)
#     savetxt(file_name, arr, delimiter=',')


# @timing
def auto_save(arr: iter, file_name='default') -> None:
    """Save the list of values automatically"""
    savetxt(file_name, arr, delimiter=',')


# @timing
# @njit
def differentiate(arr: NDArray[np.floating], incr: np.floating | float) -> NDArray[np.floating]:
    # np.diff is exactly f[i+1] - f[i]
    diff = np.diff(arr) / incr
    # Append a zero at the end to keep the array size identical and alignment correct
    return np.append(diff, 0.0)


# def differentiate(arr: NDArray[np.floating], incr: np.floating | float) -> NDArray[np.floating]:
#     return gradient(arr, incr)


# @timing
def opt_linear(
        x_arr: NDArray[np.floating], y_arr: NDArray[np.floating]
        ) -> tuple[ndarray | Iterable | int | float, Any, Any, Any, Any]:
    # linear: t * slope + intercept
    return optimize.curve_fit(linear, x_arr, y_arr)  # returns the parameters of the fitting


# @timing
def extender(
        x_segm: NDArray[np.floating],
        t_segm: NDArray[np.floating],
        t_common: NDArray[np.floating]
        ) -> NDArray[np.floating]:
    """Extends a segment with zeros to match a common timeline using pre-allocation."""

    # 1. Pre-allocate the full result array with zeros
    extd_evt = np.zeros_like(t_common)

    # 2. Find the start and end indices using searchsorted
    # np.searchsorted is O(log n), whereas np.where is O(n)
    start_idx = np.searchsorted(t_common, t_segm[0])
    end_idx = start_idx + len(x_segm)

    # 3. Drop the segment into the pre-allocated array
    extd_evt[start_idx:end_idx] = x_segm

    return extd_evt


# @timing
def loop(arr1: NDArray[np.floating], arr2: NDArray[np.floating]) -> NDArray[np.floating]:
    return array(
            [
                    arr1[pos - 1: pos + 2]
                    for pos in range(len(arr2))
                    if arr2[pos] and len(arr2[pos - 1: pos + 2]) == 3
                    ]
            ).flatten()


# @timing
def get_peaks_arr(time: NDArray[np.floating], peaks: NDArray[np.floating]) -> NDArray[np.floating]:
    return array(concatenate(([loop(time, peaks)], [loop(peaks, peaks)]), axis=0).T)


# @timing
# def make_sections(
#         start: int | float = 0, total: int | float = 1800, interval: int | float = 600
#         ) -> List[Tuple[int, int]]:
#     start = int(start)
#     total = int(total)
#     interval = int(interval)
#     points = [i for i in range(start, total + 1, interval)]
#     return [(points[i], points[i + 1]) for i in range(len(points) - 1)]

@timing
def make_sections(
        start: int | float = 0, total: int | float = 1800, interval: int | float = 600
        ) -> List[Tuple[int, int]]:
    start, total, interval = int(start), int(total), int(interval)

    # RAM FIX: Use the memory-efficient range object directly.
    points = range(start, total + 1, interval)

    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


# @timing
def apply_by_continuous(
        function: callable,
        arr: NDArray[np.floating],  # arr[time, resp]
        increment: dict,
        ) -> NDArray[NDArray[np.floating]]:
    """Calculates the average value for a certain increment in time.
    If no values are found in that increment then the value is 0.0"""
    # local_vtp = vtp
    local_vtp = vtp_relative
    # d_t = round(float(arr[0][1] - arr[0][0]), 5)
    increment["start"] = arr[0][0]
    increment["end"] = arr[0][-1]
    interv = [p_time for p_time in arange(increment["start"], increment["end"], increment["increment"])]
    return np.array(
            [
                    [
                            sum(pair) / 2,
                            function(arr[1][local_vtp(pair[0], arr[0]):local_vtp(pair[1], arr[0])])
                            # function(arr[1][local_vtp(pair[0] - arr[0][0], d_t):local_vtp(pair[1] - arr[0][0], d_t)])
                            ]
                    for pair in pairwise(interv)
                    ]
            ).T


# @timing
def apply_by_discrete(
        function: callable,
        arr: NDArray[NDArray[np.floating]],
        increment: dict,
        ) -> NDArray[NDArray[np.floating]]:
    """Calculates the average value for a certain increment in time.
    If no values are found in that increment then the value is 0.0"""
    l_function: list[list[float]] = []
    interv: NDArray[NDArray[np.floating]] = intervals(increment).astype(np.float32)
    for pair in pairwise(interv):
        pres_time = sum(pair) / 2
        section = where((pair[0] <= arr[0]) & (arr[0] < pair[1]))
        if len(arr[0][section]):
            l_function.append([pres_time, function(arr[1][section])])
        else:
            l_function.append([pres_time, 0.0])

    return array(l_function).T


# @timing
def apply_by(
        function: callable,
        arr: NDArray[NDArray[np.floating]],
        increment: dict,
        continuous: bool = False
        ) -> NDArray[NDArray[np.floating]]:
    """Calculates the average value for a certain increment in time.
    If no values are found in that increment then the value is 0.0"""
    if continuous:
        return apply_by_continuous(function, arr, increment)
    else:
        return apply_by_discrete(function, arr, increment)


# @timing
def average_by(
        arr: NDArray[NDArray[np.floating]],
        increment: dict,
        continuous: bool = False
        ) -> NDArray[NDArray[np.floating]]:
    """Calculates the average value for a certain increment in time.
    If no values are found in that increment then the average value is 0.0"""
    return apply_by(nanmean, arr, increment, continuous)


# @timing
def count_ones(arr: NDArray[np.floating]) -> int:
    return np.count_nonzero(arr == 1)


# @timing
def event_count(
        arr: NDArray[NDArray[np.floating]],
        increment: dict,
        continuous: bool = False
        ) -> NDArray[NDArray[np.floating]]:
    """Counts the number of events in a certain increment in time.
    If no values are found in that increment then the count is set as 0.0"""
    return apply_by(len, arr, increment, continuous)


# @timing
def event_pr(arr: NDArray[NDArray[np.floating]], increment: dict) -> NDArray[NDArray[np.floating]]:
    """Calculates the probability of an event for a certain increment in time.
    If no values are found in that increment then the probability is set as 0.0"""
    # Float conversion for effective normalization
    counts: NDArray[NDArray[np.floating]] = event_count(arr, increment).astype(np.float32)
    counts[1] = counts[1] / np.sum(counts[1])  # Normalization of the values
    return counts


# @timing
# def intervals(increment: dict) -> NDArray[NDArray[np.floating]]:
#     ratio = increment["end"] / increment["increment"]
#     n_times = round(ratio, 0)
#     if n_times - ratio >= 0:
#         tail = n_times
#     else:
#         tail = n_times + 1
#     arr = [p_time for p_time in arange(0, tail * increment["increment"] + 1, increment["increment"])]
#     if arr[0] != increment["start"]:
#         arr[0] = increment["start"]
#     if arr[-1] != increment["end"]:
#         arr[-1] = increment["end"]
#     return np.array(arr)

# @timing
def intervals(increment: dict) -> NDArray[np.floating]:
    ratio = increment["end"] / increment["increment"]
    n_times = round(ratio, 0)
    tail = n_times if (n_times - ratio >= 0) else n_times + 1

    # RAM FIX: Generate the numpy array directly. No list comprehensions.
    arr = np.arange(0, tail * increment["increment"] + 1, increment["increment"])

    if arr.size > 0:
        if arr[0] != increment["start"]:
            arr[0] = increment["start"]
        if arr[-1] != increment["end"]:
            arr[-1] = increment["end"]

    return arr


# @timing
def event_fr(arr: NDArray[NDArray[np.floating]], increment: dict) -> NDArray[NDArray[np.floating]]:
    """Calculates the frequency of an event for a certain increment in time.
    If no values are found in that increment then the frequency is set as 0.0"""
    # Float conversion for effective normalization
    counts: NDArray[NDArray[np.floating]] = event_count(arr, increment).astype(np.float32)
    interv: NDArray[NDArray[np.floating]] = intervals(increment).astype(np.float32)
    interv = diff(interv)
    counts[1] = counts[1] / interv
    return counts


# @timing
def event_aft(arr: NDArray[NDArray[np.floating]], increment: dict) -> NDArray[NDArray[np.floating]]:
    """Counts the number of events for a certain increment in time, and divides by the average of the events values.
    If no values are found in that increment then the count is set as 0.0"""
    # Float conversion for effective normalization
    counts: NDArray[NDArray[np.floating]] = event_count(arr, increment).astype(np.float32)
    # averages: NDArray[NDArray[np.floating]] = average_by(arr, increment).astype(np.float32)
    inter_events = np.append([0.0], np.diff(arr[0]))
    interv: NDArray[NDArray[np.floating]] = average_by(np.array([arr[0], inter_events]), increment).astype(
            np.float32
            )
    counts[1] = counts[1] / interv[1]  # Count/Average
    counts = (counts.T[~np.isnan(counts[0]) & ~np.isnan(counts[1])]).T  # Removes pairs that present np.nan values
    return counts


def save_plot(
        values, name_params: dict, units: str, plot_increment: dict, bins: int, func: callable, plot=True
        ) -> None:
    """Saves the data and plots it, ensuring unique names for burst/isolated subsets."""

    # ---------------------------------------------------------
    # CRITICAL FIX 1: The Early Exit Guard
    # Check if values is None, empty, or if the data arrays have length 0.
    # ---------------------------------------------------------
    if values is None or len(values) < 2 or len(values[0]) == 0 or len(values[1]) == 0:
        print(f"  -> Skipping save/plot for {name_params.get('analysis_type', 'Unknown')}: No data points available.")
        return  # Exit the function immediately!

    # Sanitize the array (converts any internal Nones to np.nan)
    values = np.array(values, dtype=float)

    # 1. CALCULATE VALUES
    if callable(func):
        func_values = func(values, plot_increment)
        func_name = func.__name__
    else:
        func_values = values
        func_name = ""

    # 2. GENERATE UNIQUE FILENAMES
    # Sanitize the analysis_type (remove spaces/dashes) for a clean filename
    clean_type = name_params["analysis_type"].replace(" - ", "_").replace(" ", "_").replace("*", "_")

    # We add clean_type to the list to prevent EPSC_Amplitude and Burst_AP_Amplitude from colliding
    base_name_list = name_params["common_name"] + [name_params["sweep_number"], clean_type, name_params["parameter"]]

    out_name = name_params["file_parent"] + make_name(base_name_list)
    out_name_mean = name_params["file_parent"] + make_name(base_name_list + [func_name])

    # 3. SAVE DATA
    if callable(func):
        auto_save(values.T, out_name)
        auto_save(func_values.T, out_name_mean)
    else:
        auto_save(values.T, out_name)

    # ---------------------------------------------------------
    # CRITICAL FIX 2: NaN-Safe Stats Calculation
    # ---------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_val = np.nanmean(values[1])
        median_val = np.nanmedian(values[1])

    # 4. PLOTTING
    if plot:
        plt.rcParams.update({'font.size': 8})

        def on_close(event):
            event.canvas.figure.clear()
            plt.close(event.canvas.figure)
            gc.collect()

        # Create a single figure with 1 row, 2 columns.
        # Adjusted figsize to standard 16:9-ish proportion for split pane.
        fig, (ax_hist, ax_time) = plt.subplots(1, 2, figsize=(7, 3))

        # ---------------------------------------------------------
        # LEFT PLOT: Histogram
        # ---------------------------------------------------------
        if not np.all(np.isnan(values[1])):
            ax_hist.hist(values[1], bins=bins, color='gray', alpha=0.7)
            if not np.isnan(mean_val):
                ax_hist.axvline(mean_val, color='r', label=f"Mean: {mean_val:.2f}")
            if not np.isnan(median_val):
                ax_hist.axvline(median_val, color='k', linestyle='--', label=f"Median: {median_val:.2f}")
        else:
            # If no valid data, plot a clean text placeholder instead of crashing
            ax_hist.text(
                    0.5, 0.5, 'Insufficient data\nfor histogram',
                    ha='center', va='center', transform=ax_hist.transAxes, color='gray'
                    )
            ax_hist.set_yticks([])  # Hide y-axis ticks for the empty plot

        ax_hist.set_title(f"Distribution: {name_params['analysis_type']}")
        ax_hist.set_xlabel(units)
        ax_hist.legend(prop={'size': 6})

        # ---------------------------------------------------------
        # RIGHT PLOT: Time Course
        # ---------------------------------------------------------
        ax_time.plot(values[0], values[1], "r+", alpha=0.3, label="Raw Events")
        ax_time.plot(
                func_values[0], func_values[1],
                'bo', label=f"Binned {func_name}", alpha=0.6, markersize=8
                )

        # Only plot the mean line if it's a valid number
        if not np.isnan(mean_val):
            ax_time.axhline(mean_val, color='r', linestyle='--', label=f"Mean: {mean_val:.2f}")

        ax_time.set_title(f"{name_params['analysis_type']}\n(n={len(values[1])})")
        ax_time.set_ylabel(units)
        ax_time.legend(prop={'size': 6})

        # Format spacing, hook the closing event, and show
        plt.tight_layout()
        fig.canvas.mpl_connect('close_event', on_close)
        plt.show(block=False)


# @timing
# def save_plot(
#         values, name_params: dict, units: str, plot_increment: dict, bins: int, func: callable, plot=True
#         ) -> None:
#     """Saves the data and plots it, ensuring unique names for burst/isolated subsets."""
#
#     # ---------------------------------------------------------
#     # CRITICAL FIX 1: The Early Exit Guard
#     # Check if values is None, empty, or if the data arrays have length 0.
#     # ---------------------------------------------------------
#     if values is None or len(values) < 2 or len(values[0]) == 0 or len(values[1]) == 0:
#         print(f"  -> Skipping save/plot for {name_params.get('analysis_type', 'Unknown')}: No data points available.")
#         return  # Exit the function immediately!
#
#     # Sanitize the array (converts any internal Nones to np.nan)
#     values = np.array(values, dtype=float)
#
#     # 1. CALCULATE VALUES
#     if callable(func):
#         func_values = func(values, plot_increment)
#         func_name = func.__name__
#     else:
#         func_values = values
#         func_name = ""
#
#     # 2. GENERATE UNIQUE FILENAMES
#     # Sanitize the analysis_type (remove spaces/dashes) for a clean filename
#     clean_type = name_params["analysis_type"].replace(" - ", "_").replace(" ", "_").replace("*", "_")
#
#     # We add clean_type to the list to prevent EPSC_Amplitude and Burst_AP_Amplitude from colliding
#     base_name_list = name_params["common_name"] + [name_params["sweep_number"], clean_type, name_params["parameter"]]
#
#     out_name = name_params["file_parent"] + make_name(base_name_list)
#     out_name_mean = name_params["file_parent"] + make_name(base_name_list + [func_name])
#
#     # 3. SAVE DATA
#     if callable(func):
#         auto_save(values.T, out_name)
#         auto_save(func_values.T, out_name_mean)
#     else:
#         auto_save(values.T, out_name)
#
#     # ---------------------------------------------------------
#     # CRITICAL FIX 2: NaN-Safe Stats Calculation
#     # ---------------------------------------------------------
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", category=RuntimeWarning)
#         mean_val = np.nanmean(values[1])
#         median_val = np.nanmedian(values[1])
#
#     # 4. PLOTTING
#     if plot:
#         plt.rcParams.update({'font.size': 8})
#
#         def on_close(event):
#             event.canvas.figure.clear()
#             plt.close(event.canvas.figure)
#             gc.collect()
#
#         # --- Figure 1: Time Course ---
#         fig1, ax1 = plt.subplots(figsize=(3, 3))
#         ax1.plot(values[0], values[1], "r+", alpha=0.3, label="Raw Events")
#         ax1.plot(
#                 func_values[0], func_values[1],
#                 'bo', label=f"Binned {func_name}", alpha=0.6, markersize=8
#                 )
#
#         # Only plot the mean line if it's a valid number
#         if not np.isnan(mean_val):
#             ax1.axhline(mean_val, color='r', linestyle='--', label=f"Mean: {mean_val:.2f}")
#
#         ax1.set_title(f"{name_params['analysis_type']}\n(n={len(values[1])})")
#         ax1.set_ylabel(units)
#         ax1.legend(prop={'size': 6})
#
#         fig1.canvas.mpl_connect('close_event', on_close)
#         plt.show(block=False)
#
#         # --- Figure 2: Histogram ---
#         fig2, ax2 = plt.subplots(figsize=(3, 2))
#
#         # ---------------------------------------------------------
#         # CRITICAL FIX 3: NaN-Safe Histogram Plotting
#         # ---------------------------------------------------------
#         if not np.all(np.isnan(values[1])):
#             ax2.hist(values[1], bins=bins, color='gray', alpha=0.7)
#             if not np.isnan(mean_val):
#                 ax2.axvline(mean_val, color='r', label=f"Mean: {mean_val:.2f}")
#             if not np.isnan(median_val):
#                 ax2.axvline(median_val, color='k', linestyle='--', label=f"Median: {median_val:.2f}")
#         else:
#             # If no valid data, plot a clean text placeholder instead of crashing
#             ax2.text(
#                     0.5, 0.5, 'Insufficient data\nfor histogram',
#                     ha='center', va='center', transform=ax2.transAxes, color='gray'
#                     )
#             ax2.set_yticks([])  # Hide y-axis ticks for the empty plot
#
#         ax2.set_title(f"Distribution: {name_params['analysis_type']}")
#         ax2.set_xlabel(units)
#         ax2.legend(prop={'size': 6})
#
#         fig2.canvas.mpl_connect('close_event', on_close)
#         plt.show(block=False)


def get_safe_filename(text: str) -> str:
    """Removes illegal characters from a string so it can be a Windows/Mac filename."""
    if not text:
        return ""
    # Replaces < > : " / \ | ? * with an underscore
    return re.sub(r'[<>:"/\\|?*]', '_', text)


def get_previous_folder(context: str = "") -> Optional[str]:
    """Retrieves the previously opened folder from a file."""
    safe_context = get_safe_filename(context)
    # Adding an underscore makes it cleaner: "Save to JSON__previous_folder.txt"
    file_name = f"{safe_context}_previous_folder.txt" if safe_context else "previous_folder.txt"

    try:
        with open(file_name, "r") as f:
            return f.read().strip()
    except (FileNotFoundError, OSError):
        # Catch OSError too! That's what actually caused your crash.
        return None


def save_previous_folder(folder_path: str, context: str = ""):
    safe_context = get_safe_filename(context)
    file_name = f"{safe_context}_previous_folder.txt" if safe_context else "previous_folder.txt"

    try:
        with open(file_name, "w") as f:
            f.write(folder_path)
    except Exception as e:
        print(f"Failed to save folder history: {e}")


# @timing
# def get_previous_folder(context: str = "") -> Optional[str]:
#     """Retrieves the previously opened folder from a file."""
#     try:
#         print(f"Delete me after... {context + "previous_folder.txt"}")
#         with open(context + "previous_folder.txt", "r") as f:
#             return f.read().strip()
#     except FileNotFoundError:
#         print(f"Delete me after FileNotFoundError")
#         return None


# @timing
# def save_previous_folder(folder_path: str, context: str = "") -> None:
#     """Saves the given folder path to a file."""
#     name = context + "previous_folder.txt"
#     with open(name, "w") as f:
#         f.write(folder_path)


# @timing
def mse(arr1, arr2) -> float:
    """Calculates minimum square error for two arrays of the same length"""
    return np.mean(np.square(arr1 - arr2))


# @timing
# # @njit
# def prev_change(der_test_resp: np.ndarray, prev: float = 0.0) -> np.ndarray:
#     """
#     Optimizes the given Python loop using NumPy for faster execution.
#     """
#     der_test_resp = np.array(der_test_resp)  # Convert to NumPy array if it's a list
#     # Create a shifted array to check the previous element
#     shifted_resp = np.concatenate(
#             ([prev], der_test_resp[:-1])
#             )  # prepends a zero to the array and removes the last element.
#     # Find the indices where the previous element is not zero
#     indices_to_zero = shifted_resp != prev
#     # Set the corresponding elements in the original array to zero
#     der_test_resp[indices_to_zero] = prev
#
#     return der_test_resp

# @timing
def prev_change(der_test_resp: np.ndarray, prev: float = 0.0) -> np.ndarray:
    der_test_resp = np.array(der_test_resp)

    # RAM FIX: Slice comparisons avoid creating a shifted copy of the array in RAM.
    if len(der_test_resp) > 1:
        # Find where the previous element (offset by 1) is not equal to 'prev'
        indices_to_zero = der_test_resp[:-1] != prev
        # Apply the mask offset by 1 to modify the current elements
        der_test_resp[1:][indices_to_zero] = prev

    return der_test_resp


# @timing
def down_sample_function(arr, down_sample=10):
    return arr[::down_sample]


# @timing
def down_sample_function_t(arr, accept_mask, down_sample=10):
    # 1. Create a mask for the downsampling (every Nth element)
    # np.arange creates indices, then we check the modulo
    periodic_mask = (np.arange(len(arr)) % down_sample == 0)
    # 2. Combine with the accept_mask using a bitwise OR (|)
    # This keeps elements where val == 1 OR the index is a multiple of down_sample
    combined_mask = (accept_mask >= 1) | periodic_mask
    # 3. Use boolean indexing to filter the array
    return arr[combined_mask]


# @timing
def smoothing(resp, points, sharpness=4):
    """
    Smooths an array using Overlap-Add convolution to prevent out-of-memory
    errors on massive datasets (e.g., 40M+ points).
    """
    # 1. Calculate the effective points (width) for a single pass
    # Using the property: sigma_total = sigma * sqrt(n)
    # eff_points = points * np.sqrt(repetitions)

    # 2. Generate the single, wider kernel
    # (Assuming conv_vector is defined elsewhere in your lib_utility or lib_event_detection)
    kernel = conv_vector(points, 'g', sharpness)

    # 3. Padding logic (remains the same to prevent edge diving)
    pad_len = len(kernel)
    padded_resp = np.pad(resp, pad_len, mode='edge')

    # 4. Memory-Efficient Convolution
    # oaconvolve chunks the massive padded_resp array, preventing huge RAM spikes
    result = oaconvolve(padded_resp, kernel, mode='same')

    # 5. Slice and return
    return result[pad_len:-pad_len]


def reset_array(arr: np.ndarray, point: int | np.ndarray, value: float = 1.0) -> np.ndarray:
    """Returns a new array of zeros with specific indices set to a value."""
    # Create a completely new array with the same shape and type as the input
    new_arr = np.zeros_like(arr)

    # Assign the value to the specific indices on the NEW array
    new_arr[point] = value

    return new_arr


# @timing
def split_position(arr: np.ndarray, direction: int) -> signedinteger[_32Bit | _64Bit] | None:
    match direction:
        case -1:
            return np.argmin(arr)
        case 1:
            return np.argmax(arr)
        case _:
            return None


# @timing
def remove_shift(arr_base, arr_resp):
    """Remove shift of the response over time."""
    # pr_lin = opt_linear(arr_base[0], arr_base[1])
    slope, intercept, r_value, p_value, std_err, intercept_stderr = lin_fit(arr_base[1], arr_base[0])
    # arr_resp[1] -= linear(arr_resp[0], *pr_lin[0])
    arr_resp[1] -= linear(arr_resp[0], slope, intercept)
    return arr_resp


# @timing
def make_name(lst: list, extension='.csv') -> str:
    lst_str = [str(i) for i in lst]
    return "_".join(lst_str) + extension


# @timing
def replace(this: str, that: str, string: str) -> str:
    """Replaces every occurrence of 'this' with 'that' in 'string'"""
    return that.join(string.split(this))


@timing
def load_dict(const_file: str, const: dict):
    print(f"{const_file = }")
    file_path = Path(const_file)
    if file_path.exists():
        with open(const_file, "rb") as f:
            print("Loading...")
            loaded_dict = json.load(f)
        const.update(loaded_dict)  # Intended to make sure that new variables are present with default values
        loaded_dict = copy.deepcopy(const)  # updated version of loaded_dict
        return loaded_dict
    else:
        print("Dict not found, using default...")
        return const


@timing
def save_dict(const_file: str, loaded_dict: dict):
    # Save the dictionary to a json file
    with open(const_file, "w") as f:
        json.dump(loaded_dict, f)


# @timing
def file_info(path_to_file, parameter):
    p = Path(path_to_file)
    match parameter:
        case 'path':
            return str(p)
        case 'name':
            return p.name
        case 'number':
            return p.name.split("_")[-1].split(".")[0]
        case 'parent':
            return p.parent.as_posix() + '/'
        case _:
            print("No parameter was entered")
            return None


# @timing
def calculate_area(resp_time, resp):
    # return integrate.simpson(resp, resp_time)
    return trapz(resp, resp_time)


# @timing
def correct_bound(value):
    if value <= 0:
        return 0
    else:
        return value


# @timing
@njit
def exp_to_lin(arr: np.ndarray, direction: int) -> np.ndarray:
    """Transforms an exponential decay curve to a linear curve.
    The exponential form has to be: I(t) = pk0 * exp(-t / t0)
    The linear form is: Ln(I(t)) = Ln(pk0) - t/t0"""
    if len(arr):
        match direction:
            case 1:  # positive going
                y = arr
            case -1:  # negative going
                y = -arr
            case _:
                raise ValueError("Wrong direction")
        return np.log(y + 1.0)
    else:
        raise ValueError("Array is empty")


# @timing
@njit
def lin_fit(y_var, x_var):
    """
    Fits data to a linear relation: y = intercept + slope * x
    Manual implementation for Numba compatibility.

    Returns: slope, intercept, r_value, p_value, std_err, intercept_stderr
    (Note: p_value and stderrs are returned as 0.0)
    """
    n = len(x_var)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Calculate sums required for linear regression
    S_x = np.sum(x_var)
    S_y = np.sum(y_var)
    S_xx = np.sum(x_var ** 2)
    S_yy = np.sum(y_var ** 2)
    S_xy = np.sum(x_var * y_var)

    denom = n * S_xx - S_x ** 2

    if denom == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Calculate Slope and Intercept
    slope = (n * S_xy - S_x * S_y) / denom
    intercept = (S_y - slope * S_x) / n

    # Calculate R value (Pearson coefficient)
    r_num = n * S_xy - S_x * S_y
    r_den = np.sqrt((n * S_xx - S_x ** 2) * (n * S_yy - S_y ** 2))

    if r_den == 0:
        r_value = 0.0
    else:
        r_value = r_num / r_den

    # P-value and Standard Errors are difficult to compute in nopython mode
    # (requires t-distribution CDF). We return 0.0 to satisfy the unpacking
    # signature expected by 'exp_fit'.
    return slope, intercept, r_value, 0.0, 0.0, 0.0


# @timing
@njit
def exp_fit(response, time, direction):
    """Fits data to an exponential decay.
    I(t) = i0 + pk0 * exp(-t / t0), returns i0, pk0, t0 and pearson_r"""
    lin_resp = exp_to_lin(response, direction)
    inv_t0, lin_pk0, r_value, p_value, std_err, intercept_stderr = lin_fit(lin_resp, time)
    fit_pk0 = np.exp(lin_pk0) * direction
    fit_t0 = 1.0 / inv_t0
    fit_i_0 = 0  # Maintained just for compatibility
    return fit_i_0, fit_pk0, fit_t0, r_value  # i0, pk0, t0, r


# @timing
def parabolic_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Parabolic fit of an array of points:
    y(x) = ax + bx + cx², returns a, b, c"""
    n = x.size
    sum_y = np.sum(y)
    sum_x = np.sum(x)
    sum_x2 = np.sum(np.power(x, 2))
    sum_x3 = np.sum(np.power(x, 3))
    sum_x4 = np.sum(np.power(x, 4))
    sum_x_y = np.sum(x * y)
    sum_x2_y = np.sum(np.power(x, 2) * y)
    coefficients = np.array(
            [
                    [n, sum_x, sum_x2],
                    [sum_x, sum_x2, sum_x3],
                    [sum_x, sum_x3, sum_x4]
                    ]
            )
    constants = np.array(
            [sum_y, sum_x_y, sum_x2_y]
            )
    print(f"{coefficients=}")
    print(f"{constants=}")
    return np.linalg.solve(coefficients, constants)  # a, b, c


def remove_nan(arr: np.ndarray) -> np.ndarray:
    """Removes nan values from numpy arrays"""
    return arr[~np.isnan(arr)]


def get_names(ori_inst, script):
    file_name: str = ori_inst.get_info('file', 'name')
    file_name = replace(".", "_", file_name)  # Dot removal
    print(f"{file_name = }")
    file_number: str = ori_inst.get_info('file', 'number')
    print(f"{file_number = }")
    file_parent: str = ori_inst.get_info('file', 'parent')
    print(f"{file_parent = }")
    script_name = file_info(script, 'name')
    script_name = replace(".", "_", script_name)  # Dot removal
    print(f"{script_name = }")

    return file_name, file_number, file_parent, script_name


def calculate_fitted_angle(group1: NDArray, group2: NDArray) -> float:
    """
    Calculates the angle between two lines of best fit.
    """
    try:
        # Fit lines (y = mx + b)
        m1, b1 = np.polyfit(group1[:, 0], group1[:, 1], 1)
        m2, b2 = np.polyfit(group2[:, 0], group2[:, 1], 1)

        # If slopes are nearly identical, it's essentially a straight line
        if np.isclose(m1, m2, atol=1e-5):
            return 180.0

        # Calculate intersection point
        x_int = (b2 - b1) / (m1 - m2)
        y_int = m1 * x_int + b1
        p_intersect = np.array([x_int, y_int])

        # Vectors from intersection to the centroids
        v1 = np.mean(group1, axis=0) - p_intersect
        v2 = np.mean(group2, axis=0) - p_intersect

        mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if mag1 == 0 or mag2 == 0:
            return 180.0

        cos_theta = np.clip(np.dot(v1, v2) / (mag1 * mag2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    except (np.RankWarning, ValueError):
        return 180.0


def calculate_oriented_angle(x_group: NDArray, y_group: NDArray) -> float:
    """
    Calculates the interior angle at vertex b (index 1).
    Assumes y_group has already been normalized to the x_group scale.
    """
    # 1. Define vectors from vertex P1
    ax, ay = x_group[0] - x_group[1], y_group[0] - y_group[1]
    bx, by = x_group[2] - x_group[1], y_group[2] - y_group[1]

    # 2. Magnitudes (using pre-normalized components)
    mag_a = np.sqrt(ax ** 2 + ay ** 2)
    mag_b = np.sqrt(bx ** 2 + by ** 2)

    # Safety check for overlapping points
    if mag_a == 0 or mag_b == 0:
        return 180.0

    # 3. Dot product and angle calculation
    dot_product = (ax * bx) + (ay * by)
    cos_theta = np.clip(dot_product / (mag_a * mag_b), -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_theta)))


def get_rolling_angles(data_x: NDArray, data_y: NDArray) -> NDArray:
    """
    Rolling window through data using 3-point vertex logic.
    Normalizes the Y array globally based on the absolute magnitude
    of the X and Y ranges before iterating.
    """
    n = len(data_y)

    # 1. Assess absolute magnitude of the global ranges
    # This evaluates the entire trace to establish a single, consistent aspect ratio
    x_range = np.abs(np.max(data_x) - np.min(data_x))
    y_range = np.abs(np.max(data_y) - np.min(data_y))

    # 2. Apply Normalization to the entire Y array
    # If the array is perfectly flat, we skip division to avoid NaN
    if y_range > 0 and x_range > 0:
        scale_factor = y_range / x_range
        norm_y = data_y / scale_factor
    else:
        norm_y = data_y

    # 3. Initialize array with 180.0 (straight line baseline)
    angles = np.full(n, 180.0, dtype=np.float64)

    # 4. Rolling window loop over the normalized data
    for i in range(1, n - 1):
        angles[i] = calculate_oriented_angle(
                data_x[i - 1: i + 2],
                norm_y[i - 1: i + 2]
                )

    return angles


def sort_vectors_by_first(vec_primary: NDArray, vec_secondary: NDArray) -> tuple[NDArray, NDArray]:
    """
    Sorts two vectors based on the values of the first vector in ascending order.

    Parameters:
    vec_primary: The vector that determines the order (e.g., data_x).
    vec_secondary: The vector to be reordered alongside the primary (e.g., data_y).

    Returns:
    tuple: (sorted_primary, sorted_secondary)
    """
    # Get the indices that would sort the primary vector
    sort_indices = np.argsort(vec_primary)

    # Apply those indices to both vectors
    sorted_primary = vec_primary[sort_indices]
    sorted_secondary = vec_secondary[sort_indices]

    return sorted_primary, sorted_secondary
