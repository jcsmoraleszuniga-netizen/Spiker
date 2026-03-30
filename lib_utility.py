import copy
import functools
import json
# import pickle
import timeit
from itertools import pairwise, repeat
from pathlib import Path
from tkinter import filedialog
import numpy as np
from numba import njit, jit
from numpy import diff, exp, array, nanmean, std, arange, ones, sign, stack, trapz, where, zeros, mean, median, \
    percentile, \
    delete, \
    savetxt, gradient, concatenate, ndarray, dtype, signedinteger
from matplotlib import pyplot as plt
from numpy._typing import _32Bit, _64Bit
from scipy.signal import fftconvolve
from scipy.stats import stats, linregress
from typing import List, Tuple, Callable, Any, Optional, Iterable
from numpy.typing import NDArray
from scipy import optimize


def timing(function: Callable):
    @functools.wraps(function)
    def elapsed(*args):
        start = timeit.default_timer()
        print(f"\n{function.__name__:->30}")
        result = function(*args)
        total_time = timeit.default_timer() - start
        print(f"\n{function.__name__:+>40} {total_time = :.5f}")
        return result

    return elapsed


@njit
def vtp(value: float | np.floating, value_increment: float | np.floating) -> int | np.integer:
    """Transform Values To Points, given an increment"""
    return int(value / value_increment)


def ptv(value: int | np.integer, value_increment: float | np.floating) -> float | np.floating:
    """Transform Values To Points, given an increment"""
    return value * value_increment


@timing
@njit
def exp_decay(
        t: NDArray[np.floating], i0: np.floating, pk0: np.floating, t0: np.floating
        ) -> NDArray[np.floating]:
    return i0 + pk0 * exp(t / t0)


@timing
def exp_rise(
        t: NDArray[np.floating], top: np.floating, bott: np.floating, v50: np.floating, slope: np.floating
        ) -> NDArray[np.floating]:
    return top + (bott - top) / (1 + (v50 - t) / slope)


@timing
def linear(
        t: NDArray[np.floating], slope: np.floating, intercept: np.floating
        ) -> NDArray[np.floating]:
    return t * slope + intercept


@timing
def conv_vector(n_p: int, c_type: str = 'g', sharpness: int = 2) -> NDArray[np.floating]:
    conv: NDArray[np.floating] = array([])
    max_diff: float = 0.0006
    if n_p < 9:
        raise ValueError(f" Use a value of n_p >= 9 (minimum number of points). Try again.")
    else:
        if c_type == 'g':  # Gaussian convolution vector
            sd: np.floating = std(arange(n_p / sharpness))
            x = arange(n_p)
            conv = exp(-(((x - n_p / 2) / sd) ** 2) / 2) / (sd * np.sqrt(2 * np.pi))
            if np.abs(1 - np.sum(conv)) >= max_diff:
                print(f"{n_p = }")
                print(f"{sharpness = }")
                print(f"{sd = }")
                print(f"{x = }")
                print(f"{conv = }")
                print(f"{np.sum(conv) = }")
                raise ValueError(
                        f"Vector sum is significantly different from 1: {1 - np.sum(conv) = :.6f}. Use a value of n_p >= 9. "
                        f"\nMaximum difference accepted for the sum is: |1 - np.sum(conv)| < {max_diff}."
                        )
        elif c_type == 'f':  # Flat convolution vector
            conv = ones(n_p) / n_p
    return conv


# @timing
# @njit
# def crossing_point(
#         arr: NDArray[np.floating]
#         ) -> int | None:
#     """Returns the position where the array intersect with 0.0"""
#     sign_array: NDArray[np.floating] = sign(arr)
#     previous = sign_array[0]
#     for index, value in enumerate(sign_array):
#         if value - previous != 0:
#             return index
#         else:
#             previous = value
#     return None

# @timing
# @njit
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


@timing
def find_over_threshold(  # It's faster without @njit.
        response: NDArray[np.floating], threshold: NDArray[np.floating], direction: int
        ) -> NDArray[np.floating]:
    events_over_threshold: NDArray[np.floating] = array([])
    stacked: NDArray[np.floating] = stack((response, threshold), axis=0)
    if direction > 0:
        events_over_threshold = np.max(stacked, axis=0) - threshold
    elif direction < 0:
        events_over_threshold = np.min(stacked, axis=0) - threshold
    return where(np.abs(events_over_threshold) > 0, 1, 0)


@timing
@njit
def find_peaks(
        over_threshold: NDArray[np.floating], response: NDArray[np.floating], direction: int,
        search_width: float | np.floating, time_increment: float | np.floating
        ) -> NDArray[np.floating]:
    peaks: NDArray[np.floating] = zeros(len(over_threshold))
    half_width: int = max(1, vtp(search_width / 2, time_increment))
    quart_width: int = max(1, vtp(search_width / 4, time_increment))
    for pos in range(half_width + 1, len(over_threshold)):  # In case a peak is at "0" position

        if over_threshold[pos]:
            window = response[pos - quart_width: pos + half_width]
            match direction:
                case -1:
                    if np.min(window) == response[pos]:
                        peaks[pos - quart_width: pos + half_width] = 0
                        peaks[pos] = 1
                case 1:
                    if np.max(window) == response[pos]:
                        peaks[pos - quart_width: pos + half_width] = 0
                        peaks[pos] = 1
                case _:
                    print("Wrong direction")
    return peaks


@timing
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


@timing
def find_outliers(arr: NDArray[np.floating]) -> ndarray[Any, dtype[signedinteger[Any] | dtype]]:
    q1: np.floating = percentile(arr, 25)
    q3: np.floating = percentile(arr, 75)
    iqr: np.floating = q3 - q1
    threshold = 1.5 * iqr
    return where((arr < q1 - threshold) | (arr > q3 + threshold))[0]


@timing
def remove_outlier(array_2d: NDArray[np.floating]) -> NDArray[np.floating]:
    """Identify  the outliers in the 'y' axis of the array and then removes them with their respective 'x' values"""
    return delete(array_2d, find_outliers(array_2d.T[1]), axis=0)


@timing
def save(arr: iter, title_="save") -> None:
    """Pop up a file dialog to save the list of values"""
    files = [('All Files', '*.*'), ('CSV Files', '*.csv'), ('Text Document', '*.txt')]
    file_name = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=files, title=title_)
    savetxt(file_name, arr, delimiter=',')


@timing
def auto_save(arr: iter, file_name='default') -> None:
    """Save the list of values automatically"""
    savetxt(file_name, arr, delimiter=',')


# @timing
# @njit
def differentiate(arr: NDArray[np.floating], incr: np.floating | float) -> NDArray[np.floating]:
    return gradient(arr) / incr


@timing
def opt_linear(
        x_arr: NDArray[np.floating], y_arr: NDArray[np.floating]
        ) -> tuple[ndarray | Iterable | int | float, Any, Any, Any, Any]:
    # linear: t * slope + intercept
    return optimize.curve_fit(linear, x_arr, y_arr)  # returns the parameters of the fitting


# @timing
# def opt_expdec(
#         time: NDArray[np.floating], voltage: NDArray[np.floating], bounds=([- 10, 0, 0], [20, 50, 20])
#         ) -> tuple[ndarray | Iterable | int | float, Any, Any, Any, Any]:
#     # bounds = ([- 10, 0, 0], [20, 50, 20])  # Bounds to initialize the fitting process
#     return optimize.curve_fit(exp_decay, time, voltage, bounds=bounds)


# @timing
# def extender(
#         x_segm: NDArray[np.floating], t_segm: NDArray[np.floating], t_common: NDArray[np.floating]
#         ) -> NDArray[np.floating]:
#     back_segment: NDArray[np.floating] = t_common[:where(t_segm[0] == t_common)[0][0]]
#     front_segment: NDArray[np.floating] = t_common[where(t_segm[-1] == t_common)[0][0] + 1:]
#     extd_evt = concatenate((zeros(len(back_segment)), x_segm), axis=None)
#     extd_evt = concatenate((extd_evt, zeros(len(front_segment))), axis=None)
#     return extd_evt

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


@timing
def loop(arr1: NDArray[np.floating], arr2: NDArray[np.floating]) -> NDArray[np.floating]:
    return array(
            [
                    arr1[pos - 1: pos + 2]
                    for pos in range(len(arr2))
                    if arr2[pos] and len(arr2[pos - 1: pos + 2]) == 3
                    ]
            ).flatten()


@timing
def get_peaks_arr(time: NDArray[np.floating], peaks: NDArray[np.floating]) -> NDArray[np.floating]:
    return array(concatenate(([loop(time, peaks)], [loop(peaks, peaks)]), axis=0).T)


@timing
def make_sections(
        start: int | float = 0, total: int | float = 1800, interval: int | float = 600
        ) -> List[Tuple[int, int]]:
    start = int(start)
    total = int(total)
    interval = int(interval)
    points = [i for i in range(start, total + 1, interval)]
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


@timing
def apply_by_continuous(
        function: callable,
        arr: NDArray[NDArray[np.floating]],
        increment: dict,
        ) -> NDArray[NDArray[np.floating]]:
    """Calculates the average value for a certain increment in time.
    If no values are found in that increment then the value is 0.0"""
    local_vtp = vtp
    d_t = round(float(arr[0][1] - arr[0][0]), 5)
    increment["start"] = arr[0][0]
    increment["end"] = arr[0][-1]
    interv = [p_time for p_time in arange(increment["start"], increment["end"], increment["increment"])]
    return np.array(
            [
                    [
                            sum(pair) / 2,
                            function(arr[1][local_vtp(pair[0] - arr[0][0], d_t):local_vtp(pair[1] - arr[0][0], d_t)])
                            ]
                    for pair in pairwise(interv)
                    ]
            ).T


@timing
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


@timing
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


@timing
def average_by(
        arr: NDArray[NDArray[np.floating]],
        increment: dict,
        continuous: bool = False
        ) -> NDArray[NDArray[np.floating]]:
    """Calculates the average value for a certain increment in time.
    If no values are found in that increment then the average value is 0.0"""
    return apply_by(nanmean, arr, increment, continuous)


@timing
def count_ones(arr: NDArray[np.floating]) -> int:
    return np.count_nonzero(arr == 1)


@timing
def event_count(
        arr: NDArray[NDArray[np.floating]],
        increment: dict,
        continuous: bool = False
        ) -> NDArray[NDArray[np.floating]]:
    """Counts the number of events in a certain increment in time.
    If no values are found in that increment then the count is set as 0.0"""
    return apply_by(len, arr, increment, continuous)


@timing
def event_pr(arr: NDArray[NDArray[np.floating]], increment: dict) -> NDArray[NDArray[np.floating]]:
    """Calculates the probability of an event for a certain increment in time.
    If no values are found in that increment then the probability is set as 0.0"""
    # Float conversion for effective normalization
    counts: NDArray[NDArray[np.floating]] = event_count(arr, increment).astype(np.float32)
    counts[1] = counts[1] / np.sum(counts[1])  # Normalization of the values
    return counts


@timing
def intervals(increment: dict) -> NDArray[NDArray[np.floating]]:
    ratio = increment["end"] / increment["increment"]
    n_times = round(ratio, 0)
    if n_times - ratio >= 0:
        tail = n_times
    else:
        tail = n_times + 1
    arr = [p_time for p_time in arange(0, tail * increment["increment"] + 1, increment["increment"])]
    if arr[0] != increment["start"]:
        arr[0] = increment["start"]
    if arr[-1] != increment["end"]:
        arr[-1] = increment["end"]
    return np.array(arr)


@timing
def event_fr(arr: NDArray[NDArray[np.floating]], increment: dict) -> NDArray[NDArray[np.floating]]:
    """Calculates the frequency of an event for a certain increment in time.
    If no values are found in that increment then the frequency is set as 0.0"""
    # Float conversion for effective normalization
    counts: NDArray[NDArray[np.floating]] = event_count(arr, increment).astype(np.float32)
    interv: NDArray[NDArray[np.floating]] = intervals(increment).astype(np.float32)
    interv = diff(interv)
    counts[1] = counts[1] / interv
    return counts


@timing
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


@timing
def save_plot(
        values, name_params: dict, units: str, plot_increment: dict, bins: int, func: callable, plot=True
        ) -> None:
    """Saves the data and plots it."""
    if callable(func):
        func_values = func(values, plot_increment)
        func_name = func.__name__
    else:
        func_values = values
        func_name = ""
    out_name = name_params["file_parent"] + make_name(
            name_params["common_name"] + [name_params["sweep_number"]] + [name_params["parameter"]]
            )
    out_name_mean = name_params["file_parent"] + make_name(
            name_params["common_name"] + [name_params["sweep_number"]] + [name_params["parameter"], func_name]
            )
    print(f"{out_name = }")
    print(f"{out_name_mean = }")
    if callable(func):
        auto_save(values.T, out_name)
        auto_save(func_values.T, out_name_mean)
    else:
        auto_save(values.T, out_name)
    mean_val = mean(values[1])
    median_val = median(values[1])
    if plot:
        # Plotting time course of values
        plt.figure(figsize=(3, 2.5))
        plt.rcParams.update({'font.size': 8})
        plt.plot(values[0], values[1], "r+", label=name_params["parameter"])
        plt.plot(
                func_values[0],
                func_values[1],
                'bo', label=f"{func_name} {name_params["parameter"]}", alpha=0.5, markersize=10
                )
        plt.axhline(0, color='g', linestyle='dashed', linewidth=1)
        plt.axhline(mean_val, color='r', linestyle='dashed', linewidth=1, label=f"{mean_val = :.4f}{units}")
        plt.axhline(median_val, color='k', linestyle='dashed', linewidth=1, label=f"{median_val = :.4f}{units}")
        plt.title(f"{name_params["analysis_type"]} time course. {len(values[1])} events.")
        plt.legend()
        plt.show(block=False)
        # Histogram of values
        plt.figure(figsize=(3, 2.5))
        plt.hist(values[1], bins)
        plt.axvline(0, color='g', linestyle='dashed', linewidth=1)
        plt.axvline(mean_val, color='r', linestyle='dashed', linewidth=1, label=f"{mean_val = :.4f}{units}")
        plt.axvline(median_val, color='k', linestyle='dashed', linewidth=1, label=f"{median_val = :.4f}{units}")
        plt.title(f"{name_params["parameter"]} ({name_params["analysis_type"]}). {len(values[1])} events.")
        plt.legend(loc='upper right')
        plt.show(block=False)


@timing
def get_previous_folder(context: str = "") -> Optional[str]:
    """Retrieves the previously opened folder from a file."""
    try:
        with open(context + "previous_folder.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


@timing
def save_previous_folder(folder_path: str, context: str = "") -> None:
    """Saves the given folder path to a file."""
    name = context + "previous_folder.txt"
    with open(name, "w") as f:
        f.write(folder_path)


@timing
def mse(arr1, arr2) -> float:
    """Calculates minimum square error for two arrays of the same length"""
    return np.mean(np.square(arr1 - arr2))


@timing
# @njit
def prev_change(der_test_resp: np.ndarray, prev: float = 0.0) -> np.ndarray:
    """
    Optimizes the given Python loop using NumPy for faster execution.
    """
    der_test_resp = np.array(der_test_resp)  # Convert to NumPy array if it's a list
    # Create a shifted array to check the previous element
    shifted_resp = np.concatenate(
            ([prev], der_test_resp[:-1])
            )  # prepends a zero to the array and removes the last element.
    # Find the indices where the previous element is not zero
    indices_to_zero = shifted_resp != prev
    # Set the corresponding elements in the original array to zero
    der_test_resp[indices_to_zero] = prev

    return der_test_resp


@timing
def down_sample_function(arr, down_sample=10):
    return arr[::down_sample]


@timing
def down_sample_function_t(arr, accept_mask, down_sample=10):
    # 1. Create a mask for the downsampling (every Nth element)
    # np.arange creates indices, then we check the modulo
    periodic_mask = (np.arange(len(arr)) % down_sample == 0)
    # 2. Combine with the accept_mask using a bitwise OR (|)
    # This keeps elements where val == 1 OR the index is a multiple of down_sample
    combined_mask = (accept_mask == 1) | periodic_mask
    # 3. Use boolean indexing to filter the array
    return arr[combined_mask]


# @timing
# def smoothing(resp, points, repetitions=1, sharpness=4):
#     # Sharpness was tested for low weight tails
#     # 1. Generate your kernel
#     kernel = conv_vector(points, 'g', sharpness)
#     # 2. Determine a safe padding length (the length of the kernel is usually plenty)
#     pad_len = len(kernel)
#     # 3. Pad the response with its own edge values to prevent diving to zero
#     padded_resp = np.pad(resp, pad_len, mode='edge')
#     if repetitions > 1:
#         for _ in repeat(None, repetitions):
#             # Shifting to the left self.resp = np.append(response[1:], [0])
#             padded_resp = np.append(
#                     # np.convolve is better for short arrays
#                     # fftconvolve is better for long arrays
#                     # 4. Convolve the padded array (this will be longer than your original signal)
#                     fftconvolve(padded_resp, kernel, mode='same')[1:],
#                     # np.convolve(padded_resp, kernel , mode='same')[1:],
#                     [0]
#                     )
#     else:
#         padded_resp = fftconvolve(padded_resp, kernel, mode='same')
#     # 5. Slice off exactly the amount you padded to return to shape (4853,)
#     return padded_resp[pad_len:-pad_len]

@timing
def smoothing(resp, points, repetitions=1, sharpness=4):
    # 1. Calculate the effective points (width) for a single pass
    # Using the property: sigma_total = sigma * sqrt(n)
    eff_points = points * np.sqrt(repetitions)
    # 2. Generate the single, wider kernel
    kernel = conv_vector(eff_points, 'g', sharpness)
    # 3. Padding logic (remains the same to prevent edge diving)
    pad_len = len(kernel)
    padded_resp = np.pad(resp, pad_len, mode='edge')
    # 4. Single Convolution pass
    result = fftconvolve(padded_resp, kernel, mode='same')
    # 5. Slice and return
    return result[pad_len:-pad_len]


# @timing
# def reset_array(arr: np.ndarray, point: int | np.ndarray, value: float = 1.0) -> np.ndarray:
#     arr = arr * 0.0  # Makes everything 0.0
#     arr[point] = value  # Makes the point the only peak
#     return arr

# @timing
def reset_array(arr: np.ndarray, point: int | np.ndarray, value: float = 1.0) -> np.ndarray:
    """Resets the array to zero and sets specific indices to a value in-place."""
    # .fill(0) is the fastest way to wipe an existing array in-place
    arr.fill(0.0)
    # Assign the value (works for both a single int or an array of indices)
    arr[point] = value
    return arr


@timing
def split_position(arr: np.ndarray, direction: int) -> signedinteger[_32Bit | _64Bit] | None:
    match direction:
        case -1:
            return np.argmin(arr)
        case 1:
            return np.argmax(arr)
        case _:
            return None


@timing
def remove_shift(arr_base, arr_resp):
    """Remove shift of the response over time."""
    # pr_lin = opt_linear(arr_base[0], arr_base[1])
    slope, intercept, r_value, p_value, std_err, intercept_stderr = lin_fit(arr_base[1], arr_base[0])
    # arr_resp[1] -= linear(arr_resp[0], *pr_lin[0])
    arr_resp[1] -= linear(arr_resp[0], slope, intercept)
    return arr_resp


@timing
def make_name(lst: list, extension='.csv') -> str:
    lst_str = [str(i) for i in lst]
    return "_".join(lst_str) + extension


@timing
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


@timing
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
    The exponential form has to be: I(t) = i0 + pk0 * exp(-t / t0)
    The linear form is: Ln(I(t) - i0) = Ln(pk0) - t/t0"""
    if len(arr):
        # y: np.ndarray = np.array([])
        match direction:
            case 1:  # positive going
                i_0 = np.min(arr)
                y = arr - i_0
            case -1:  # negative going
                i_0 = np.max(arr)
                y = -(arr - i_0)
            case _:
                raise ValueError("Wrong direction")
        return np.log(y + 1.0)
    else:
        raise ValueError("Array is empty")


# @timing
# def lin_fit(y_var, x_var):
#     """Fits data to a linear relation.
#     y_var(x_var) = intercept + slope * x_var,
#     returns slope, intercept, r_value, p_value, std_err and intercept_stderr"""
#     result = linregress(x_var, y_var)
#     return result.slope, result.intercept, result.rvalue, result.pvalue, result.stderr, result.intercept_stderr


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
    fit_i_0 = np.average(response - fit_pk0 * np.exp(time / fit_t0))
    return fit_i_0, fit_pk0, fit_t0, r_value  # i0, pk0, t0, r


@timing
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
