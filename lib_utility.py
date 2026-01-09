import copy
import functools
import json
# import pickle
import timeit
from itertools import repeat
from pathlib import Path
from tkinter import filedialog
import numpy as np
# from numba import njit
from numpy import exp, array, nanmean, std, arange, ones, sign, flipud, stack, trapz, where, zeros, mean, median, \
    percentile, \
    delete, \
    savetxt, gradient, concatenate, ndarray, dtype, signedinteger
import lib_gui as gui
from matplotlib import pyplot as plt
from numpy._typing import _32Bit, _64Bit
from scipy.stats import stats
from typing import List, Tuple, Callable, Any, Optional, Iterable
from numpy.typing import NDArray
from scipy import optimize, integrate


# from numba import njit, vectorize


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


# @timing
# @njit
def vtp(value: float | np.floating, value_increment: float | np.floating) -> int | np.integer:
    """Transform Values To Points, given an increment"""
    return int(value / value_increment)


def ptv(value: int | np.integer, value_increment: float | np.floating) -> float | np.floating:
    """Transform Values To Points, given an increment"""
    return value * value_increment


@timing
def exp_decay(
        t: NDArray[np.floating], i0: np.floating, pk0: np.floating, t0: np.floating
        ) -> NDArray[np.floating]:
    return i0 + pk0 * exp(-t / t0)


@timing
def exp_rise(
        t: NDArray[np.floating], top: np.floating, bott: np.floating, v50: np.floating, slope: np.floating
        ) -> NDArray[np.floating]:
    return top + (bott - top) / (1 + (v50 - t) / slope)


# @timing
def linear(
        t: NDArray[np.floating], slope: np.floating, intercept: np.floating
        ) -> NDArray[np.floating]:
    return t * slope + intercept


# @njit
def conv_vector(n_p: int, c_type: str = 'g', sharpness: int = 2) -> NDArray[np.floating]:
    conv: NDArray[np.floating] = array([])
    max_diff: float = 0.0006
    if c_type == 'g':  # Gaussian convolution vector
        sd: np.floating = std(arange(n_p / sharpness))
        x = arange(n_p)
        conv = exp(-(((x - n_p / 2) / sd) ** 2) / 2) / (sd * np.sqrt(2 * np.pi))
        if np.abs(1 - np.sum(conv)) >= max_diff:  # TODO assess if it's necessary to use numpy versions
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


# @njit
def crossing_point(
        arr: NDArray[np.floating]
        ) -> int | None:
    """Returns the position where the array intersect with 0.0"""
    sign_array: NDArray[np.floating] = sign(arr)
    previous = sign_array[0]
    for index, value in enumerate(sign_array):
        if value - previous != 0:
            return index
        else:
            previous = value
    return None


@timing
def find_over_threshold(
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
# @njit
def find_peaks(
        over_threshold: NDArray[np.floating], response: NDArray[np.floating], direction: int,
        search_width: float | np.floating, time_increment: float | np.floating
        ) -> NDArray[np.floating]:
    peaks: NDArray[np.floating] = zeros(len(over_threshold))
    half_width: int = max(1, vtp(search_width / 2, time_increment))
    prev = 0
    for pos in range(half_width + 1, len(over_threshold)):  # In case a peak is at "0" position
        # if over_threshold[pos] and (pos > half_width + prev):
        if over_threshold[pos]:
            window = response[pos - half_width: pos + half_width]
            match direction:
                case -1:
                    if np.min(window) == response[pos]:
                        peaks[pos - half_width: pos + half_width] = 0
                        peaks[pos] = 1
                        # prev = pos
                case 1:
                    if np.max(window) == response[pos]:
                        peaks[pos - half_width: pos + half_width] = 0
                        peaks[pos] = 1
                        # prev = pos
                case _:
                    print("Wrong direction")
            # if direction < 0 and (np.min(window) == response[pos]):
            #     peaks[pos] = 1
            #     prev = pos
            # elif direction > 0 and (np.max(window) == response[pos]):
            #     peaks[pos] = 1
            #     prev = pos
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


def differentiate(arr: NDArray[np.floating], incr: np.floating | float) -> NDArray[np.floating]:
    return gradient(arr) / incr


# @timing
def opt_linear(
        x_arr: NDArray[np.floating], y_arr: NDArray[np.floating]
        ) -> tuple[ndarray | Iterable | int | float, Any, Any, Any, Any]:
    # linear: t * slope + intercept
    return optimize.curve_fit(linear, x_arr, y_arr)  # returns the parameters of the fitting


@timing
def opt_expdec(
        time: NDArray[np.floating], voltage: NDArray[np.floating], bounds=([- 10, 0, 0], [20, 50, 20])
        ) -> tuple[ndarray | Iterable | int | float, Any, Any, Any, Any]:
    # bounds = ([- 10, 0, 0], [20, 50, 20])  # Bounds to initialize the fitting process
    return optimize.curve_fit(exp_decay, time, voltage, bounds=bounds)


def extender(
        x_segm: NDArray[np.floating], t_segm: NDArray[np.floating], t_common: NDArray[np.floating]
        ) -> NDArray[np.floating]:
    back_segment: NDArray[np.floating] = t_common[:where(t_segm[0] == t_common)[0][0]]
    front_segment: NDArray[np.floating] = t_common[where(t_segm[-1] == t_common)[0][0] + 1:]
    extd_evt = concatenate((zeros(len(back_segment)), x_segm), axis=None)
    extd_evt = concatenate((extd_evt, zeros(len(front_segment))), axis=None)
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
    print(f"{points = }")
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


@timing
# @njit()
def apply_by_continuous(
        function: callable,
        arr: NDArray[NDArray[np.floating]],
        increment: int | float = 60
        ) -> NDArray[NDArray[np.floating]]:
    """Calculates the average value for a certain increment in time.
    If no values are found in that increment then the value is 0.0"""
    local_vtp = vtp
    d_t = round(float(arr[0][1] - arr[0][0]), 5)  # TODO find a better way to enter the minimal interval
    half_range = int(local_vtp(increment, d_t) / 2)
    return np.array(
            [
                    [
                            p_time,
                            function(
                                    arr[1][
                                    local_vtp(p_time - arr[0][0], d_t) - half_range:
                                    local_vtp(p_time - arr[0][0], d_t) + half_range
                                    ]
                                    )
                            ]
                    for p_time in arange(arr[0][0] + (increment / 2), arr[0][-1], increment)
                    ]
            ).T


@timing
def apply_by_discrete(
        function: callable,
        arr: NDArray[NDArray[np.floating]],
        increment: int | float = 60,
        ) -> NDArray[NDArray[np.floating]]:
    """Calculates the average value for a certain increment in time.
    If no values are found in that increment then the value is 0.0"""
    l_function: list[list[float]] = []
    prev_time: float = 0.0
    for pres_time in arange(increment / 2, arr[0][-1] + increment, increment):
        section = where((prev_time <= arr[0]) & (arr[0] < (pres_time + increment / 2)))
        if prev_time > arr[0][-1]:
            print(f"        End of the array!!")
            break
        elif len(arr[0][section]):
            l_function.append([pres_time, function(arr[1][section])])
        else:
            l_function.append([pres_time, 0.0])
        prev_time = pres_time + increment / 2

    return array(l_function).T


@timing
def apply_by(
        function: callable,
        arr: NDArray[NDArray[np.floating]],
        increment: int | float = 60,
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
        arr: NDArray[NDArray[np.floating]], increment: int = 60, continuous: bool = False
        ) -> NDArray[NDArray[np.floating]]:
    """Calculates the average value for a certain increment in time.
    If no values are found in that increment then the average value is 0.0"""
    return apply_by(nanmean, arr, increment, continuous)


# @timing
def count_ones(arr: NDArray[np.floating]) -> int:
    return np.count_nonzero(arr == 1)


@timing
def event_count(
        arr: NDArray[NDArray[np.floating]], increment: int = 60, continuous: bool = False
        ) -> NDArray[NDArray[np.floating]]:
    """Counts the number of events in a certain increment in time.
    If no values are found in that increment then the count is set as 0.0"""
    return apply_by(len, arr, increment, continuous)


@timing
def event_pr(arr: NDArray[NDArray[np.floating]], increment: int = 60) -> NDArray[NDArray[np.floating]]:
    """Calculates the probability of an event for a certain increment in time.
    If no values are found in that increment then the probability is set as 0.0"""
    # Float conversion for effective normalization
    counts: NDArray[NDArray[np.floating]] = event_count(arr, increment).astype(np.float32)
    counts[1] = counts[1] / np.sum(counts[1])  # Normalization of the values
    return counts


@timing
def event_fr(arr: NDArray[NDArray[np.floating]], increment: int = 60) -> NDArray[NDArray[np.floating]]:
    """Calculates the frequency of an event for a certain increment in time.
    If no values are found in that increment then the frequency is set as 0.0"""
    # Float conversion for effective normalization
    counts: NDArray[NDArray[np.floating]] = event_count(arr, increment).astype(np.float32)
    counts[1] = counts[1] / increment  # Normalization of the values
    return counts


@timing
def event_aft(arr: NDArray[NDArray[np.floating]], increment: int = 60) -> NDArray[NDArray[np.floating]]:
    """Counts the number of events for a certain increment in time, and divides by the average of the events values.
    If no values are found in that increment then the count is set as 0.0"""
    # Float conversion for effective normalization
    counts: NDArray[NDArray[np.floating]] = event_count(arr, increment).astype(np.float32)
    averages: NDArray[NDArray[np.floating]] = average_by(arr, increment).astype(np.float32)
    inter_events = np.append([0.0], np.diff(arr[0]))
    intervals: NDArray[NDArray[np.floating]] = average_by(np.array([arr[0], inter_events]), increment).astype(
            np.float32
            )
    counts[1] = counts[1] / intervals[1]  # Count/Average
    counts = (counts.T[~np.isnan(counts[0]) & ~np.isnan(counts[1])]).T  # Removes pairs that present np.nan values
    return counts


@timing
def save_plot(values, name_params: dict, units: str, averaging: int, bins: int, func: callable, plot=True) -> None:
    """Saves the data and plots it."""
    if callable(func):
        func_values = func(values, averaging)
        print(f"{values.shape=}")
        print(f"{values=}")
        func_name = func.__name__
    else:
        func_values = values
        print(f"{values.shape=}")
        print(f"{values=}")
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


# def save_plot_func(
#         values: NDArray[NDArray[np.floating]], func: Callable,
#         out_name_func_val: str, func_name: str, value_label: str, averaging: int
#         ) -> None:
#     """It Shows where are occurring most of the events"""
#     plt.figure()
#     plt.plot(values[0], values[1], "b", label=value_label)
#     fun_bin: NDArray[NDArray[np.floating]] = func(values, averaging)
#     plt.plot(fun_bin[0], fun_bin[1], 'ro', label=f"{func_name} of {value_label}", markersize=10, alpha=0.5)
#
#     plt.title(f"{value_label} time course. {len(values[1])} events.")
#     plt.legend(loc='upper right')
#     plt.show(block=False)
#     auto_save(fun_bin.T, out_name_func_val)


def get_previous_folder(context: str = "") -> Optional[str]:
    """Retrieves the previously opened folder from a file."""
    try:
        with open(context + "previous_folder.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def save_previous_folder(folder_path: str, context: str = "") -> None:
    """Saves the given folder path to a file."""
    name = context + "previous_folder.txt"
    with open(name, "w") as f:
        f.write(folder_path)


def mse(arr1, arr2) -> float:
    """Calculates minimum square error for two arrays of the same length"""
    return np.mean(np.square(arr1 - arr2))


@timing
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


# @timing
def smoothing(resp, points, repetitions=1, sharpness=4):
    # Sharpness was tested for low weight tails
    for _ in repeat(None, repetitions):
        # Shifting to the left self.resp = np.append(response[1:], [0])
        resp = np.append(
                # np.convolve is better for short arrays
                np.convolve(resp, conv_vector(points, 'g', sharpness), mode='same')[1:],
                [0]
                )
    return resp


def reset_array(arr: np.ndarray, point: int | np.ndarray, value: float = 1.0) -> np.ndarray:
    arr = arr * 0.0  # Makes everything 0.0
    arr[point] = value  # Makes the point the only peak
    return arr


def split_position(arr: np.ndarray, direction: int) -> signedinteger[_32Bit | _64Bit] | None:
    match direction:
        case -1:
            return np.argmin(arr)
        case 1:
            return np.argmax(arr)
        case _:
            return None


def remove_shift(arr_base, arr_resp):
    """Remove shift of the response over time."""
    pr_lin = opt_linear(arr_base[0], arr_base[1])
    arr_resp[1] -= linear(arr_resp[0], *pr_lin[0])
    return arr_resp


def make_name(lst: list, extension='.csv') -> str:
    lst_str = [str(i) for i in lst]
    return "_".join(lst_str) + extension


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
def manage_settings(const_file: str, const: dict):
    """Opens a stored dictionary 'const_file' to be modified. If there is no 'const_file' stored,
    uses 'const' as a default to begin the modification"""
    loaded_dict = load_dict(const_file, const)
    updated_const: dict = gui.ConstDialog(loaded_dict, "Event detection")
    if updated_const:
        loaded_dict.update(updated_const)
        print("Constants updated.")
    else:
        print("Constants dialog canceled.")
    save_dict(const_file, loaded_dict)
    return loaded_dict


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


def calculate_area(resp_time, resp):
    # return integrate.simpson(resp, resp_time)
    return trapz(resp, resp_time)


def correct_bound(value):
    if value <= 0:
        return 0
    else:
        return value


def exp_to_lin(arr: np.ndarray, direction: int) -> np.ndarray:
    """Transforms an exponential decay curve to a linear curve.
    The exponential form has to be: I(t) = i0 + pk0 * exp(-t / t0)"""
    if len(arr):
        y = np.array([])
        match direction:
            case 1:  # positive going
                i_0 = np.min(arr)
                y = arr - i_0
            case -1:  # negative going
                i_0 = np.max(arr)
                y = -(arr - i_0)
            case _:
                print(f"Wrong direction")
        return np.log(y + 1.0)
    else:
        raise ValueError("Array is empty")


def exp_fit(response, time, direction):
    """Fits data to an exponential decay.
    I(t) = i0 + pk0 * exp(-t / t0), returns i0, pk0, t0"""
    i_0 = response[-1]
    lin_resp = exp_to_lin(response, direction)
    t_segm_arr = np.vstack([time, np.ones(len(time))]).T
    params = np.linalg.lstsq(t_segm_arr, lin_resp, rcond=None)
    inv_t0, lin_pk0 = params[0]
    fit_i_0 = (i_0 - direction)
    fit_pk0 = np.exp(lin_pk0) * direction
    fit_t0 = 1.0 / inv_t0
    pearson_r = np.corrcoef(time, lin_resp)[0, 1]
    return fit_i_0, fit_pk0, fit_t0, pearson_r  # i0, pk0, t0


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
