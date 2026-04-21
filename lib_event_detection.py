import copy
import gc
import warnings
from itertools import pairwise, repeat
from typing import List, Any

from PyQt6.QtCore import QEventLoop
from numpy import ndarray, dtype, flipud
from pyabf import ABF
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import find_objects, label
from scipy.signal import fftconvolve, correlate

from lib_utility import (
    conv_vector, down_sample_function_t, exp_decay, exp_fit, get_rolling_angles, lin_fit, parabolic_fit, remove_nan,
    sort_vectors_by_first, vtp, differentiate,
    find_over_threshold,
    find_peaks,
    crossing_point, extender, apply_by, mse, timing, prev_change, down_sample_function, smoothing, reset_array,
    remove_shift, file_info, calculate_area, correct_bound
    )
from copy import copy as cp_copy
from scipy import fft
from matplotlib.ticker import FuncFormatter


@timing
def copy_instance(ori_inst: 'EvtPro', direction: int) -> 'EvtPro':
    temp_obj = cp_copy(ori_inst)  # Instantiation of the recordings
    temp_obj.direction = direction
    return temp_obj


@timing
def make_instances(ori_inst: 'EvtPro', direction, start: float, end: float, copies: int) -> List['EvtPro']:
    temp_obj = cp_copy(ori_inst)  # Instantiation of the recordings
    temp_obj.section(start, end)
    if temp_obj.mode == "continuous":
        temp_obj.find_pulses()
    return [copy_instance(temp_obj, direction) for _ in repeat(None, copies)]


@timing
def _get_sparse_indices(arr):
    """
    Helper function to extract indices of 1s, their immediate neighbors,
    and the start/end bounds. Reduces mostly-zero arrays by 99%+.
    """
    ones = np.where(arr == 1)[0]
    if ones.size == 0:
        return np.array([0, len(arr) - 1])

    # Combine the 1s, the index before, the index after, and the bookends
    idx = np.concatenate((ones - 1, ones, ones + 1, [0, len(arr) - 1]))

    # Ensure no out-of-bounds indices, then return unique (sorted) indices
    return np.unique(np.clip(idx, 0, len(arr) - 1))


@timing
def plot_rec(rec, der, title="No Title.", values=(0.0, 0.0), mode="full", factor=1.0):
    """
    Displays a plot of the original data.
    Optimized to dynamically decimate sparse binary peak arrays and downsample noise.
    """
    fig, ax = plt.subplots(figsize=(15, 7.5))

    ax.axhline(y=0.0, color="k", linestyle='--')
    for x_value in values:
        ax.axvline(x=x_value, color="r", linestyle='--')

    # Pre-calculate scales
    der_scale = rec.t_delta * factor
    p_scale = 1.5 * factor
    dp_scale = 1.0 * factor
    zp_scale = 0.5 * factor

    # Plot full resolution for main responses
    ax.plot(rec.time, rec.resp, "k", linewidth=3.0, alpha=0.75)
    ax.plot(der.time, der.resp * der_scale, "b", linewidth=3.0, alpha=0.75)

    # --- Squeeze the binary peak arrays ---
    idx_p = _get_sparse_indices(rec.peaks)
    ax.plot(rec.time[idx_p], (rec.peaks * p_scale)[idx_p], "r", linewidth=1.0, alpha=0.75)

    idx_dp = _get_sparse_indices(rec.der_peaks)
    ax.plot(rec.time[idx_dp], (rec.der_peaks * dp_scale)[idx_dp], "g", linewidth=1.0, alpha=0.75)

    # Zero pass remains untouched as requested
    ax.plot(rec.time, rec.zero_pass * zp_scale, "b", linewidth=1.0, alpha=0.3)

    # --- Downsample ONLY the noise arrays by 10 using [::10] ---
    ax.plot(rec.time[::10], rec.peak_noise[::10], "k:")
    ax.plot(der.time[::10], (der.peak_noise * der_scale)[::10], "b:")

    ax.set_title(title)

    # 1. Show the window without triggering the global block
    plt.show(block=False)
    fig.canvas.draw()

    # 2. Pass 'event' and use 'event.canvas' to avoid closure circular reference
    def on_close(event):
        event.canvas.stop_event_loop()

    cid = fig.canvas.mpl_connect('close_event', on_close)

    # 3. Start Matplotlib's internal loop
    fig.canvas.start_event_loop(timeout=0)

    # ---------------------------------------------------------
    # SCORCHED EARTH RAM CLEARING
    # ---------------------------------------------------------
    fig.canvas.mpl_disconnect(cid)
    fig.clear()
    plt.close(fig)
    plt.close('all')

    # E. Explicitly delete the local variables holding the plot objects
    del fig, ax

    # F. Force Python's Garbage Collector to reclaim RAM
    gc.collect()


class base:

    def __init__(self):
        print(f"Initializing {self = }")
        self.mode = ""
        self.t_delta = 0.0  # minimal interval
        self.time = np.array([])  # time
        self.sweeps = np.array([])  # sweeps
        self.resp = np.array([])  # response
        self.cdac = np.array([])  # DAC

    @timing
    def _get_resp(self):
        raise NotImplementedError

    @timing
    def _get_time(self):
        raise NotImplementedError

    @timing
    def _initialize(self):
        raise NotImplementedError


class abf_numpy(ABF, base):
    """Transforms ABF files to a numpy inheriting class"""

    def __init__(self, path_to_file, initialize=True, location=0):
        super().__init__(path_to_file)
        print(f"Initializing {self = }")
        self.mode = "continuous"
        if initialize:
            self._initialize(location)

    @timing
    def _get_resp(self, location=0):
        self.resp = self.data[location]

    @timing
    def _get_time(self):
        self.setSweep(0)
        self.t_delta: np.floating = np.round(self.sweepX[1] - self.sweepX[0], decimals=8)
        print(f"{self.t_delta = }")
        # RAM FIX: Generate the continuous time directly in one shot
        total_points = len(self.sweepX) * self.sweepCount
        self.time = np.arange(total_points) * self.t_delta

    @timing
    def _get_cdac(self):
        # RAM FIX: Use np.tile to repeat the array memory-efficiently
        self.cdac = np.tile(self.sweepC, self.sweepCount)

    @timing
    def _initialize(self, location):
        self._get_resp(location)
        self._get_time()
        self._get_cdac()


class csv_numpy(base):
    """Transforms CSV files to a numpy inheriting class"""

    def __init__(self, path_to_file, initialize=True):
        super().__init__()
        print(f"Initializing {self = }")
        with open(path_to_file, 'r', encoding='utf-8-sig') as f:
            self.record = np.genfromtxt(f, delimiter=',').T
        self.mode = "sweeps"
        if initialize:
            self._initialize()

    @timing
    def _get_resp(self):
        self.sweeps = self.record[1:]  # sweeps

    @timing
    def _get_time(self):
        self.time: np.ndarray = self.record[0] - self.record[0][0]  # To ensure that starts at 0.0
        self.t_delta = np.round(
                self.record[0][1] - self.record[0][0], decimals=8
                )  # time increment  # TODO use decimal as precision for the entire script
        print(f"{self.t_delta = }")

    @timing
    def _initialize(self):
        self._get_resp()
        self._get_time()


class loadRecord(base):
    """Loads files and perform basic data manipulation"""
    instance_number = 0

    def __init__(self, path_to_file, initialize=True, location=0):
        super().__init__()
        loadRecord.instance_number += 1
        self._instance_number = loadRecord.instance_number
        self._path_to_file = path_to_file
        self.direction = 0

        if path_to_file.lower().endswith(".abf"):
            print(f"Using {abf_numpy = }")
            self.data_object = abf_numpy(path_to_file, initialize, location)
        elif path_to_file.lower().endswith(".csv"):
            print(f"Using {csv_numpy = }")
            self.data_object = csv_numpy(path_to_file, initialize)
        else:
            raise NotImplementedError("Format not recognized, use .abf or .csv.")

        self.__dict__.update(self.data_object.__dict__)
        del self.data_object
        # print(f"{self.__dict__ = }")

    @timing
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        cls.instance_number += 1
        result._instance_number += 1
        return result

    @timing
    def __len__(self):
        return len(self.time)

    @timing
    def __add__(self, other):
        # try:
        if isinstance(other, loadRecord):
            self.time = np.append(self.time, other.time)
            self.resp = np.append(self.resp, other.resp)
            self.cdac = np.append(self.cdac, other.cdac)
        elif isinstance(other, (list, tuple, np.ndarray)) and len(other) == 3:
            self.time = np.append(self.time, other[0])
            self.resp = np.append(self.resp, other[1])
            self.cdac = np.append(self.cdac, other[2])
        else:
            print(f"Type not supported. Use: list, tuple or np.ndarray of length 3.")

    @timing
    def __getitem__(self, item):
        match item:
            case 0:
                return self.time
            case 1:
                return self.resp
            case 2:
                return self.cdac
            case -1:
                return self.cdac
            case -2:
                return self.resp
            case -3:
                return self.time
            case _:
                raise IndexError

    @timing
    def set_resp(self, index):
        if 0 <= index <= len(self.sweeps):
            self.resp = self.sweeps[index]
        else:
            raise IndexError(f"The index ({index}) is out of bounds, chose between 1 and {len(self.sweeps)}.")

    @timing
    def transfer(self, other):
        if isinstance(other, loadRecord):
            self.time = other.time
            self.resp = other.resp
            self.cdac = other.cdac
        elif isinstance(other, (list, tuple, np.ndarray)):
            self.time = other[0]
            self.resp = other[1]
            self.cdac = other[2]
        else:
            print(f"Type not supported. Use: list, tuple or np.ndarray of length 3.")

    @timing
    def section(self, start, end):
        # start_pos = end_pos = 0
        print(f"{self.time[0] = }  {self.time[-1] = }")
        try:
            start_pos = np.where(start == self.time)[0][0]
        except IndexError:
            closest_pos = np.argmin(np.abs(self.time - end))
            print(f"Using {len(self.time) = } {self.time[-1] = }  {closest_pos = }  {self.time[closest_pos]}")
            start_pos = closest_pos
        try:
            end_pos = np.where(end == self.time)[0][0]
        except IndexError:
            closest_pos = np.argmin(np.abs(self.time - end))
            print(f"Using {len(self.time) = } {self.time[-1] = }  {closest_pos = }  {self.time[closest_pos]}")
            end_pos = closest_pos
        print(f"{start_pos = }  {end_pos = }")
        self.resp = self.resp[start_pos:end_pos]
        self.time = self.time[start_pos:end_pos]
        self.cdac = self.cdac[start_pos:end_pos]

    @timing
    def down_sample(self, down_sample=10, with_threshold=False, accept_mask=None):
        if not with_threshold:
            print(f"{with_threshold = }")
            self.time = down_sample_function(self.time, down_sample)
            self.resp = down_sample_function(self.resp, down_sample)
            self.cdac = down_sample_function(self.cdac, down_sample)
        elif with_threshold:
            print(f"{with_threshold = }")
            self.time = down_sample_function_t(self.time, accept_mask, down_sample)
            self.resp = down_sample_function_t(self.resp, accept_mask, down_sample)
            self.cdac = down_sample_function_t(self.cdac, accept_mask, down_sample)

    @timing
    def get_stack(self, size=2):
        if size == 2:
            return np.stack((self.time, self.resp), axis=0).T
        elif size == 3:
            return np.stack((self.time, self.resp, self.cdac), axis=0).T
        return None

    @timing
    def get_info(self, from_what="file", parameter=""):
        match from_what:
            case 'file':
                return file_info(self._path_to_file, parameter)
            case 'script':
                return file_info(__file__, parameter)
            case _:
                print("No file/script was entered")
                return None

    def clean(self):
        self.time = np.array([])
        self.resp = np.array([])
        self.cdac = np.array([])


class Fourier(loadRecord):
    """Apply Fourier analysis to the recordings"""

    def __init__(self, path_to_file, initialize=True, location=0):
        super().__init__(path_to_file, initialize, location)
        self.fft_series = np.array([])
        self.fft_domain = np.array([])
        self.freq_increment = 0

    @timing
    def _get_fft_series(self, option):
        self.fft_series = fft.rfft(option)
        print(f"{self.fft_series = }")

    @timing
    def _get_fft_domain(self):
        self.fft_domain = fft.rfftfreq(self.time.shape[-1])
        self.freq_increment = self.fft_domain[1] - self.fft_domain[0]

    @timing
    def get_fft(self, component='resp'):
        match component:
            case 'resp':
                option = self.resp
            case 'time':
                option = self.time
            case 'cdac':
                option = self.cdac
            case _:
                option = None
        self._get_fft_series(option)
        self._get_fft_domain()

    @timing
    def get_ifft(self):
        self.resp = fft.irfft(self.fft_series)

    @timing
    def filter(self, freq_width, filter_array, attenuation):
        freq_width = freq_width * self.t_delta  # conversion for fft
        w_pos = vtp(freq_width / 2, self.freq_increment)
        frequencies_to_filter = np.array(filter_array) * self.t_delta  # conversion for fft
        fft_series = self.fft_series
        for freq in frequencies_to_filter:
            f_pos = vtp(freq, self.freq_increment)
            if freq == 0.0:
                fft_series[:w_pos + 1] *= attenuation
            else:
                fft_series[f_pos - w_pos: f_pos + w_pos + 1] *= attenuation
        self.fft_series = fft_series

    @timing
    def fft_plot(self, title="Theoretical FFT", max_plot_points=500000):
        print(f"{self.fft_domain = }  {(1 / self.t_delta) = }  {self.t_delta = }")
        # --- RAM FIX 1: In-Place Array Math ---
        f_r_theoretical = self.fft_domain * (1.0 / self.t_delta)
        # Calculate absolute values (creates 1 new array instead of 3)
        f_s_theoretical = np.abs(self.fft_series)
        # Apply scaling IN-PLACE. This modifies the existing array without using extra RAM.
        scale_factor = 2.0 / len(self.time)
        f_s_theoretical *= scale_factor
        # --------------------------------------
        # --- RAM FIX 2: Matplotlib Decimation ---
        # Plotting millions of points kills Matplotlib.
        # If the array is huge, we slice it to skip points just for the visual plot.
        # (The actual math data remains untouched).
        if len(f_r_theoretical) > max_plot_points:
            step = len(f_r_theoretical) // max_plot_points
            plot_x = f_r_theoretical[::step]
            plot_y = f_s_theoretical[::step]
        else:
            plot_x = f_r_theoretical
            plot_y = f_s_theoretical
        # ----------------------------------------
        plt.figure()
        plt.title(title)
        # Plot the optimized arrays
        plt.plot(plot_x, plot_y, linewidth=0.05)
        plt.axhline()
        plt.yscale('log', base=np.e)
        two_decimal_lambda_formatter = FuncFormatter(lambda x, pos: f"{x:.3f}")
        plt.gca().yaxis.set_major_formatter(two_decimal_lambda_formatter)
        plt.axhline(y=10, color='r', linestyle='dashed', label="10 [pA]")
        plt.axhline(y=0.0001, color='k', linestyle='dashed', label="0.0001 [pA]")
        plt.xlabel("Frequencies [Hz]")
        plt.ylabel("Amplitude (Log Scale)")
        plt.legend(loc="upper right")
        plt.grid(True, which="both", ls="-", lw=0.5)
        plt.show(block=False)


class Analyzer(Fourier):
    """Analyzes the recordings"""

    def __init__(self, path_to_file="", initialize=True, location=0):
        super().__init__(path_to_file, initialize, location)
        self.pul_attrs = {}  # Initialization
        self.pulses_peaks = np.array([])
        self.peak_noise = np.array([])  # Initialization
        self.std = 0.0  # Initialization
        self.derivative = np.array([])  # Initialization
        self.der_peaks = np.array([])  # Initialization
        self.o_thresh = np.array([])  # Initialization
        self.peaks = np.array([])  # Initialization
        self.zero_pass = np.array([])  # Initialization
        self.inp_res = np.array([])  # Initialization
        self.mem_cap = np.array([])  # Initialization
        self.acc_res = np.array([])  # Initialization
        self.area: dict[str, float | ndarray[Any, dtype]] = {"Area": 0.0, "Amplitude": 0.0, "rTTP": 0.0}

    @timing
    def get_smooth(self, smooth_width=0.001, sharpness=4):
        n_p = vtp(smooth_width, self.t_delta)
        self.resp = smoothing(self.resp, n_p, sharpness)

    @timing
    def get_section_area(self, baseline_start=371, baseline_end=376, response_end=441, linear_fit=False):
        """Calculates the area, peak and rTTP"""
        b_s_p = np.where(self.time == baseline_start)[0][0]
        b_e_p = np.where(self.time == baseline_end)[0][0]
        r_e_p = np.where(self.time == response_end)[0][0]
        points = vtp(0.002, self.t_delta)

        shift = np.nanmean(self.resp[b_s_p:b_e_p])

        resp = smoothing(copy.deepcopy(self.resp[b_e_p:r_e_p]) - shift, points,  2)
        r_time = copy.deepcopy(self.time[b_e_p:r_e_p])
        base = smoothing(copy.deepcopy(self.resp[b_s_p:b_e_p]) - shift, points,  2)
        b_time = copy.deepcopy(self.time[b_s_p:b_e_p])

        if linear_fit:
            r_time, resp = remove_shift(np.array([b_time, base]), np.array([r_time, resp]))

        self.area["Area"] = calculate_area(r_time, resp)

        match self.direction:
            case 1:
                self.area["Amplitude"] = np.max(resp)
                self.area["rTTP"] = r_time[np.argmax(resp)] - r_time[0]
            case -1:
                self.area["Amplitude"] = np.min(resp)
                self.area["rTTP"] = r_time[np.argmin(resp)] - r_time[0]
            case _:
                print(f"Invalid direction: {self.direction}")

        plt.figure()
        plt.axhline(0, color='g', linestyle='dashed', linewidth=1)
        plt.plot(r_time, resp)
        plt.title(f"Area={self.area["Area"]:.2f}, Amplitude={self.area["Amplitude"]:.2f}, rTTP={self.area["rTTP"]:.2f}")
        plt.show()

    @timing
    # @njit
    def find_pulses(self, direction: int = -1) -> None:
        """It gives the position of the start of the control pulses"""
        der_test_resp = differentiate(self.cdac, self.t_delta)
        der_test_resp = prev_change(der_test_resp)
        threshold = der_test_resp * 0.0 + self.direction * 9 * np.std(der_test_resp)
        o_thresh = find_over_threshold(der_test_resp, threshold, direction)
        self.pulses_peaks = find_peaks(o_thresh, der_test_resp, direction, self.t_delta, self.t_delta)
        self.pul_attrs = {evt_pos: {} for evt_pos, val in enumerate(self.pulses_peaks) if val}

    @timing
    def del_pulses(self, del_length=0.75, target_val=2000.0):
        peaks = self.pulses_peaks
        resp_copy = copy.deepcopy(self.resp)
        time_copy = copy.deepcopy(self.time)
        # cdac_copy = copy.deepcopy(self.cdac)
        del_range = vtp(del_length, self.t_delta)
        for pos, val in enumerate(peaks):
            if val:
                resp_copy[pos:pos + del_range] = target_val
        index = np.argwhere(resp_copy == target_val)
        self.time = np.delete(time_copy, index)
        self.resp = np.delete(resp_copy, index)
        # self.resp = np.delete(cdac_copy, index)

    @timing
    def get_pk_noise(self, time_frame=0.2, n_deviations=3, resp_increment=0.5, std_increment=10, sharpness=2):
        n_p = vtp(time_frame, self.t_delta)
        print(f"Delete me after {n_p=}")
        smoothed_resp = smoothing(self.resp, n_p, sharpness)
        print(f"Delete me after {smoothed_resp=}")
        # 1. Use a separate variable name for the dictionary
        resp_inc_dict = {"start": self.time[0], "end": self.time[-1], "increment": resp_increment}
        print(f"Delete me after {resp_inc_dict=}")
        std_resp = apply_by(np.std, np.array([self.time, self.resp]), resp_inc_dict, True)
        print(f"Delete me after {std_resp=}")
        # Safety Check: Did resp_increment fail?
        if std_resp.size == 0 or len(std_resp[0]) == 0:
            print("Warning: std_resp is empty. Falling back to 0 noise.")
            self.std = 0.0
            self.peak_noise = smoothed_resp
            return

        # 2. Calculate the ACTUAL time span dynamically
        start_time = std_resp[0][0]
        end_time = std_resp[0][-1]
        time_span = end_time - start_time

        # 3. The Mechanism: Intelligently scale std_increment if it's too big
        if std_increment >= time_span:
            warnings.warn(
                    f"std_increment ({std_increment}) is larger than the time span ({time_span:.1f}s). "
                    f"Adjusting dynamically..."
                    )
            # Fall back to splitting the array into 2 chunks.
            # (Ensure it's never smaller than resp_increment to prevent math errors)
            std_increment = max(time_span / 2.0, resp_increment)

        std_inc_dict = {"start": start_time, "end": end_time, "increment": std_increment}
        print(f"Delete me after: {std_inc_dict=}")

        std_min = apply_by(np.min, std_resp, std_inc_dict, True)

        # 4. The Safety Net: If apply_by STILL returns empty, don't crash.
        if std_min.size == 0 or len(std_min[0]) == 0:
            print("Warning: apply_by returned an empty array for std_min. Using global minimum instead.")
            self.std = np.nanmean(std_resp[1])  # Fallback to the mean of the whole section
            global_min = np.nanmin(std_resp[1])
            self.peak_noise = smoothed_resp + (global_min * n_deviations * self.direction)
        else:
            # Standard successful execution
            self.std = np.nanmean(std_min[1])
            self.peak_noise = smoothed_resp + np.interp(
                    self.time, std_min[0],
                    std_min[1] * n_deviations * self.direction
                    )

    # def get_pk_noise(self, time_frame=0.2, n_deviations=3, resp_increment=0.5, std_increment=10, sharpness=2):
    #     n_p = vtp(time_frame, self.t_delta)
    #     # fftconvolve is better for long arrays
    #     # smoothed_resp = fftconvolve(self.resp, conv_vector(n_p, 'g', sharpness), mode='valid')  # response
    #     smoothed_resp = smoothing(self.resp, n_p, 1, sharpness)
    #     # Calculates the standard deviation every resp_increment
    #     resp_increment = {"start": self.time[0], "end": self.time[-1], "increment": resp_increment}
    #     std_resp = apply_by(np.std, np.array([self.time, self.resp]), resp_increment, True)
    #     # Selects the minimum value every std_increment
    #     if std_increment > self.time[-1]:
    #         warnings.warn(f"{std_increment = } is bigger than the time interval {int(self.time[-1]) = }.")
    #         std_increment = int((self.time[-1])/2)
    #     std_increment = {"start": std_resp[0][0], "end": std_resp[0][-1], "increment": std_increment}
    #     print(f"Delete me after {std_increment=}")
    #     std_min = apply_by(np.min, std_resp, std_increment, True)
    #     print(f"Delete me after {std_min=}")
    #     self.std = np.nanmean(std_min[1])
    #     self.peak_noise = smoothed_resp + np.interp(
    #             self.time, std_min[0],
    #             std_min[1] * n_deviations * self.direction
    #             )

    @timing
    def get_derv(self):
        self.derivative = differentiate(self.resp, self.t_delta)

    @timing
    def get_o_thresh(self):
        self.o_thresh = find_over_threshold(self.resp, self.peak_noise, self.direction)

    @timing
    def get_peaks(self, search_width=0.01, shift_time=0.001, kernel_length=0.004):
        self.get_o_thresh()
        n_p = vtp(kernel_length, self.t_delta)
        kernel = conv_vector(n_p, 'g', 2)
        single_pulse = np.zeros(2 * n_p)
        single_pulse[n_p] = 1
        min_val = np.round(np.max(fftconvolve(single_pulse, kernel, mode='same')), decimals=2)
        smoothed_thre = smoothing(np.where(self.o_thresh, 1, 0), n_p, 2)
        smoothed_thre_bool = smoothed_thre > min_val
        smoothed_thre_norm = np.where(smoothed_thre_bool, 1, 0)

        # plt.figure()
        # plt.plot(self.time, self.resp)
        # plt.plot(self.time, self.o_thresh * 10)
        # plt.plot(self.time, smoothed_thre * 10)
        # plt.plot(self.time, smoothed_thre_norm * 10, "k")
        # plt.title("Testing over-threshold, delete me after")
        # plt.show(block=False)

        self.peaks = find_peaks(smoothed_thre_norm, self.resp, self.direction, search_width, self.t_delta)
        # Vectorized artifact removal
        # Only proceed if there are artifacts to filter against
        if np.any(self.pulses_peaks):
            shift = vtp(shift_time, self.t_delta)
            # 1. Identify where artifacts are (boolean array)
            has_artifact = (self.pulses_peaks != 0).astype(int)
            # 2. Create a "smearing" kernel
            # The slice [i-shift : i+shift] has a width of 2*shift
            # We use 'mode=same' so the window is centered on the artifact
            kernel_size = 2 * shift
            if kernel_size < 1: kernel_size = 1
            kernel = np.ones(kernel_size, dtype=int)
            # 3. Convolve to create a "danger zone" mask
            # This returns True wherever an artifact is within the shift distance
            artifact_mask = np.convolve(has_artifact, kernel, mode='same') > 0
            # 4. Zero out peaks that fall inside the danger zone
            self.peaks[artifact_mask] = 0.0

    @timing
    def get_z_pass(self, zero_pass_frame=0.005, delete_peaks=True):
        spaces = vtp(zero_pass_frame, self.t_delta)
        zero_pass = np.zeros(len(self.peaks))
        max_length = len(self.peaks)
        tmp_peaks = np.copy(self.peaks)
        front = np.array([])
        back = np.array([])
        for evt_pos, value in enumerate(tmp_peaks):
            if value:
                if spaces <= evt_pos <= max_length - spaces:
                    front = np.copy(self.resp[evt_pos:evt_pos + spaces])
                    back = np.copy(flipud(self.resp[evt_pos - spaces:evt_pos + 1]))
                elif 0 >= evt_pos - spaces:
                    front = np.copy(self.resp[evt_pos:evt_pos + spaces])
                    back = np.copy(flipud(self.resp[:evt_pos + 1]))
                elif max_length <= evt_pos + spaces:
                    front = np.copy(self.resp[evt_pos:])
                    back = np.copy(flipud(self.resp[evt_pos - spaces:evt_pos + 1]))
                else:
                    warnings.warn("An unexpected error occurred in 'get_z_pass'.")

                match self.direction:
                    case -1:
                        if np.max(front) < 0:
                            front -= np.max(front)
                        if np.max(back) < 0:
                            back -= np.max(back)
                    case 1:
                        if np.min(front) > 0:
                            front -= np.min(front)
                        if np.min(back) > 0:
                            back -= np.min(back)
                    case _:
                        print("Wrong direction at get_z_pass")
                aft_zero = crossing_point(front)
                bef_zero = crossing_point(back)
                if None in (aft_zero, bef_zero):
                    if delete_peaks:
                        self.peaks[evt_pos] = 0
                else:
                    zero_pass[evt_pos - bef_zero] = -1
                    zero_pass[evt_pos + aft_zero] = 1

        self.zero_pass = zero_pass

    @timing
    def get_rescap(
            self, beg_rc=0.005, end_rc=0.1, beg_ir=0.2, end_ir=0.35
            ):
        exp_fit_local = exp_fit
        lin_fit_local = lin_fit
        np_average = np.average
        np_abs = np.abs
        beg_cap = vtp(beg_rc, self.t_delta)  # To avoid Rs artifact
        end_cap = vtp(end_rc, self.t_delta)
        beg_res = vtp(beg_ir, self.t_delta)  # To avoid passive responses
        end_res = vtp(end_ir, self.t_delta)
        current_pulse = None
        current_ires = np.array([])
        for pos, value in enumerate(self.pulses_peaks):
            if value:
                # Storing the time of the peak
                self.pul_attrs[pos]["t_o_p"] = self.time[pos]
                # Current pulse intensity
                pulse_slice = slice(pos + beg_cap, pos + end_cap)
                pulse_base_slice = slice(pos - 2 * beg_cap, pos - beg_cap)
                if current_pulse is None:
                    current_pulse = np_average(self.cdac[pulse_slice]) - np_average(self.cdac[pulse_base_slice])
                # Membrane capacitance calculations and fitting
                rc_slice = slice(pos + beg_cap, pos + end_cap)
                time_rc = self.time[rc_slice] - self.time[pos]
                voltage_rc = self.resp[rc_slice] - np_average(self.resp[pulse_base_slice])
                # Fits data to an exponential decay. I(t) = i0 + pk0 * exp(-t / t0), returns i0, pk0, t0"""
                fit_i0, fit_pk0, fit_t0, pearson_r = exp_fit_local(voltage_rc, time_rc, -1 * self.direction)
                # Membrane resistance (input resistance) calculations and fitting
                ir_slice = slice(pos + beg_res, pos + end_res)
                if len(current_ires) == 0:
                    current_ires = self.cdac[ir_slice]
                voltage_ires = self.resp[ir_slice]
                ires, intercept, r_value, p_value, std_err, i_stderr = lin_fit_local(voltage_ires, current_ires)
                # Storing
                self.pul_attrs[pos]["input_res"] = ires * 10 ** 3  # In mega Ohms
                self.pul_attrs[pos]["mem_cap"] = np_abs((10 ** 3) * fit_t0 * current_pulse / fit_pk0)  # In pico Farads
                self.pul_attrs[pos]["mem_tau"] = fit_t0  # In seconds

        self.inp_res = np.array([self.get_pulse_arr("t_o_p"), self.get_pulse_arr("input_res")]).T
        self.mem_cap = np.array([self.get_pulse_arr("t_o_p"), self.get_pulse_arr("mem_cap")]).T

    @timing
    def get_rira(self, beg_ar=0.02, end_ar=0.005, beg_ir=0.25, end_ir=0.750):
        lin_fit_local = lin_fit
        np_average = np.average
        np_min = np.min
        np_abs = np.abs
        bas_acc = vtp(beg_ar - 0.001, self.t_delta)
        beg_acc = vtp(beg_ar, self.t_delta)
        end_acc = vtp(end_ar, self.t_delta)
        beg_res = vtp(beg_ir, self.t_delta)
        end_res = vtp(end_ir, self.t_delta)
        voltage_pulse = None
        voltage_ires = np.array([])
        for pos, value in enumerate(self.pulses_peaks):
            if value:
                # Storing the time of the peak
                self.pul_attrs[pos]["t_o_p"] = self.time[pos]
                # Access resistance calculations
                pulse_slice = slice(pos + end_acc, pos + beg_acc)
                pulse_base_slice = slice(pos - beg_acc, pos - end_acc)
                if voltage_pulse is None:
                    voltage_pulse = np_average(self.cdac[pulse_slice]) - np_average(self.cdac[pulse_base_slice])
                current_acc = self.resp[pos - beg_acc:pos + end_acc]
                base_current = np_average(current_acc[:bas_acc])
                c_diff = base_current - np_min(current_acc)
                # Input resistance calculations and fitting
                ir_slice = slice(pos + beg_res, pos + end_res)
                if len(voltage_ires) == 0:
                    voltage_ires = self.cdac[ir_slice]
                current_ires = self.resp[ir_slice]
                ires, intercept, r_value, p_value, std_err, i_stderr = lin_fit_local(voltage_ires, current_ires)
                # Storing
                self.pul_attrs[pos]["acc_res"] = np_abs(voltage_pulse / c_diff) * 1000  # In Mega Ohms
                self.pul_attrs[pos]["input_res"] = ires * 1000  # In Mega Ohms
                self.pul_attrs[pos]["r_value"] = r_value

        self.inp_res = np.array([self.get_pulse_arr("t_o_p"), self.get_pulse_arr("input_res")]).T
        self.acc_res = np.array([self.get_pulse_arr("t_o_p"), self.get_pulse_arr("acc_res")]).T

    @timing
    def get_iv(self, beg_ar=0.02, end_ar=0.005, beg_ir=0.25, end_ir=0.750, holding=-60):
        # lin_fit_local = lin_fit
        np_average = np.average
        np_min = np.min
        np_abs = np.abs
        bas_acc = vtp(beg_ar - 0.001, self.t_delta)
        beg_acc = vtp(beg_ar, self.t_delta)
        end_acc = vtp(end_ar, self.t_delta)
        beg_res = vtp(beg_ir, self.t_delta)
        end_res = vtp(end_ir, self.t_delta)
        voltage_pulse = None
        voltage_ires = np.array([])
        for pos, value in enumerate(self.pulses_peaks):
            if value:
                # Storing the time of the peak
                self.pul_attrs[pos]["t_o_p"] = self.time[pos]
                # Access resistance calculations
                pulse_slice = slice(pos + end_acc, pos + beg_acc)
                pulse_base_slice = slice(pos - beg_acc, pos - end_acc)
                if voltage_pulse is None:
                    voltage_pulse = np_average(self.cdac[pulse_slice]) - np_average(self.cdac[pulse_base_slice])
                current_acc = self.resp[pos - beg_acc:pos + end_acc]
                base_current = np_average(current_acc[:bas_acc])
                c_diff = base_current - np_min(current_acc)
                # Input resistance calculations and fitting
                ir_slice = slice(pos + beg_res, pos + end_res)
                if len(voltage_ires) == 0:
                    voltage_ires = self.cdac[ir_slice]
                current_ires = self.resp[ir_slice]
                current_time = self.time[ir_slice]
                # Storing
                self.pul_attrs[pos]["acc_res"] = np_abs(voltage_pulse / c_diff) * 1000  # In Mega Ohms
                self.pul_attrs[pos]["iv_res"] = current_ires
                # self.pul_attrs[pos]["iv_res_smooth"] = smoothing(current_ires, n_p, 1, 4)
                self.pul_attrs[pos]["iv_time"] = current_time

        iv_iter = tuple(
                zip(
                        self.get_pulse_arr("iv_time"),
                        self.get_pulse_arr("iv_res"),
                        self.get_pulse_arr("acc_res"),
                        )
                )
        base_resp = iv_iter[0]

        plt.figure()
        plt.axhline(0.0, color="k", linestyle='--')
        plt.plot(self.time, self.resp, "r")
        for curr in iv_iter:
            plt.plot(curr[0], base_resp[1], "g")
            plt.plot(curr[0], curr[1] - base_resp[1], "b")
            plt.plot(curr[0], np.zeros_like(voltage_ires) + curr[2], "k")
            plt.plot(curr[0], np.zeros_like(voltage_ires) + base_resp[2], "g")
        plt.title(f"Pulse difference.")
        plt.show(block=False)

        fig, ax = plt.subplots()
        n_p = vtp(0.002, self.t_delta)
        i = (len(iv_iter) - 1) * 10
        for curr in reversed(iv_iter):
            plt.plot(
                    voltage_ires + holding,
                    smoothing(curr[1] - base_resp[1], n_p, 4),
                    label=f"{i}[s]",
                    # alpha=0.75
                    )
            i -= 10
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('Voltage [mV]', loc='right')  # 'loc' can be 'left', 'center', or 'right'
        ax.set_ylabel('Current [pA]', loc='top', rotation=0)  # 'loc' can be 'bottom', 'center', or 'top'
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.legend()
        plt.title(f"I-V Graph.")
        plt.show(block=False)

    @timing
    def get_pulse_arr(self, name):
        return np.array([values[name] for values in self.pul_attrs.values()])


class EvtPro(Analyzer):
    """Event detection class"""

    def __init__(self, path_to_file="", initialize=True, location=0):
        super().__init__(path_to_file, initialize, location)
        self.events_attrs = {}
        self.default_event = {
                "slope"           : None,  # Rise-slope value
                "slope_peak_delta": None,  # Time difference between the rise-slope and the peak
                "t_o_p"           : None,  # Time of the alignment-peak
                "t_o_s"           : None,  # time of the max slope
                "t_o_zs"          : None,  # time of zero pass start
                "t_o_zp"          : None,  # time of zero pass peak
                "t_segm"          : None,  # time segment
                "r_segm"          : None,  # response segment
                "p_segm"          : None,  # peak segment, where the peak is located
                "d_segm"          : None,  # derivative segment
                "s_segm"          : None,  # max slope location
                "z_segm"          : None,  # zero pass locations, critical points
                "b_amp"           : None,  # Baseline amplitude
                "amplitude"       : None,  # Absolute amplitude of the response
                "r_ifreq"         : None,  # Instantaneous frequency of the response
                "r_inter"         : None,  # Interval between the response and the previous response
                # I(t) = i0 + pk0 * exp(-t / t0)
                "fit_min"         : None,  # Fit of i0
                "fit_peak"        : None,  # Fit of pk0
                "tau"             : None,  # Exp. decay fit constant, t0
                "r_decay"         : None,  # Pearson's R of the fit
                "mse_fit"         : None,  # Minimal standard error of the fit
                "r_auc"           : None,  # Area under the curve of the response
                "threshold_segm"  : None,  # threshold segment
                "ap_threshold"    : None,  # threshold value
                }
        self.ps_nsfa_values = {}
        self.burst_attrs = {}
        self.freq_blocks = np.array([])
        self._events_positions = np.array([])  # Initialization
        self.common_time = np.array([])  # Initialization

    @timing
    def _select_events(self, slope_peak_time, max_slope):
        initial_msp_p: int = vtp(slope_peak_time, self.t_delta)
        self.events_attrs = {}
        # FIX: Get indices of peaks once. This avoids looping through every sample.
        peak_indices = np.where(self.peaks)[0]
        num_peaks = len(peak_indices)
        if num_peaks == 0:
            print("No peaks detected.")
            return
        # We iterate through the peaks directly.
        # To analyze peak 'i', we look at 'i-1' for the start and 'i+1' for the end.
        for i in range(num_peaks):
            evt_pos = peak_indices[i]
            # Boundary logic:
            # prev_pos: the end of the previous peak (or start of file)
            # next_pos: the start of the next peak (or end of file)
            prev_pos = peak_indices[i - 1] if i > 0 else 0
            next_pos = peak_indices[i + 1] if i < num_peaks - 1 else len(self.time) - 1
            # 1. Determine msp_p (Peak-slope distance)
            # if evt_pos - prev_pos > initial_msp_p:
            #     msp_p = evt_pos - prev_pos
            # else:
            #     msp_p = initial_msp_p
            if prev_pos:
                msp_p = evt_pos - prev_pos
            else:
                msp_p = evt_pos
            # cond_slope_range = evt_pos - msp_p > 0.0
            max_slope_region = slice(evt_pos - msp_p, evt_pos + 1)
            # Check if the region has derivative data
            # if cond_slope_range and len(self.der_peaks[max_slope_region]) > 0:
            if len(self.der_peaks[max_slope_region]):
                cond_max_slope = np.max(self.der_peaks[max_slope_region])
                if cond_max_slope:
                    self.events_attrs[evt_pos] = self.default_event.copy()
                    # Slope calculations
                    rel_slope_pos = crossing_point(flipud(self.der_peaks[max_slope_region]))
                    slope_value = flipud(self.derivative[max_slope_region])[rel_slope_pos]
                    slope_time = flipud(self.time[max_slope_region])[rel_slope_pos]
                    # Directional check
                    is_rejected = False
                    match self.direction:
                        case -1:
                            if not max_slope < slope_value: is_rejected = True
                        case 1:
                            if not max_slope > slope_value: is_rejected = True
                    if is_rejected:
                        print(
                                f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected (slope threshold)"
                                f" {slope_value=} {max_slope=}"
                                )
                        self.events_attrs.pop(evt_pos, None)
                        continue
                    # Rise-start and Peak location (Zero-pass)
                    abs_slope_pos = vtp(slope_time - self.time[0], self.t_delta)
                    starting_region = slice(max(0, prev_pos - 1), abs_slope_pos + 1)
                    peak_region = slice(abs_slope_pos, next_pos + 1)
                    zs_p = crossing_point(flipud(self.zero_pass[starting_region]))
                    zp_p = crossing_point(self.zero_pass[peak_region])
                    if zs_p is None or zp_p is None:
                        print(
                                f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected (crossings not found) {zs_p = } {zp_p = }"
                                )
                        self.events_attrs.pop(evt_pos, None)
                        continue
                    # Data Assignment
                    t_o_zs = flipud(self.time[starting_region])[zs_p]
                    t_o_zp = self.time[peak_region][zp_p]
                    # Update attributes
                    self.events_attrs[evt_pos].update(
                            {
                                    "t_o_s"           : slope_time,
                                    "slope_peak_delta": self.time[evt_pos] - slope_time,
                                    "rise_slope_val"  : slope_value,
                                    "t_o_zs"          : t_o_zs,
                                    "t_o_zp"          : t_o_zp,
                                    "peak_error"      : self.time[evt_pos] - t_o_zp,
                                    "rise_time_peak"  : t_o_zp - t_o_zs,
                                    "rise_time_der"   : self.time[evt_pos] - t_o_zs,
                                    "slope_pos_delta" : evt_pos - abs_slope_pos,
                                    "start_pos_delta" : evt_pos - vtp(t_o_zs - self.time[0], self.t_delta),
                                    "peak_pos_delta"  : evt_pos - vtp(t_o_zp - self.time[0], self.t_delta)
                                    }
                            )
                else:
                    # FIX: Added this else block to catch the silent rejection when max slope is 0
                    print(f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected ({cond_max_slope=}) {evt_pos - msp_p=}")
            else:
                print(f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected (No slope region)")
        print(f" Accepted events: {len(self.events_attrs)}, Rejected: {num_peaks - len(self.events_attrs)}")

    @timing
    def _event_sections(self, t_aft, baseline_time, zp_to_pp, peak_to_peak=0.001, max_rise_time=0.00263):
        a_p: int = vtp(t_aft, self.t_delta)
        baseline_points = vtp(baseline_time, self.t_delta)
        evts_order = list(enumerate(self.events_attrs.keys()))
        for order, evt_pos in evts_order:
            # Assessment of the start of the next event. If evt_pos is the last, then use the end as next_pos
            if order + 1 < len(evts_order):
                next_pos = evts_order[order + 1][1]
                next_pos_start = (next_pos - self.events_attrs[next_pos]["start_pos_delta"])
            else:
                next_pos_start = len(self.peaks) - 1
            # Slice from evt_pos until next event start
            decay_region = slice(evt_pos, next_pos_start)
            # To determine if there are intermediate peaks
            decay_peaks = self.peaks[decay_region]
            decay_peaks_pos = np.where(decay_peaks == 1.0)[0][1:]
            decay_peaks_condition = (self.time[decay_peaks_pos + evt_pos] - self.time[
                evt_pos]) > peak_to_peak
            if decay_peaks_pos[decay_peaks_condition].size:
                inter_pos = evt_pos + decay_peaks_pos[decay_peaks_condition][0]
            else:
                inter_pos = next_pos_start
            # To determine if a pulse is in range
            if self.cdac.size and np.sum(self.pulses_peaks[decay_region]):
                pulse_pos = np.argmax(self.pulses_peaks[decay_region]) + evt_pos
            else:
                pulse_pos = next_pos_start
            # To assess the closest interference position
            if evt_pos < np.min([inter_pos, pulse_pos]):
                interference_pos = np.min([inter_pos, pulse_pos])
            else:
                interference_pos = next_pos_start
            # To determine if the interference is inside the region of interest
            if interference_pos < evt_pos + a_p:
                print(f"There was an interference in the event {self.time[evt_pos]:12.4f}")
                end_roi_pos = interference_pos
            else:
                end_roi_pos = evt_pos + a_p
            # Slice for the region of interest
            roi_slice = slice(
                    evt_pos - self.events_attrs[evt_pos]["start_pos_delta"] - baseline_points,
                    end_roi_pos + 1
                    )
            if end_roi_pos - (evt_pos - self.events_attrs[evt_pos]["start_pos_delta"]) < 2:
                print(
                        f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected, short response "
                        f"{end_roi_pos=} {self.events_attrs[evt_pos]["start_pos_delta"]=}"
                        )
                self.events_attrs.pop(evt_pos, f"{evt_pos=} not found")
                continue
            t_segm = self.time[roi_slice]  # time segment
            t_o_zs = self.events_attrs[evt_pos]["t_o_zs"]
            t_o_zp = self.events_attrs[evt_pos]["t_o_zp"]
            peak_error = self.events_attrs[evt_pos]["peak_error"]
            if t_o_zs not in t_segm or t_o_zp not in t_segm or abs(peak_error) > zp_to_pp:
                print(
                        f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected,"
                        f" {t_segm[0]=:5.4f} {t_o_zs=:5.4f} {t_o_zp=:5.4f} {t_segm[-1]=:5.4f}."
                        f" {abs(peak_error)=:2.4f}  {zp_to_pp=}"
                        )
                self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")
                continue

            rise_time_peak = self.events_attrs[evt_pos]["rise_time_peak"]
            rise_time_der = self.events_attrs[evt_pos]["rise_time_der"]
            # if rise_time_peak > max_rise_time or rise_time_der > max_rise_time:
            #     print(
            #             f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected,"
            #             f" {rise_time_peak=:5.4f} {rise_time_der=:5.4f} {max_rise_time=:5.4f}."
            #             )
            #     self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")
            #     continue

            z_segm = reset_array(self.zero_pass[roi_slice], np.where(t_segm == t_o_zp)[0][0])
            z_segm[np.where(t_segm == t_o_zs)[0][0]] = -1
            r_segm = self.resp[roi_slice]  # response segment
            p_segm = self.peaks[roi_slice]
            if np.sum(p_segm) > 1:
                print(f"{evt_pos} {self.time[evt_pos]:12.4f}[s] multiple peaks!! ({np.sum(p_segm)=})")
                p_segm = reset_array(self.peaks[roi_slice], np.where(t_segm == self.time[evt_pos])[0][0])
            elif np.sum(p_segm) == 0:
                print(
                        f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected, No peaks!!! "
                        f" {np.sum(p_segm)=} {t_segm[0]=:5.5f}  {t_segm[-1]=:5.5f}."
                        )
                self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")
                continue
            d_segm = self.derivative[roi_slice]  # derivative segment
            s_segm = reset_array(
                    self.der_peaks[roi_slice],
                    np.where(t_segm == self.events_attrs[evt_pos]["t_o_s"])[0][0]
                    )
            self.events_attrs[evt_pos]["end_time"] = t_segm[-1] - self.time[evt_pos]
            self.events_attrs[evt_pos]["t_segm"] = t_segm
            self.events_attrs[evt_pos]["r_segm"] = r_segm
            self.events_attrs[evt_pos]["p_segm"] = p_segm
            self.events_attrs[evt_pos]["d_segm"] = d_segm
            self.events_attrs[evt_pos]["s_segm"] = s_segm
            self.events_attrs[evt_pos]["z_segm"] = z_segm
        print(f" Accepted events: {len(self.events_attrs)}, Rejected: {len(evts_order) - len(self.events_attrs)}")

    @timing
    def get_evt(
            self, slope_peak_time: float = 0.005, max_slope: float = -15000, peak_to_peak: float = 0.01,
            t_bef=0.02, t_aft=0.04, zp_to_pp=0.002, baseline_time=0.002, max_rise_time=0.00263
            ):
        """Stores the position of the peak of an event.
        Selects events with the peak occurring after the max slope"""
        self._select_events(slope_peak_time, max_slope)
        self._event_sections(t_aft, baseline_time, zp_to_pp, peak_to_peak, max_rise_time)

    @timing
    def identify_evoked(  # TODO finish this function, incomplete
            self,
            pp1_r1=1.0, pp1_r2=1.08, pp1_artifact=0.002,
            pp2_r1=2.0, pp2_r2=2.08, pp2_artifact=0.002,
            search_resp=0.01
            ):
        delta_pos_p1r1 = vtp(pp1_r1, self.t_delta)
        delta_pos_p1r2 = vtp(pp1_r2, self.t_delta)
        delta_pos_p1ar = vtp(pp1_artifact, self.t_delta)
        delta_pos_p2r1 = vtp(pp2_r1, self.t_delta)
        delta_pos_p2r2 = vtp(pp2_r2, self.t_delta)
        # delta_pos_p2ar = vtp(pp2_artifact, self.t_delta)
        delta_search = vtp(search_resp, self.t_delta)

        mask_p1r1 = np.zeros_like(self.pulses_peaks)
        mask_p1r2 = np.zeros_like(self.pulses_peaks)
        mask_p2r1 = np.zeros_like(self.pulses_peaks)
        mask_p2r2 = np.zeros_like(self.pulses_peaks)
        for pos, value in enumerate(self.pulses_peaks):
            if value:
                slice_p1r1 = slice(pos + delta_pos_p1r1 + delta_pos_p1ar, pos + delta_pos_p1r1 + delta_search)
                mask_p1r1[slice_p1r1] = 1
                slice_p1r2 = slice(pos + delta_pos_p1r2 + delta_pos_p1ar, pos + delta_pos_p1r2 + delta_search)
                mask_p1r2[slice_p1r2] = 1
                slice_p2r1 = slice(pos + delta_pos_p2r1 + delta_pos_p1ar, pos + delta_pos_p2r1 + delta_search)
                mask_p2r1[slice_p2r1] = 1
                slice_p2r2 = slice(pos + delta_pos_p2r2 + delta_pos_p1ar, pos + delta_pos_p2r2 + delta_search)
                mask_p2r2[slice_p2r2] = 1
        evts_attrs_copy = copy.deepcopy(self.events_attrs)
        # plt.figure()
        # plt.plot(self.time, mask)
        # plt.plot(self.time, self.pulses_peaks)
        # plt.title("Testing, delete me after")
        # plt.show(block=False)
        plt.figure()
        for evt_pos in evts_attrs_copy.keys():
            # if mask_p1r1[evt_pos]:
            #     plt.plot(
            #             self.events_attrs[evt_pos]["t_segm"] + self.events_attrs[evt_pos]["t_o_p"],
            #             self.events_attrs[evt_pos]["r_segm"], "r"
            #             )
            # elif mask_p1r2[evt_pos]:
            #     plt.plot(
            #             self.events_attrs[evt_pos]["t_segm"] + self.events_attrs[evt_pos]["t_o_p"],
            #             self.events_attrs[evt_pos]["r_segm"], "b"
            #             )
            if mask_p2r1[evt_pos]:
                plt.plot(
                        self.events_attrs[evt_pos]["t_segm"] + self.events_attrs[evt_pos]["t_o_p"],
                        self.events_attrs[evt_pos]["r_segm"], "k"
                        )
            elif mask_p2r2[evt_pos]:
                plt.plot(
                        self.events_attrs[evt_pos]["t_segm"] + self.events_attrs[evt_pos]["t_o_p"],
                        self.events_attrs[evt_pos]["r_segm"], "g"
                        )
        plt.title(f"Testing, delete after, PP1R1")
        plt.show(block=False)

    @timing
    def get_alig(self, alignment='p'):
        t_o_p = 0
        # RAM FIX: Replaced copy.deepcopy with list(dict.keys()) for safe in-place iteration
        for evt_pos in list(self.events_attrs.keys()):
            match alignment:
                case 'p':
                    t_o_p = self.time[evt_pos]
                case 'z':
                    t_o_p = self.events_attrs[evt_pos]["t_o_zp"]
                case 's':
                    t_o_p = self.events_attrs[evt_pos]["t_o_s"]
            t_segm = self.events_attrs[evt_pos]["t_segm"]
            t_segm = np.round(t_segm - t_o_p, decimals=8)
            self.events_attrs[evt_pos]["t_o_p"] = t_o_p
            self.events_attrs[evt_pos]["t_segm"] = t_segm

    @timing
    def _get_adj(self, baseline_time=0.005, adjust=True):
        base_pos = vtp(baseline_time, self.t_delta)

        # RAM FIX: Replaced copy.deepcopy
        for evt_pos in list(self.events_attrs.keys()):
            z_segm = self.events_attrs[evt_pos]["z_segm"]
            r_segm = self.events_attrs[evt_pos]["r_segm"]
            if self.events_attrs[evt_pos]["b_amp"] is None:
                z_pos = np.argmin(z_segm)
                if z_pos:
                    self.events_attrs[evt_pos]["b_amp"] = np.mean(
                            r_segm[correct_bound(z_pos - base_pos):z_pos + 1]
                            )
                else:
                    self.events_attrs[evt_pos]["b_amp"] = np.mean(r_segm[:1])
            if adjust:
                self.events_attrs[evt_pos]["r_segm"] = r_segm - self.events_attrs[evt_pos]["b_amp"]

    @timing
    def get_amplitudes(
            self, min_ampl: float = -1.48, baseline_time: float = 0.005,
            peak_radius: float = 0.0, peak_type: str = 'p', adjust=True
            ) -> None:
        self._get_adj(baseline_time, adjust)
        p_r_p = vtp(peak_radius, self.t_delta)

        # RAM FIX: Replaced copy.deepcopy. list() is required here because we are popping items.
        initial_event_count = len(self.events_attrs)
        for evt_pos in list(self.events_attrs.keys()):
            r_segm = self.events_attrs[evt_pos]["r_segm"]
            match peak_type:
                case "p":
                    p_segm = self.events_attrs[evt_pos]["p_segm"]
                    p_t_p = np.argmax(p_segm)
                case "z":
                    z_segm = self.events_attrs[evt_pos]["z_segm"]
                    p_t_p = np.argmax(z_segm)
                case _:
                    raise ValueError(f"Wrong peak type ('p' or 'z')")
            if peak_radius > 0.0:
                amplitude = np.nanmean(r_segm[p_t_p - p_r_p: p_t_p + p_r_p + 1])
            else:
                amplitude = r_segm[p_t_p]
            if amplitude * self.direction > min_ampl * self.direction:
                self.events_attrs[evt_pos]["amplitude"] = amplitude
            else:
                print(f"Event at {self.time[evt_pos]:12.4f}[s] rejected {min_ampl=:6.2f} and {amplitude:6.2f}")
                self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")
        print(f" Accepted events: {len(self.events_attrs)}, Rejected: {initial_event_count - len(self.events_attrs)}")

    @timing
    def get_arr(self, name, element_type="evt"):
        match element_type:
            case "evt":
                return np.array([values[name] for values in self.events_attrs.values()])
            case "pul":
                return np.array([values[name] for values in self.pul_attrs.values()])
            case "burst":
                return np.array([values[name] for values in self.burst_attrs.values()])
        return None

    @timing
    def get_frequencies(self):
        t_o_p = self.get_arr("t_o_p")
        # Safety check: You need at least 2 peaks to calculate a frequency.
        # np.diff will fail or np.min will throw an error if the array is too small.
        if len(t_o_p) < 2:
            return
        # 1. Fast NumPy math
        freq_array = 1.0 / np.diff(t_o_p)
        # 2. Fast NumPy concatenation (completely avoids Python lists)
        inst_freqs = np.concatenate(([np.min(freq_array)], freq_array))
        # 3. Zip and update (this is as fast as a dict update can get)
        for evt_pos, r_ifreq in zip(self.events_attrs, inst_freqs):
            self.events_attrs[evt_pos]["r_ifreq"] = r_ifreq

    # def get_frequencies(self):
    #     freq_array = 1.0 / np.diff(self.get_arr("t_o_p"))
    #     min_freq_array = np.min(freq_array)
    #     inst_freqs = [min_freq_array] + list(freq_array)
    #     for evt_pos, r_ifreq in zip(self.events_attrs, inst_freqs):
    #         self.events_attrs[evt_pos]["r_ifreq"] = r_ifreq

    @timing
    def get_intervals(self):
        for evt_pos, r_inter in zip(self.events_attrs, np.append([0.0], np.diff(self.get_arr("t_o_p")))):
            self.events_attrs[evt_pos]["r_inter"] = r_inter

    @timing
    def get_replaced(self, substitute: 'EvtPro', baseline_time: float = 0.005) -> None:
        self.transfer(substitute)
        for evt_pos in self.events_attrs:
            pre = int(np.where(self.events_attrs[evt_pos]["t_segm"] == 0.0)[0])
            post = len(self.events_attrs[evt_pos]["t_segm"]) - pre
            # To preserve alignment
            p_o_p = vtp(self.events_attrs[evt_pos]["t_o_p"], self.t_delta) - vtp(self.time[0], self.t_delta)
            # To preserve alignment
            self.events_attrs[evt_pos]["r_segm"] = self.resp[p_o_p - pre: p_o_p + post]
        self._get_adj(baseline_time)

    @timing
    def fit_events(
            self, gaussian_window=0.002, fit_beg=0.00, fit_end=0.015, pearson_r_min=0.8, sharpness=2,
            tau_min=0.001, tau_max=0.01, normal_mse_fit_max=3.0, n_limit=100, std=1.0, min_length=0.003
            ):
        """Select the events based in the fitting to exponential (decay).
        The events must be aligned previously to this analysis."""
        evts_attrs_copy = copy.deepcopy(self.events_attrs)
        n_p = vtp(gaussian_window, self.t_delta)
        local_exp_fit = exp_fit  # Local assignment for fast lookup on local scope
        local_mse = mse  # Local assignment for fast lookup on local scope
        min_len_pts = vtp(min_length, self.t_delta)
        pos_fit_start = vtp(fit_beg, self.t_delta)
        pos_fit_end = vtp(fit_end, self.t_delta)
        plt.figure()  # delete me
        for evt_pos in evts_attrs_copy.keys():
            r_segm = self.events_attrs[evt_pos]["r_segm"]
            t_segm = self.events_attrs[evt_pos]["t_segm"]
            p_segm = self.events_attrs[evt_pos]["p_segm"]
            ds_end = 0
            peak_pos = np.argmax(p_segm)
            smoothed_resp = smoothing(r_segm, n_p, sharpness)
            fit_region = slice(peak_pos + pos_fit_start, peak_pos + pos_fit_end)
            s_resp_s = smoothed_resp[fit_region]
            if s_resp_s.size == 0:
                print(f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected {s_resp_s.size=}")
                self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")
                continue
            match self.direction:
                case 1:
                    ds_end = np.argmin(s_resp_s)
                case -1:
                    ds_end = np.argmax(s_resp_s)
                case _:
                    print(f"Wrong direction")
            # Fitting assessment
            fit_region_restricted = slice(peak_pos + pos_fit_start, peak_pos + ds_end)
            r_s_short = smoothed_resp[fit_region_restricted]
            t_short = t_segm[fit_region_restricted]
            if ds_end == 0 or ds_end <= pos_fit_start or r_s_short.size < min_len_pts or r_s_short.size != t_short.size:
                print(f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected {r_s_short.size=} {t_short.size=}")
                self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")
                continue
            # fit_i_0, fit_pk0, fit_t0, pearson_r = local_exp_fit(
            #         r_s_short,
            #         t_short - t_segm[peak_pos + pos_fit_start],
            #         self.direction
            #         )
            fit_i_0, fit_pk0, fit_t0, pearson_r = local_exp_fit(
                    r_s_short,
                    t_short,
                    self.direction
                    )
            # mse calculation
            r_short = r_segm[fit_region_restricted]
            # mse_fit = local_mse(
            #         r_short,
            #         exp_decay((t_short - t_segm[peak_pos + pos_fit_start]), fit_i_0, fit_pk0, fit_t0)
            #         )
            mse_fit = local_mse(
                    r_short,
                    exp_decay(t_short, fit_i_0, fit_pk0, fit_t0)
                    )
            normal_mse_fit = mse_fit / self.events_attrs[evt_pos]["amplitude"]
            # Conditions for acceptance
            condition_pearson = abs(pearson_r) >= pearson_r_min
            condition_mse = abs(normal_mse_fit) <= normal_mse_fit_max
            condition_tau = tau_max >= abs(fit_t0) >= tau_min
            condition_fit_i_0 = abs(fit_i_0) <= abs(n_limit * std)
            condition_direction = fit_pk0 * self.direction - fit_i_0 * self.direction > std
            if condition_pearson and condition_mse and condition_tau and condition_fit_i_0 and condition_direction:
                self.events_attrs[evt_pos]["fit_min"] = fit_i_0
                self.events_attrs[evt_pos]["fit_peak"] = fit_pk0
                self.events_attrs[evt_pos]["tau"] = fit_t0
                self.events_attrs[evt_pos]["r_decay"] = pearson_r
                self.events_attrs[evt_pos]["mse_fit"] = mse_fit

                # Testing, delete me after
                if r_s_short.size < 50:
                    # plt.figure()
                    plt.plot(
                            t_segm - t_segm[peak_pos],
                            r_segm,
                            alpha=0.1,
                            )
                    # plt.plot(
                    #         t_short - t_segm[peak_pos - pos_fit_start],
                    #         exp_decay((t_short - t_segm[peak_pos - pos_fit_start]), fit_i_0, fit_pk0, fit_t0),
                    #         "b:"
                    #         )
                    plt.plot(
                            t_short - t_segm[peak_pos],
                            exp_decay(t_short, fit_i_0, fit_pk0, fit_t0),
                            "b:"
                            )
                    # plt.title(f"Testing fitting, delete me after")
                    # plt.show()
            else:
                print(
                        f"{evt_pos} {self.time[evt_pos]:12.3f}[s] rejected "
                        f" ({condition_pearson} {pearson_r=:2.3f} {condition_mse} {normal_mse_fit=:3.3f} "
                        f"{condition_tau} {fit_t0=:3.5f} {condition_fit_i_0} {fit_i_0=:2.5f} "
                        f"{condition_direction} {fit_pk0=:3.5f} {n_limit=:3.5f}  {std=:3.5f}  {n_limit*std=:3.5f}) "
                        )
                self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")

        print(f" Accepted events: {len(self.events_attrs)}, Rejected: {len(evts_attrs_copy) - len(self.events_attrs)}")
        plt.title(f"Testing fitting, delete me after")
        plt.show(block=False)

    @timing
    def _get_comm_intrv(self):
        """Determination of the common relative time interval for all events."""
        # RAM FIX: Using a direct list comprehension to concatenation skips
        # the creation of an intermediate generic Python object array
        events_time_list = [evt_values["t_segm"] for evt_values in self.events_attrs.values()]
        if events_time_list:
            events_time = np.concatenate(events_time_list)
            self.common_time = np.unique(events_time)
        else:
            self.common_time = np.array([])

    @timing
    def get_extended(self):
        if self.events_attrs:  # More Pythonic/faster than checking len()
            self._get_comm_intrv()

            # RAM/SPEED FIX: Bind to a local variable to prevent repeated class-level lookups
            common = self.common_time

            # RAM/SPEED FIX: Use .items() to get a direct reference to the inner dictionary.
            # This avoids 12 separate dictionary lookups per loop.
            for evt_pos, evt in self.events_attrs.items():
                t_segm = evt["t_segm"]

                # Reassigning directly overwrites the old array,
                # allowing the garbage collector to immediately free the old RAM
                evt["r_segm"] = extender(evt["r_segm"], t_segm, common)
                evt["p_segm"] = extender(evt["p_segm"], t_segm, common)
                evt["s_segm"] = extender(evt["s_segm"], t_segm, common)
                evt["z_segm"] = extender(evt["z_segm"], t_segm, common)
                evt["d_segm"] = extender(evt["d_segm"], t_segm, common)

                evt["t_segm"] = common
        else:
            print("No events detected!!")

    @timing
    def ps_nsfa(
            self, fit_start=0.001, fit_end=0.01, n_limit=1.0, peak_radius=0.0
            ):
        end_pos = vtp(fit_end, self.t_delta)
        local_exp_fit = exp_fit  # Local assignment for fast lookup on local scope
        avg_resp = np.nanmean(self.get_arr("r_segm"), axis=0)
        avg_time = np.nanmean(self.get_arr("t_segm"), axis=0)  # to use common_time use get_extended first

        p_r_p = vtp(peak_radius, self.t_delta)
        p_t_p = np.argmin(avg_resp)
        if peak_radius > 0.0:
            mean_peak = np.nanmean(avg_resp[p_t_p - p_r_p: p_t_p + p_r_p + 1])
        else:
            mean_peak = avg_resp[p_t_p]

        mean_resp_diff_pow2 = []
        plt.figure(figsize=(3, 2.5))

        for evt_pos in self.events_attrs:
            r_segm = self.events_attrs[evt_pos]["r_segm"]
            t_segm = self.events_attrs[evt_pos]["t_segm"]
            amplitude = self.events_attrs[evt_pos]["amplitude"]
            end_time = self.events_attrs[evt_pos]["end_time"]
            if avg_time[p_t_p + end_pos] <= end_time:
                plt.plot(t_segm, r_segm, "r", alpha=0.3)
                plt.plot(t_segm, avg_resp * (amplitude / mean_peak), "b", alpha=0.3)
                plt.plot(t_segm, r_segm - avg_resp * (amplitude / mean_peak), "b", alpha=0.1)
                mean_resp_diff_pow2.append(np.power(r_segm - avg_resp * (amplitude / mean_peak), 2))

        mean_resp_diff_pow2 = np.array(mean_resp_diff_pow2)
        n_e = len(mean_resp_diff_pow2)
        var_resp = np.sum(mean_resp_diff_pow2, axis=0) / n_e  # Variance around the scaled mean
        plt.plot(avg_time, var_resp, "k")
        plt.plot(avg_time, avg_resp, "k")

        start_pos = vtp(fit_start, self.t_delta)
        slice_section = slice(p_t_p + start_pos, p_t_p + end_pos)
        time_section = avg_time[slice_section]
        variance_section = var_resp[slice_section]
        response_section = avg_resp[slice_section]
        fit_i_0, fit_pk0, fit_t0, pearson_r = local_exp_fit(
                response_section,
                time_section - avg_time[p_t_p + start_pos],
                self.direction
                )
        extra_resp_made = np.linspace(
                fit_i_0 - 0.1,
                np.round(
                        exp_decay((time_section[0] - avg_time[p_t_p + start_pos]), fit_i_0, fit_pk0, fit_t0)
                        ).astype(int),
                np.round(np.abs(response_section[0])).astype(int)
                )
        extra_time_made = fit_t0 * np.log((extra_resp_made - fit_i_0) / fit_pk0) + avg_time[p_t_p + start_pos]
        extra_resp = exp_decay((extra_time_made - avg_time[p_t_p + start_pos]), fit_i_0, fit_pk0, fit_t0)
        plt.plot(time_section, response_section, "bo")
        plt.plot(extra_time_made, extra_resp, "go", markersize=12)
        plt.plot(
                time_section,
                exp_decay((time_section - avg_time[p_t_p + start_pos]), fit_i_0, fit_pk0, fit_t0),
                "g"
                )

        bins = [(start <= time_section) & (time_section < end) for start, end in pairwise(np.flip(extra_time_made))]
        binned_resp = remove_nan(np.array([np.nanmean(response_section[section]) for section in bins]))
        binned_var = remove_nan(np.array([np.nanmean(variance_section[section]) for section in bins]))
        binned_resp_clean = binned_resp[binned_resp < self.std * self.direction * n_limit]  # removing background noise
        binned_var_clean = binned_var[binned_resp < self.std * self.direction * n_limit]  # removing background noise
        coefficients = parabolic_fit(
                binned_resp_clean,
                binned_var_clean
                )
        intercept = coefficients[0]
        unitary_current = coefficients[1]
        channel_count = -1 / coefficients[2]
        p_0 = np.min(binned_resp_clean) / (unitary_current * channel_count)
        self.ps_nsfa_values = {
                "intercept"      : intercept,
                "i"              : unitary_current,
                "N"              : channel_count,
                "p_0"            : p_0,
                "binned_current" : binned_resp_clean,
                "binned_variance": binned_var_clean,
                "#events"        : n_e
                }

    def plot_ps_nsfa(self, axis_lim=((-10, 1), (1, 10))):
        # Implement the plotting of current vs variance for psNSFA
        current = self.ps_nsfa_values["binned_current"]
        variance = self.ps_nsfa_values["binned_variance"]
        intercept = self.ps_nsfa_values["intercept"]
        unitary_current = self.ps_nsfa_values["i"]
        channel_count = self.ps_nsfa_values["N"]
        p_0 = self.ps_nsfa_values["p_0"]
        n_e = self.ps_nsfa_values["#events"]
        # Plotting psNSFA
        plt.figure(figsize=(3, 2.5))
        plt.axhline(0.0, color="k", linestyle='--')
        plt.axvline(0.0, color="k", linestyle='--')
        plt.axvline(self.std * 3.0 * self.direction, color="r", linestyle='--')
        plt.axvline(self.std * 2.0 * self.direction, color="r", linestyle='--')
        plt.axvline(self.std * self.direction, color="r", linestyle='--')
        plt.plot(current, variance, "ko")
        artificial_current = np.linspace(
                0.0,
                np.min(current),
                np.round(np.abs(np.min(current))).astype(int)
                )
        label = f"0:{intercept:2.1f},i:{unitary_current:2.1f},N:{channel_count:2.1f},P0:{p_0:1.2f} {n_e}"
        plt.plot(
                artificial_current,
                unitary_current * artificial_current - np.power(artificial_current, 2) / channel_count,
                "r:",
                label=label,
                )
        (x0, x1), (y0, y1) = axis_lim
        plt.axhline(intercept, color="r", linestyle='--')
        plt.xlim(x0, x1)  # Set x-axis limits from 0 to 6
        plt.ylim(y0, y1)  # Set y-axis limits from 5 to 35
        plt.title(f"Average and Var around the mean {self.time[0]:4.2f} {self.time[-1]:4.2f}")
        plt.legend(loc='upper left')
        plt.show(block=False)

    @timing
    def get_auc(self, already_adjusted=True, min_auc=0.02):
        b_amp = 0

        # RAM FIX: Iterate over a list of keys instead of deepcopying the whole dictionary
        initial_event_count = len(self.events_attrs)
        for evt_pos in list(self.events_attrs.keys()):
            evt = self.events_attrs[evt_pos]  # SPEED FIX: Bind locally to avoid repeated lookups

            if not already_adjusted:
                b_amp = evt["b_amp"]
            if evt["ap_threshold"] is not None:
                start_pos = np.argmax(evt["threshold_segm"])
            else:
                start_pos = np.argmin(evt["z_segm"])
            peak_pos = np.argmax(evt["p_segm"])
            from_peak_segm = evt["r_segm"][peak_pos:] - b_amp
            crossing_pos = crossing_point(from_peak_segm)

            if crossing_pos is None:
                match self.direction:
                    case -1:
                        crossing_pos = np.argmax(from_peak_segm)
                    case 1:
                        crossing_pos = np.argmin(from_peak_segm)
                    case _:
                        print("Wrong direction in AUC")
            end_pos = peak_pos + crossing_pos
            area = calculate_area(
                    evt["t_segm"][start_pos:end_pos],
                    evt["r_segm"][start_pos:end_pos] - b_amp
                    )
            if area * self.direction > min_auc * self.direction:
                evt["r_auc"] = area
            else:
                print(
                        f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected, low AUC"
                        f" {area=:3.3f}  {min_auc=}"
                        )
                self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")
                continue
        print(f" Accepted events: {len(self.events_attrs)}, Rejected: {initial_event_count - len(self.events_attrs)}")

    @timing
    def get_threshold(self, derivative_order: int = 3):
        # SPEED FIX: Use .items() to instantly grab the inner dictionary
        for evt_pos, evt in self.events_attrs.items():

            start_pos = np.argmin(evt["z_segm"])
            peak_pos = np.argmax(evt["p_segm"])
            end_pos = peak_pos

            r_section, d1_section = sort_vectors_by_first(
                    evt["r_segm"][start_pos:end_pos],
                    evt["d_segm"][start_pos:end_pos]
                    )
            # r_section = evt["r_segm"][start_pos:end_pos]
            # d1_section = evt["d_segm"][start_pos:end_pos]
            max_slope_pos = np.argmax(d1_section)

            # --- DYNAMIC DERIVATIVE CALCULATION ---
            # Start with the 1st derivative
            current_deriv = d1_section
            # current_deriv = smoothing(current_deriv, 20, repetitions=1, sharpness=2)
            # plt.figure()
            # plt.plot(r_section, d1_section, "ko", label="1st Deriv", alpha=0.3)
            # angles = get_rolling_angles(r_section, current_deriv)
            # print(f"Delete me after {len(np.unique(angles))} {np.unique(angles)=}")
            # angles *= 100
            # plt.plot(r_section, angles, alpha=0.3)

            # Loop to calculate the 2nd, 3rd, 4th, ..., Nth derivative dynamically
            for i in range(2, derivative_order + 1):
                current_deriv = differentiate(current_deriv, 1)
                # plt.plot(r_section, current_deriv, alpha=0.3, linestyle='--')
                # plt.plot(r_section, prev_deriv - current_deriv, alpha=0.3)
                # Optional: If you want to plot EVERY intermediate step, uncomment the line below
                # plt.plot(r_section, current_deriv, alpha=0.3)

            # current_deriv is now your target derivative (e.g., d4, d5, d10)
            # plt.plot(r_section, current_deriv, "b", label=f"{derivative_order}th Deriv")

            # Calculate threshold using the final target derivative
            threshold_pos = np.argmax(current_deriv[:max_slope_pos])
            ap_threshold = r_section[threshold_pos]
            evt["ap_threshold"] = ap_threshold
            # plt.axhline(0, color="k", linestyle='--')
            # plt.axvline(ap_threshold, color="g", linestyle='--', label="Threshold")
            # plt.title(f"Delete me after... Testing threshold.. {self.time[evt_pos]=}")
            # plt.legend()
            # plt.show(block=False)

            # RAM/SPEED FIX: Replaced python's copy.deepcopy with Numpy's native .copy()
            threshold_segm = reset_array(evt["p_segm"], start_pos + threshold_pos)
            evt["threshold_segm"] = threshold_segm

    # def get_threshold(self):
    #     # SPEED FIX: Use .items() to instantly grab the inner dictionary
    #     for evt_pos, evt in self.events_attrs.items():
    #         start_pos = np.argmin(evt["z_segm"])
    #         peak_pos = np.argmax(evt["p_segm"])
    #         end_pos = peak_pos
    #         r_section = evt["r_segm"][start_pos:end_pos]
    #         d1_section = evt["d_segm"][start_pos:end_pos]
    #         max_slope_pos = np.argmax(d1_section)
    #         d2_section = differentiate(d1_section, 1.0)
    #         d3_section = differentiate(d2_section, 1.0)
    #         d4_section = differentiate(d3_section, 1.0)
    #         plt.figure()
    #         plt.plot(r_section, d1_section, "k")
    #         plt.plot(r_section, d2_section, "r")
    #         plt.plot(r_section, d3_section, "b")
    #         plt.plot(r_section, d4_section, "b")
    #         threshold_pos = np.argmax(d4_section[:max_slope_pos])
    #         ap_threshold = r_section[threshold_pos]
    #         evt["ap_threshold"] = ap_threshold
    #         plt.axvline(ap_threshold, color="g", linestyle='--')
    #         plt.title(f"Delete me after... Testing threshold.. {self.time[evt_pos]=}")
    #         plt.show(block=False)
    #
    #         # RAM/SPEED FIX: Replaced python's copy.deepcopy with Numpy's native .copy()
    #         # temp_p_segm = evt["p_segm"].copy()
    #         threshold_segm = reset_array(evt["p_segm"], start_pos + threshold_pos)
    #         evt["threshold_segm"] = threshold_segm

    @timing
    def get_half_width(self):
        print(f"Not implemented!!! {self}")

    @timing
    def burst(self, kernel_length=0.5):
        self.burst_attrs = {}
        freq_timecourse = np.array([self.get_arr("t_o_p"), self.get_arr("r_ifreq")])
        freq_timecourse[1][0] = np.nanmin(freq_timecourse[1])
        freq_timecourse_full = np.zeros_like(self.time)
        # 1. Create a boolean mask where the full time array matches the event times
        mask = np.isin(self.time, freq_timecourse[0])
        # 2. Assign the frequencies into those specific 'True' slots
        freq_timecourse_full[mask] = freq_timecourse[1]
        n_p = vtp(kernel_length, self.t_delta)
        smoothed_freq = smoothing(freq_timecourse_full, n_p, 2)
        kernel = conv_vector(n_p, 'g', 2)
        single_pulse = np.zeros(2 * n_p)
        single_pulse[n_p] = freq_timecourse[1][0]
        min_convolved = fftconvolve(single_pulse, kernel, mode='same')
        min_val = min_convolved[min_convolved > 0.0].max()
        smoothed_freq_bool = smoothed_freq > min_val
        smoothed_freq_norm = np.where(smoothed_freq_bool, 1.0, 0.0)
        # 'label' returns the labeled array and the number of features found
        labeled_array, section_count = label(smoothed_freq_norm)
        print(f"Number of sections (before): {section_count}")
        # OPTIMIZATION: Get slices for every burst in a single pass
        burst_slices = find_objects(labeled_array)
        # for burst_index in range(1, section_count + 1):
        for burst_index, burst_slice in enumerate(burst_slices, 1):
            if burst_slice is None:
                continue
            # burst_slice is a tuple containing a 1D slice: (slice(start, end),)
            sl = burst_slice[0]
            # 1. INSTANTLY slice the data. No full-array searching.
            temp_burst_freq = freq_timecourse_full[sl]
            temp_burst_time = self.time[sl]
            temp_nonzero_freq = temp_burst_freq > 0
            temp_events_count = np.sum(temp_nonzero_freq)
            # 2. Reset the burst area to 0 directly using the slice
            smoothed_freq_norm[sl] = 0.0
            if temp_nonzero_freq.any() and temp_events_count > 1:
                tmp_avg_freq = np.average(temp_burst_freq[temp_nonzero_freq])
                tmp_time_arr = temp_burst_time[temp_nonzero_freq]
                tmp_min_time = tmp_time_arr.min()
                tmp_max_time = tmp_time_arr.max()
                # These are local searches on the tiny sliced array (Fast)
                tmp_min_time_pos = np.where(temp_burst_time == tmp_min_time)[0][0]
                tmp_max_time_pos = np.where(temp_burst_time == tmp_max_time)[0][0]
                # 3. Target the exact sub-slice directly using standard math
                target_slice = slice(sl.start + tmp_min_time_pos, sl.start + tmp_max_time_pos)
                smoothed_freq_norm[target_slice] = tmp_avg_freq
                # 4. INSTANTLY calculate the global position using the slice start index
                tmp_burst_pos = sl.start + tmp_min_time_pos
                mask = (freq_timecourse[0] >= tmp_min_time) & (freq_timecourse[0] <= tmp_max_time)
                self.burst_attrs[tmp_burst_pos] = dict(
                        start_time=tmp_min_time,
                        end_time=tmp_max_time,
                        burst_length=(tmp_max_time - tmp_min_time),
                        avg_freq=tmp_avg_freq,
                        burst_freq_power=tmp_avg_freq * temp_events_count,
                        burst_index=burst_index,
                        burst_depolarization=np.average(self.resp[sl]),
                        burst_freq_integration=calculate_area(
                                freq_timecourse[0][mask], freq_timecourse[1][mask] ** 2
                                ) / np.abs(np.average(self.resp[sl]) + 40),
                        )
        self.freq_blocks = smoothed_freq_norm
        # Relabeling to ensure continuous numbering if any bursts were disqualified (< 2 APs)
        labeled_array, section_count = label(smoothed_freq_norm)
        for burst_index, (burst_pos, attrs) in enumerate(self.burst_attrs.items(), 1):
            attrs['burst_index'] = burst_index
        print(f"Number of sections (after): {section_count}")

    @timing
    def get_correlation(self, start=0, end=1800):
        # 1. Extract the 'Gold Standard' Baseline Template
        # Using : for slicing ensures we get a range of values
        idx_start = int(vtp(start, self.t_delta))
        idx_end = int(vtp(end, self.t_delta))

        # freq_bool = self.freq_blocks > 0.0
        # freq_norm = np.where(freq_bool, 1.0, 0.0)

        freq_norm = self.freq_blocks

        baseline_template = freq_norm[idx_start: idx_end]

        # 2. Cross-Correlate Template against the ENTIRE recording
        # mode='same' ensures the output length matches self.resp length
        correlation = correlate(freq_norm, baseline_template, mode='same', method='fft')

        # 3. Normalization (0 to 1 scale for easier interpretation)
        max_val = np.max(np.abs(correlation))
        if max_val != 0:
            correlation /= max_val

        # 4. Plotting the full timeline
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Plot 1: The full voltage trace (Baseline -> RF -> Post-RF)
        ax1.plot(self.time, freq_norm, color='gray', alpha=0.5, label='Full Recording')

        # Visual cues for different phases (adjust times based on your protocol)
        ax1.axvspan(start, end, color='green', alpha=0.15, label='Baseline Kernel')
        ax1.axvspan(600, 1200, color='red', alpha=0.1, label='RF Phase')
        ax1.axvspan(1200, self.time[-1], color='blue', alpha=0.1, label='Post-RF / Recovery')

        ax1.set_ylabel("Voltage (mV)")
        ax1.set_title("Full Session Trace: Baseline, RF, and Recovery")
        ax1.legend(loc='upper right')

        # Plot 2: Correlation Strength across the entire timeline
        ax2.plot(self.time, correlation, color='darkblue', linewidth=1)
        ax2.set_ylabel("Similarity Index")
        ax2.set_xlabel("Time (s)")
        ax2.set_title("Cross-Correlation Strength vs. Baseline Template")
        ax2.grid(True, linestyle=':', alpha=0.7)

        plt.tight_layout()
        plt.show(block=False)

        return correlation

    @timing
    def show_all_events(
            self, title: str = 'Recording with selected events', show_events: bool = True, adjust=True
            ) -> None:
        fig, ax = plt.subplots(figsize=(5, 2.5))
        plt.rcParams.update({'font.size': 8})
        ax.axhline(y=0.0, color="k", linestyle='--')
        ax.plot(self.time, self.resp, "k", linewidth=3, alpha=0.1)

        if show_events:
            for evt in self.events_attrs.values():
                t_o_p = evt["t_o_p"]
                t_segm = evt["t_segm"]
                p_segm = evt["p_segm"]
                s_segm = evt["s_segm"]
                z_segm = evt["z_segm"]
                amplitude = evt["amplitude"]
                b_amp = evt["b_amp"]
                threshold_segm = evt["threshold_segm"]

                if not adjust:
                    amplitude -= b_amp
                if threshold_segm is not None:
                    ap_threshold = evt["ap_threshold"]
                    ax.plot(t_segm + t_o_p, b_amp + threshold_segm * (ap_threshold - b_amp), "bo")

                ax.plot(t_segm + t_o_p, b_amp + z_segm * amplitude / 4, "g")
                ax.plot(t_segm + t_o_p, b_amp + s_segm * amplitude / 2, "b")
                ax.plot(t_segm + t_o_p, b_amp + p_segm * amplitude, "r", linewidth=1)

        if self.freq_blocks.size:
            ax.plot(self.time, self.freq_blocks, "g", lw=4.0)
            burst_freq_power = np.array(
                    [
                            self.get_arr("start_time", "burst"),
                            self.get_arr("burst_freq_power", "burst")
                            ]
                    )
            ax.plot(burst_freq_power[0], burst_freq_power[1], "bo", ms=10.0, alpha=0.5)
            burst_freq_integration = np.array(
                    [
                            self.get_arr("start_time", "burst"),
                            self.get_arr("burst_freq_integration", "burst")
                            ]
                    )
            ax.plot(burst_freq_integration[0], burst_freq_integration[1], "b+", ms=10.0, alpha=0.5)

        ax.set_title(f"{title} {len(self.events_attrs)}.")

        # ---------------------------------------------------------
        # ASYNC RAM CLEARING BLOCK (Non-Blocking)
        # ---------------------------------------------------------
        def on_close(event):
            # When the user clicks the "X", wipe this specific figure from RAM
            event.canvas.figure.clear()
            plt.close(event.canvas.figure)
            import gc

            gc.collect()

        # Connect the callback
        fig.canvas.mpl_connect('close_event', on_close)

        # Show the plot and immediately return control to the main script
        plt.show(block=False)
        fig.canvas.draw()

    # def show_all_events(
    #         self, title: str = 'Recording with selected events', show_events: bool = True, adjust=True
    #         ) -> None:
    #     plt.figure(figsize=(5, 2.5))
    #     plt.rcParams.update({'font.size': 8})
    #     plt.axhline(y=0.0, color="k", linestyle='--')
    #     plt.plot(self.time, self.resp, "k", linewidth=3, alpha=0.1)
    #     if show_events:
    #         # SPEED FIX: Iterate directly over the values. This binds 'evt' to the inner dictionary,
    #         # completely bypassing the need to look up the key 8+ times per loop.
    #         for evt in self.events_attrs.values():
    #             t_o_p = evt["t_o_p"]
    #             t_segm = evt["t_segm"]
    #             p_segm = evt["p_segm"]
    #             s_segm = evt["s_segm"]
    #             z_segm = evt["z_segm"]
    #             amplitude = evt["amplitude"]
    #             b_amp = evt["b_amp"]
    #             threshold_segm = evt["threshold_segm"]
    #             if not adjust:
    #                 amplitude -= b_amp
    #             if threshold_segm is not None:
    #                 ap_threshold = evt["ap_threshold"]
    #                 plt.plot(t_segm + t_o_p, b_amp + threshold_segm * (ap_threshold - b_amp), "bo")
    #             plt.plot(t_segm + t_o_p, b_amp + z_segm * amplitude / 4, "g")
    #             plt.plot(t_segm + t_o_p, b_amp + s_segm * amplitude / 2, "b")
    #             plt.plot(t_segm + t_o_p, b_amp + p_segm * amplitude, "r", linewidth=1)
    #     if self.freq_blocks.size:
    #         plt.plot(self.time, self.freq_blocks, "g", lw=4.0)
    #         burst_freq_power = np.array(
    #                 [
    #                         self.get_arr("start_time", "burst"),
    #                         self.get_arr("burst_freq_power", "burst")
    #                         ]
    #                 )
    #         plt.plot(burst_freq_power[0], burst_freq_power[1], "bo", ms=10.0, alpha=0.5)
    #         burst_freq_integration = np.array(
    #                 [
    #                         self.get_arr("start_time", "burst"),
    #                         self.get_arr("burst_freq_integration", "burst")
    #                         ]
    #                 )
    #         plt.plot(burst_freq_integration[0], burst_freq_integration[1], "b+", ms=10.0, alpha=0.5)
    #     plt.title(f"{title} {len(self.events_attrs)}.")
    #     plt.show(block=False)

    @timing
    @timing
    def show_events_aligned(self, title):
        r_segm_arr = self.get_arr("r_segm")
        t_segm_arr = self.get_arr("t_segm")

        events_num = len(r_segm_arr)
        if events_num == 0:
            print("No events to plot.")
            return

        alpha_val = 1.0 / (events_num + 1) + 0.03
        fig, ax = plt.subplots(figsize=(3, 2.5))

        for event in r_segm_arr:
            ax.plot(self.common_time, event, "k", alpha=alpha_val)

        ax.plot(
                np.nanmean(t_segm_arr, axis=0),
                np.nanmean(r_segm_arr, axis=0),
                "r:"
                )

        ax.axhline(y=0.0, color='r', linestyle='dashed')
        ax.set_title(f"{title} {events_num}.")

        # ---------------------------------------------------------
        # ASYNC RAM CLEARING BLOCK (Non-Blocking)
        # ---------------------------------------------------------
        def on_close(event):
            event.canvas.figure.clear()
            plt.close(event.canvas.figure)
            import gc

            gc.collect()

        fig.canvas.mpl_connect('close_event', on_close)

        plt.show(block=False)
        fig.canvas.draw()
    # def show_events_aligned(self, title):
    #     # SPEED/RAM FIX: Extract arrays exactly once and cache them locally
    #     r_segm_arr = self.get_arr("r_segm")
    #     t_segm_arr = self.get_arr("t_segm")
    #
    #     events_num = len(r_segm_arr)
    #     if events_num == 0:
    #         print("No events to plot.")
    #         return
    #
    #     alpha_val = 1.0 / (events_num + 1) + 0.03
    #     plt.figure(figsize=(3, 2.5))
    #
    #     # RAM FIX: Iterate directly over the cached array.
    #     # This completely removes the need for np.concatenate and massive matrix transpositions (.T)
    #     for event in r_segm_arr:
    #         plt.plot(self.common_time, event, "k", alpha=alpha_val)
    #
    #     # SPEED FIX: Use the cached arrays for the nanmean calculations
    #     plt.plot(
    #             np.nanmean(t_segm_arr, axis=0),
    #             np.nanmean(r_segm_arr, axis=0),
    #             "r:"
    #             )
    #
    #     plt.axhline(y=0.0, color='r', linestyle='dashed')
    #     plt.title(f"{title} {events_num}.")
    #     plt.show(block=False)
