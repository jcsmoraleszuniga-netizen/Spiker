import copy
from itertools import pairwise, repeat
from pprint import pprint
from typing import List, Any

from fontTools.unicodedata import block
from numpy import ndarray, dtype, flipud
from pyabf import ABF
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
from lib_utility import (
    exp_fit, parabolic_fit, remove_nan, vtp, conv_vector, differentiate, find_over_threshold, find_peaks, opt_linear,
    linear,
    opt_expdec,
    crossing_point, extender, exp_decay, apply_by, exp_to_lin, mse, timing, prev_change,
    down_sample_function, ptv, smoothing, reset_array, split_position, remove_shift, file_info,
    calculate_area, correct_bound
    )
from copy import copy as cp_copy
from scipy import fft
from matplotlib.ticker import ScalarFormatter, FuncFormatter


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
def plot_rec(rec, der, title="No Title.", values=(0.0, 0.0), mode="full", factor=1.0):
    """
    Displays a plot of the original data, optionally with vertical lines.
    """
    plt.figure(figsize=(5, 2.5))
    plt.axhline(y=0.0, color="k", linestyle='--')
    for x_value in values:
        plt.axvline(x=x_value, color="r", linestyle='--')
    match mode:
        case "basic":
            plt.plot(rec.time, rec.resp)
        case "derivative":
            plt.plot(rec.time, rec.resp, "k")
            plt.plot(der.time, der.resp * rec.t_delta, "r")
            plt.plot(rec.time, rec.peak_noise, "k:")
            plt.plot(der.time, der.peak_noise * der.t_delta, "r:")
        case "full":
            plt.figure()
            plt.plot(rec.time, rec.resp, "k")
            plt.plot(der.time, der.resp * rec.t_delta, "r")
            plt.plot(rec.time, rec.peaks * 3.0 * factor, "k", linewidth=3.0)
            plt.plot(rec.time, rec.der_peaks * 2.0 * factor, "g", linewidth=3.0)
            plt.plot(rec.time, rec.zero_pass * 1.0 * factor, "b", linewidth=3.0)
            plt.plot(rec.time, rec.peak_noise, "k:")
            plt.plot(der.time, der.peak_noise * der.t_delta, "r:")
    plt.title(title)
    plt.show(block=False)


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

    def __init__(self, path_to_file, initialize=True):
        super().__init__(path_to_file)
        print(f"Initializing {self = }")
        self.mode = "continuous"
        if initialize:
            self._initialize()

    @timing
    def _get_resp(self):
        self.resp = self.data[0]

    @timing
    def _get_time(self):  # TODO assess the convenience of using "absoluteTime: bool = True"
        self.setSweep(0)
        time = self.sweepX
        self.t_delta: np.floating = np.round(self.sweepX[1] - self.sweepX[0], decimals=4)  # time increment
        cont_time = [(time := self.sweepX + (time[-1] + self.t_delta)) for _ in repeat(None, self.sweepCount - 1)]
        cont_time.insert(0, self.sweepX)
        self.time = np.array(tuple(cont_time)).flatten()

    @timing
    def _get_cdac(self):
        sweep_dac = self.sweepC
        self.cdac = np.array([sweep_dac for _ in repeat(None, self.sweepCount)]).flatten()

    @timing
    def _initialize(self):
        self._get_resp()
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
        self.t_delta = np.round(self.record[0][1] - self.record[0][0], decimals=4)  # time increment

    @timing
    def _initialize(self):
        self._get_resp()
        self._get_time()


class loadRecord(base):
    """Loads files and perform basic data manipulation"""
    instance_number = 0

    def __init__(self, path_to_file, initialize=True):
        super().__init__()
        loadRecord.instance_number += 1
        self._instance_number = loadRecord.instance_number
        self._path_to_file = path_to_file
        self.direction = 0

        if path_to_file.lower().endswith(".abf"):
            print(f"Using {abf_numpy = }")
            self.data_object = abf_numpy(path_to_file, initialize)
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
        start_pos = end_pos = 0
        try:
            start_pos = np.where(start == self.time)[0][0]
        except IndexError:
            if start < self.time[0]:
                print(f"Using {0} {self.time[0] = }")
                start_pos = 0
        try:
            end_pos = np.where(end == self.time)[0][0]
        except IndexError:
            if end > self.time[-1]:
                print(f"Using {len(self.time) = } {self.time[-1] = }")
                end_pos = len(self.time)

        self.resp = self.resp[start_pos:end_pos]
        self.time = self.time[start_pos:end_pos]
        self.cdac = self.cdac[start_pos:end_pos]

    @timing
    def down_sample(self, down_sample=10):
        self.time = down_sample_function(self.time, down_sample)
        self.resp = down_sample_function(self.resp, down_sample)
        self.cdac = down_sample_function(self.cdac, down_sample)

    @timing
    def get_stack(self, size=2):
        # TODO implement this method with a dictionary, and add others like self.derv to this dictionary
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

    def __init__(self, path_to_file, initialize=True):
        super().__init__(path_to_file, initialize)
        self.fft_series = np.array([])
        self.fft_domain = np.array([])
        self.freq_increment = 0

    @timing
    def _get_fft_series(self, option):
        self.fft_series = fft.rfft(option)

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
    def fft_plot(self, title="Theoretical FFT"):
        print(f"A phase")
        f_r_theoretical = self.fft_domain * (1 / self.t_delta)  # Restores the proportions
        f_s_theoretical = 2 * np.abs(self.fft_series) / len(self.time)  # Restores the proportions
        print(f"B phase")
        plt.figure()
        plt.title(title)
        plt.plot(f_r_theoretical, f_s_theoretical, linewidth=0.05)
        print(f"C phase")
        plt.axhline()
        plt.yscale('log', base=np.e)
        two_decimal_lambda_formatter = FuncFormatter(lambda x, pos: f"{x:.3f}")
        plt.gca().yaxis.set_major_formatter(two_decimal_lambda_formatter)
        plt.axhline(y=10, color='r', linestyle='dashed', label="10 [pA]")
        plt.axhline(y=0.0001, color='k', linestyle='dashed', label="0.0001 [pA]")
        plt.xlabel("Frequencies [Hz]")
        plt.ylabel("Amplitude (Log Scale)")
        plt.legend(loc="upper right")
        plt.grid(True, which="both", ls="-", lw=0.5)  # Add a grid for better readability on log scales
        plt.show(block=False)


class Analyzer(Fourier):
    """Analyzes the recordings"""

    def __init__(self, path_to_file="", initialize=True):
        super().__init__(path_to_file, initialize)
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
    def get_smooth(self, smooth_width=0.001, repetitions=1, sharpness=4):
        print(f"{smooth_width = } {self.t_delta = }")
        n_p = vtp(smooth_width, self.t_delta)
        self.resp = smoothing(self.resp, n_p, repetitions, sharpness)

    @timing
    def get_section_area(self, baseline_start=371, baseline_end=376, response_end=441, linear_fit=False):
        """Calculates the area, peak and rTTP"""
        b_s_p = np.where(self.time == baseline_start)[0][0]
        b_e_p = np.where(self.time == baseline_end)[0][0]
        r_e_p = np.where(self.time == response_end)[0][0]
        points = vtp(0.002, self.t_delta)

        shift = np.nanmean(self.resp[b_s_p:b_e_p])

        resp = smoothing(copy.deepcopy(self.resp[b_e_p:r_e_p]) - shift, points, 4, 2)
        r_time = copy.deepcopy(self.time[b_e_p:r_e_p])
        base = smoothing(copy.deepcopy(self.resp[b_s_p:b_e_p]) - shift, points, 4, 2)
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
    def find_pulses(self, direction: int = -1) -> None:
        """It gives the position of the start of the control pulses"""
        der_test_resp = differentiate(self.cdac, self.t_delta)
        der_test_resp = prev_change(der_test_resp)
        threshold = der_test_resp * 0.0 + self.direction * 9 * np.std(der_test_resp)
        o_thresh = find_over_threshold(der_test_resp, threshold, direction)
        self.pulses_peaks = find_peaks(o_thresh, der_test_resp, direction, self.t_delta, self.t_delta)
        self.pul_attrs = {evt_pos: {} for evt_pos, _ in enumerate(self.pulses_peaks) if self.pulses_peaks[evt_pos]}

    @timing
    def del_pulses(self, del_length=0.75, target_val=2000.0):
        peaks = self.pulses_peaks
        resp_copy = copy.deepcopy(self.resp)
        time_copy = copy.deepcopy(self.time)
        # cdac_copy = copy.deepcopy(self.cdac)
        del_range = vtp(del_length, self.t_delta)
        for pos, val in enumerate(peaks):
            if val:
                resp_copy[pos:pos + del_range] = target_val  # TODO verify these arbitrary positions
        index = np.argwhere(resp_copy == target_val)
        self.time = np.delete(time_copy, index)
        self.resp = np.delete(resp_copy, index)
        # self.resp = np.delete(cdac_copy, index)

    @timing
    def get_pk_noise(self, time_frame=0.2, n_deviations=3, resp_increment=0.5, std_increment=10, sharpness=2):
        n_p = vtp(time_frame, self.t_delta)
        # fftconvolve is better for long arrays  # TODO make a function of this
        smoothed_resp = fftconvolve(self.resp, conv_vector(n_p, 'g', sharpness), mode='same')  # response
        # print(f"{smoothed_resp = }")
        # Calculates the standard deviation every resp_increment
        std_resp = apply_by(np.std, np.array([self.time, self.resp]), resp_increment, True)
        # print(f"{std_resp = }  {std_increment = }")
        # Selects the minimum value every std_increment
        if std_increment > self.time[-1]:
            print(f"Warning: {std_increment = } is bigger than the time interval {self.time[-1] = }.")
            std_increment = int(self.time[-1])
        std_min = apply_by(np.min, std_resp, std_increment, True)
        # print(f"{std_min = }")
        self.std = np.nanmean(std_min[1])
        self.peak_noise = smoothed_resp + np.interp(
                self.time, std_min[0],
                std_min[1] * n_deviations * self.direction
                )

    @timing
    def get_derv(self):
        self.derivative = differentiate(self.resp, self.t_delta)

    @timing
    def get_o_thresh(self):
        self.o_thresh = find_over_threshold(self.resp, self.peak_noise, self.direction)

    @timing
    def get_peaks(self, search_width=0.01, shift_time=0.001):
        self.get_o_thresh()
        self.peaks = find_peaks(self.o_thresh, self.resp, self.direction, search_width, self.t_delta)
        # TODO make a function out of this
        # This block is for removing the false positives generated by the peaks of the artifacts
        if len(self.pulses_peaks):
            tmp_peaks = np.copy(self.peaks)
            shift = vtp(shift_time, self.t_delta)
            for peak_pos, peak_value in enumerate(tmp_peaks):
                length_value = len(self.pulses_peaks[peak_pos - shift:peak_pos + shift])
                if length_value and peak_value and np.max(self.pulses_peaks[peak_pos - shift:peak_pos + shift]):
                    self.peaks[peak_pos] = 0.0
            # self.peaks = tmp_peaks

    @timing
    def get_z_pass(self, zero_pass_frame=0.005, delete_peaks=True):
        spaces = vtp(zero_pass_frame, self.t_delta)
        zero_pass = np.zeros(len(self.peaks))
        max_length = len(self.peaks)
        tmp_peaks = np.copy(self.peaks)
        for evt_pos, value in enumerate(tmp_peaks):
            if value and spaces <= evt_pos <= max_length - spaces:
                front = np.copy(self.resp[evt_pos:evt_pos + spaces])
                back = np.copy(flipud(self.resp[evt_pos - spaces:evt_pos + 1]))
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
                    print(f"{(aft_zero, bef_zero) = }")
                    if delete_peaks:
                        print(f"Peak {evt_pos} deleted")
                        self.peaks[evt_pos] = 0
                else:
                    zero_pass[evt_pos - bef_zero] = -1
                    zero_pass[evt_pos + aft_zero] = 1

        self.zero_pass = zero_pass

    @timing
    def get_rescap(self, std_max=0.1):  # Use a low value for restriction, like 0.1
        test_response = self.resp  # * np.where(self.mask == 1, 0, 1)  # Inverted mask # Common for both calculations
        peaks = self.pulses_peaks
        mem_res = []  # Specific for each calculation
        mem_cap = []  # Specific for each calculation
        beg_res = vtp(0.2, self.t_delta)
        end_res = vtp(0.35, self.t_delta)
        beg_cap = vtp(0.005, self.t_delta)
        end_cap = vtp(0.1, self.t_delta)
        for pos in np.arange(len(peaks)):
            if peaks[pos]:
                pos += 1  # Shift correction
                # Membrane resistance (input resistance) calculations and fitting
                current_res = -self.cdac[pos + beg_res:pos + end_res] - self.cdac[pos + end_res]
                voltage_res = -test_response[pos + beg_res:pos + end_res] + test_response[pos + end_res - 1]
                pr_lin = opt_linear(current_res, voltage_res)
                # Membrane capacitance calculations and fitting
                time_cap = self.time[pos + beg_cap:pos + end_cap] - self.time[pos + beg_cap]
                voltage_cap = test_response[pos + beg_cap:pos + end_cap] - test_response[pos + end_cap]
                pr_exp = opt_expdec(time_cap, voltage_cap)
                if np.all(np.sqrt(np.diag(pr_lin[1])) < std_max) and pr_lin[0][0] > 0:  # STD condition linear fit
                    mem_res.append((self.time[pos], pr_lin[0][0] * 10 ** 3))  # In mega ohms
                    if np.all(np.sqrt(np.diag(pr_exp[1])) < std_max) and pr_exp[0][2] > 0:  # STD condition exp fit
                        mem_cap.append((self.time[pos], pr_exp[0][2] / (pr_lin[0][0] * 10 ** -3)))  # In pF
        self.inp_res = np.array(mem_res)
        self.mem_cap = np.array(mem_cap)

    @timing
    def get_rira(self, beg_ar=0.02, end_ar=0.005, beg_ir=0.25, end_ir=0.750):
        peaks = self.pulses_peaks
        bas_acc = vtp(beg_ar - 0.001, self.t_delta)
        beg_acc = vtp(beg_ar, self.t_delta)
        end_acc = vtp(end_ar, self.t_delta)
        beg_res = vtp(beg_ir, self.t_delta)
        end_res = vtp(end_ir, self.t_delta)
        v_diff = 0
        voltage_input = np.array([])
        for pos in np.arange(len(peaks)):
            if peaks[pos]:
                # Storing the time of the peak
                self.pul_attrs[pos]["t_o_p"] = self.time[pos]
                # Access resistance calculations
                voltage_acc = self.cdac[pos - beg_acc:pos + end_acc]
                current_acc = self.resp[pos - beg_acc:pos + end_acc]
                if v_diff == 0:
                    v_diff = voltage_acc[:bas_acc][0] - np.min(voltage_acc)
                base_current = np.mean(current_acc[:bas_acc])
                c_diff = base_current - np.min(current_acc)
                # Storing access resistance values
                # self.pul_attrs[pos]["acc_res"] = (v_diff / c_diff) * 10 ** 3
                self.pul_attrs[pos]["v_pulse"] = v_diff
                self.pul_attrs[pos]["base_current"] = base_current
                self.pul_attrs[pos]["acc_res"] = (v_diff / c_diff) * 1000
                # Input resistance calculations and fitting
                if len(voltage_input) == 0:
                    voltage_input = self.cdac[pos + beg_res:pos + end_res]
                current_input = self.resp[pos + beg_res:pos + end_res]
                pr_lin = opt_linear(current_input, voltage_input)
                # STD condition linear fit
                # Storing input resistance values
                self.pul_attrs[pos]["input_res"] = pr_lin[0][0] * 1000
                self.pul_attrs[pos]["input_volt"] = voltage_input
                self.pul_attrs[pos]["input_curr"] = current_input
                self.pul_attrs[pos]["input_STD"] = np.sqrt(np.diag(pr_lin[1]))
        self.inp_res = np.array([self.get_pulse_arr("t_o_p"), self.get_pulse_arr("input_res")]).T
        self.acc_res = np.array([self.get_pulse_arr("t_o_p"), self.get_pulse_arr("acc_res")]).T

    @timing
    def get_pulse_arr(self, name):
        return np.array([values[name] for values in self.pul_attrs.values()])


class EvtPro(Analyzer):
    """Event detection class"""

    def __init__(self, path_to_file="", initialize=True):
        super().__init__(path_to_file, initialize)
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
        self._events_positions = np.array([])  # Initialization
        self.common_time = np.array([])  # Initialization

    @timing
    def _select_events(self, slope_peak_time, max_slope):
        initial_msp_p: int = vtp(slope_peak_time, self.t_delta)
        prev_pos: int = 0  # previous peak position, initialization value
        evt_pos: int = 0  # current peak position, initialization value
        self.events_attrs = {}
        peaks_copy = copy.deepcopy(self.peaks)
        last_peak = len(peaks_copy)
        for next_pos, evt_val in enumerate(peaks_copy):
            # Constraints: evt_val > 0 and evt_pos > 0
            # Peaks must be separated by at least peak_to_peak
            if (evt_val and next_pos > evt_pos > prev_pos) or (next_pos == last_peak - 1):
                # Must be 1 peak-slope before of the peak-amplitude
                if evt_pos - prev_pos < initial_msp_p:  # peak-slope must be between two peaks
                    msp_p = evt_pos - prev_pos
                else:
                    msp_p = initial_msp_p
                cond_slope_range = evt_pos - msp_p > 0.0 and evt_pos + 1 - (evt_pos - msp_p) > 0.0
                max_slope_region = slice(evt_pos - msp_p, evt_pos + 1)  # +1 for low rate sampling recordings
                cond_max_slope = np.max(self.der_peaks[max_slope_region])
                if cond_slope_range and cond_max_slope:
                    self.events_attrs[evt_pos] = self.default_event.copy()
                    rel_slope_pos = crossing_point(flipud(self.der_peaks[max_slope_region]))
                    slope_value = flipud(self.derivative[max_slope_region])[rel_slope_pos]
                    slope_time = flipud(self.time[max_slope_region])[rel_slope_pos]
                    # slope_peak_diff = self.time[evt_pos] - flipud(self.time[max_slope_region])[rel_slope_pos]
                    match self.direction:
                        case -1:  # negative going
                            if not max_slope < slope_value:
                                print(
                                        f"{evt_pos}"
                                        f" {self.time[evt_pos]:12.4f}[s] rejected ({max_slope} > {slope_value})"
                                        )
                                self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")
                                prev_pos = evt_pos  # previous peak position
                                evt_pos = next_pos
                                continue
                        case 1:  # positive going
                            if not max_slope > slope_value:
                                print(
                                        f"{evt_pos}"
                                        f" {self.time[evt_pos]:12.4f}[s] rejected ({max_slope} < {slope_value})"
                                        )
                                self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")
                                prev_pos = evt_pos  # previous peak position
                                evt_pos = next_pos
                                continue
                    # max slope position
                    abs_slope_pos = vtp(slope_time - self.time[0], self.t_delta)
                    # Looking for zero-pass rise-start location
                    starting_region = slice(prev_pos, abs_slope_pos + 1)
                    if abs_slope_pos + 1 - prev_pos == 0.0:
                        print(f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected, {abs_slope_pos + 1=} {prev_pos=}")
                        self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")
                        prev_pos = evt_pos  # previous peak position
                        evt_pos = next_pos
                        continue
                    zs_p = crossing_point(flipud(self.zero_pass[starting_region]))
                    # Looking for zero-pass peak location
                    peak_region = slice(abs_slope_pos, next_pos)
                    if next_pos - abs_slope_pos == 0.0:
                        print(f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected, {next_pos=} {abs_slope_pos=}")
                        self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")
                        prev_pos = evt_pos  # previous peak position
                        evt_pos = next_pos
                        continue
                    zp_p = crossing_point(self.zero_pass[peak_region])
                    # if None in (zs_p, zp_p) or zs_p == zp_p:
                    if None in (zs_p, zp_p):
                        print(f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected, {zs_p=} {zp_p=}")
                        self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")
                        prev_pos = evt_pos  # previous peak position
                        evt_pos = next_pos
                        continue
                    t_o_zs = flipud(self.time[starting_region])[zs_p]  # zero-pass rise-start time
                    t_o_zp = self.time[peak_region][zp_p]  # zero-pass peak time
                    self.events_attrs[evt_pos]["t_o_s"] = slope_time
                    self.events_attrs[evt_pos]["slope_peak_delta"] = self.time[evt_pos] - slope_time
                    self.events_attrs[evt_pos]["rise_slope_val"] = slope_value
                    self.events_attrs[evt_pos]["t_o_zs"] = t_o_zs
                    self.events_attrs[evt_pos]["t_o_zp"] = t_o_zp

                    self.events_attrs[evt_pos]["peak_error"] = self.time[evt_pos] - t_o_zp
                    self.events_attrs[evt_pos]["rise_time_peak"] = t_o_zp - t_o_zs  # TODO working here!!
                    self.events_attrs[evt_pos]["rise_time_der"] = self.time[evt_pos] - t_o_zs  # TODO working here!!

                    # Relative to peak positions
                    self.events_attrs[evt_pos]["slope_pos_delta"] = evt_pos - abs_slope_pos
                    self.events_attrs[evt_pos]["start_pos_delta"] = evt_pos - vtp(t_o_zs - self.time[0], self.t_delta)
                    self.events_attrs[evt_pos]["peak_pos_delta"] = evt_pos - vtp(t_o_zp - self.time[0], self.t_delta)
                    prev_pos = evt_pos  # previous peak position
                    evt_pos = next_pos
                else:
                    print(f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected ({evt_pos - msp_p=}, No slope found.)")
                    prev_pos = evt_pos  # previous peak position
                    evt_pos = next_pos
            elif evt_val and evt_pos == 0 and prev_pos == 0:
                evt_pos = next_pos
                print(f"{next_pos=}  {evt_pos=}  {prev_pos=}")
            elif evt_val and next_pos == evt_pos:
                evt_pos = next_pos
                print(f"{next_pos=}  {evt_pos=}  {prev_pos=}")
            elif evt_val and evt_pos > 0:
                print(f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected ({evt_pos - prev_pos=})")
        print(f" Accepted events: {len(self.events_attrs)}, Rejected: {np.sum(self.peaks) - len(self.events_attrs)}")

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
                evt_pos]) > peak_to_peak  # TODO working here!!
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
                end_roi_pos = interference_pos
            else:
                end_roi_pos = evt_pos + a_p
            # if end_roi_pos < evt_pos + a_p:  # delete me after, just for testing purposes
            #     print(
            #             f"{self.time[evt_pos]=:4.4f}  "
            #             f"{self.time[end_roi_pos]=:4.4f}  "
            #             f"{self.time[end_roi_pos] - self.time[evt_pos]=:4.4f}"
            #             )

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
                        f" {t_o_zs=:5.4f} {t_o_zp=:5.4f} {t_segm[0]=:5.4f}  {t_segm[-1]=:5.4f}."
                        f" {abs(peak_error)=:2.4f}  {zp_to_pp=}"
                        )
                self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")
                continue

            #########################################################
            rise_time_peak = self.events_attrs[evt_pos]["rise_time_peak"]
            rise_time_der = self.events_attrs[evt_pos]["rise_time_der"]
            if rise_time_peak > max_rise_time or rise_time_der > max_rise_time:
                print(
                        f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected,"
                        f" {rise_time_peak=:5.4f} {rise_time_der=:5.4f} {max_rise_time=:5.4f}."
                        )
                self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")
                continue
            #########################################################

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
    def get_alig(self, alignment='p'):
        t_o_p = 0  # time of peak
        evts_attrs_copy = copy.deepcopy(self.events_attrs)
        for evt_pos in evts_attrs_copy.keys():
            match alignment:
                case 'p':  # aligned to the peak
                    t_o_p = self.time[evt_pos]
                case 'z':  # aligned to the estimated peak using the zero pass from the 1st derivative
                    t_o_p = self.events_attrs[evt_pos]["t_o_zp"]
                case 's':  # aligned to the estimated max rising slope using the der peaks from the 1st derivative
                    t_o_p = self.events_attrs[evt_pos]["t_o_s"]
            t_segm = self.events_attrs[evt_pos]["t_segm"]
            # Shifting for correct alignment
            t_segm = np.round(t_segm - t_o_p, decimals=4)
            self.events_attrs[evt_pos]["t_o_p"] = t_o_p  # time of peak value{
            self.events_attrs[evt_pos]["t_segm"] = t_segm

    @timing
    def _get_adj(self, baseline_time=0.005, adjust=True):
        base_pos = vtp(baseline_time, self.t_delta)
        evts_attrs_copy = copy.deepcopy(self.events_attrs)
        for evt_pos in evts_attrs_copy:
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
            # Subtracts the baseline
            if adjust:
                self.events_attrs[evt_pos]["r_segm"] = r_segm - self.events_attrs[evt_pos]["b_amp"]

    @timing
    def get_amplitudes(
            self, min_ampl: float = -1.48, baseline_time: float = 0.005,
            peak_radius: float = 0.0, peak_type: str = 'p', adjust=True
            ) -> None:
        """Select the events based in the amplitude range.
        The events must be aligned previously to this analysis."""
        self._get_adj(baseline_time, adjust)
        p_r_p = vtp(peak_radius, self.t_delta)
        evts_attrs_copy = copy.deepcopy(self.events_attrs)
        for evt_pos in evts_attrs_copy.keys():
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
            # amplitude and direction must have the same sign
            if amplitude * self.direction > min_ampl * self.direction:
                self.events_attrs[evt_pos]["amplitude"] = amplitude
            else:
                print(f"Event at {self.time[evt_pos]:12.4f}[s] rejected {min_ampl=:6.2f} and {amplitude:6.2f}")
                self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")
        print(f" Accepted events: {len(self.events_attrs)}, Rejected: {len(evts_attrs_copy) - len(self.events_attrs)}")

    @timing
    def get_arr(self, name, element_type="evt"):
        match element_type:
            case "evt":
                return np.array([values[name] for values in self.events_attrs.values()])
            case "pul":
                return np.array([values[name] for values in self.pul_attrs.values()])
        return None

    @timing
    def get_frequencies(self):
        # inst_freqs = np.append([0.0], 1.0 / np.diff(self.get_arr("t_o_p")))  # Just for compatibility
        inst_freqs = [np.nan] + list(1.0 / np.diff(self.get_arr("t_o_p")))
        # print(f"{inst_freqs=}")
        for evt_pos, r_ifreq in zip(self.events_attrs, inst_freqs):
            self.events_attrs[evt_pos]["r_ifreq"] = r_ifreq

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
            self, gaussian_window=0.002, fit_beg=0.00, fit_end=0.015, pearson_r_min=0.8, repetitions=3, sharpness=2,
            tau_min=0.001, tau_max=0.01, normal_mse_fit_max=3.0, n_limit=100, std=1.0
            ):
        """Select the events based in the fitting to exponential (decay).
        The events must be aligned previously to this analysis."""
        # TODO make this function use p_segm and all the parameters of the slope
        evts_attrs_copy = copy.deepcopy(self.events_attrs)
        n_p = vtp(gaussian_window, self.t_delta)
        local_exp_fit = exp_fit  # Local assignment for fast lookup on local scope
        local_mse = mse  # Local assignment for fast lookup on local scope

        # plt.figure(figsize=(5, 2.5))  # Delete

        for evt_pos in evts_attrs_copy.keys():
            r_segm = self.events_attrs[evt_pos]["r_segm"]
            t_segm = self.events_attrs[evt_pos]["t_segm"]
            p_segm = self.events_attrs[evt_pos]["p_segm"]
            pos_fit_end = vtp(fit_end, self.t_delta)
            pos_fit_start = vtp(fit_beg, self.t_delta)
            ds_end = 0
            peak_pos = np.argmax(p_segm)
            smoothed_resp = smoothing(r_segm, n_p, repetitions, sharpness)
            fit_region = slice(peak_pos + pos_fit_start, peak_pos + pos_fit_end)
            s_resp_s = smoothed_resp[fit_region]
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
            # t_short = t_segm[fit_region_restricted] - t_segm[peak_pos + pos_fit_start]
            t_short = t_segm[fit_region_restricted]
            if ds_end == 0 or r_s_short.size == 0 or r_s_short.size != t_short.size:
                print(f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected {r_s_short.size=} {t_short.size=}")
                self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")
                continue
            fit_i_0, fit_pk0, fit_t0, pearson_r = local_exp_fit(
                    r_s_short,
                    t_short - t_segm[peak_pos + pos_fit_start],
                    self.direction
                    )
            # mse calculation
            r_short = r_segm[fit_region_restricted]
            mse_fit = local_mse(
                    r_short,
                    fit_pk0 * np.exp((t_short - t_segm[peak_pos + pos_fit_start]) / fit_t0) + fit_i_0
                    )
            normal_mse_fit = mse_fit / self.events_attrs[evt_pos]["amplitude"]
            # Conditions for acceptance
            pearson_condition = abs(pearson_r) >= pearson_r_min
            mse_condition = abs(normal_mse_fit) <= normal_mse_fit_max
            tau_condition = tau_max >= abs(fit_t0) >= tau_min
            fit_i_0_condition = abs(fit_i_0) <= abs(n_limit * std)
            direction_condition = fit_pk0 * self.direction - fit_i_0 * self.direction > std
            if pearson_condition and mse_condition and tau_condition and fit_i_0_condition and direction_condition:
                self.events_attrs[evt_pos]["fit_min"] = fit_i_0
                self.events_attrs[evt_pos]["fit_peak"] = fit_pk0
                self.events_attrs[evt_pos]["tau"] = fit_t0
                self.events_attrs[evt_pos]["r_decay"] = pearson_r
                self.events_attrs[evt_pos]["mse_fit"] = mse_fit

                # plt.figure(figsize=(5, 2.5))  # Delete
                # plt.rcParams.update({'font.size': 8})
                # plt.axhline(0.0, color="k", linestyle='--')  # Delete
                # plt.axhline(fit_i_0, color="b", linestyle='--')  # Delete
                # plt.plot(t_segm, r_segm, "r")  # Delete
                # label_tmp = f"{pearson_r=:.4f} {mse_fit=:.4f} {normal_mse_fit=:.4f}"
                # t_peak_end = t_segm[peak_pos + pos_fit_start:]
                # t_start = t_segm[peak_pos + pos_fit_start]
                # plt.plot(
                #         t_peak_end,
                #         fit_pk0 * np.exp((t_peak_end - t_start) / fit_t0) + fit_i_0,
                #         "b:",
                #         label=label_tmp
                #         )  # Delete
                # plt.title(f"Testing, delete after using it {len(self.events_attrs)=}")  # Delete
                # plt.legend()  # Delete
                # plt.show(block=False)  # Delete

            else:
                print(
                        f"{evt_pos} {self.time[evt_pos]:12.3f}[s] rejected "
                        f" ({pearson_condition} {pearson_r=:2.3f} {mse_condition} {normal_mse_fit=:3.3f} "
                        f"{tau_condition} {fit_t0=:3.5f} {fit_i_0_condition} {fit_i_0=:2.5f} "
                        f"{direction_condition} {fit_pk0=:3.5f} {n_limit=:3.5f}  {std=:3.5f}  {n_limit*std=:3.5f}) "
                        )
                self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")

        # plt.title(f"Testing, delete after using it {len(self.events_attrs)=}")  # Delete
        # plt.show(block=False)  # Delete

        print(f" Accepted events: {len(self.events_attrs)}, Rejected: {len(evts_attrs_copy) - len(self.events_attrs)}")

    @timing
    def _get_comm_intrv(self):
        """Determination of the common relative time interval for all events."""
        events_time = np.array([evt_values["t_segm"] for evt_values in self.events_attrs.values()], dtype=object)
        events_time = np.concatenate(events_time)
        events_time = np.unique(events_time)
        self.common_time = np.sort(events_time, axis=None)

    @timing
    def get_extended(self):
        if len(self.events_attrs):
            self._get_comm_intrv()
            for evt_pos in self.events_attrs:  # TODO find a way to iterate for all the segm values (dynamically)
                t_segm = self.events_attrs[evt_pos]["t_segm"]
                r_segm = self.events_attrs[evt_pos]["r_segm"]
                p_segm = self.events_attrs[evt_pos]["p_segm"]
                s_segm = self.events_attrs[evt_pos]["s_segm"]
                z_segm = self.events_attrs[evt_pos]["z_segm"]
                d_segm = self.events_attrs[evt_pos]["d_segm"]
                self.events_attrs[evt_pos]["r_segm"] = extender(r_segm, t_segm, self.common_time)
                self.events_attrs[evt_pos]["p_segm"] = extender(p_segm, t_segm, self.common_time)
                self.events_attrs[evt_pos]["s_segm"] = extender(s_segm, t_segm, self.common_time)
                self.events_attrs[evt_pos]["z_segm"] = extender(z_segm, t_segm, self.common_time)
                self.events_attrs[evt_pos]["d_segm"] = extender(d_segm, t_segm, self.common_time)
                self.events_attrs[evt_pos]["t_segm"] = self.common_time
        else:
            print(f"No events detected!!")

    @timing
    def ps_nsfa(self, fit_start=0.001, fit_end=0.01, n_limit=1.0, peak_radius=0.0
                ):
        end_pos = vtp(fit_end, self.t_delta)
        local_exp_fit = exp_fit  # Local assignment for fast lookup on local scope

        avg_resp = np.nanmean(self.get_arr("r_segm"), axis=0)
        avg_time = np.nanmean(self.get_arr("t_segm"), axis=0)  # to use common_time use get_extended first
        p_r_p = vtp(peak_radius, self.t_delta)
        mean_peak = np.min(avg_resp)  # TODO change this to use an average
        mean_peak_pos = np.argmin(avg_resp)

        mean_resp_diff_pow2 = []
        plt.figure(figsize=(3, 2.5))

        for evt_pos in self.events_attrs:
            r_segm = self.events_attrs[evt_pos]["r_segm"]
            t_segm = self.events_attrs[evt_pos]["t_segm"]
            amplitude = self.events_attrs[evt_pos]["amplitude"]
            end_time = self.events_attrs[evt_pos]["end_time"]
            if avg_time[mean_peak_pos + end_pos] <= end_time:
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
        slice_section = slice(mean_peak_pos + start_pos, mean_peak_pos + end_pos)
        time_section = avg_time[slice_section]
        variance_section = var_resp[slice_section]
        response_section = avg_resp[slice_section]
        fit_i_0, fit_pk0, fit_t0, pearson_r = local_exp_fit(
                response_section,
                time_section - avg_time[mean_peak_pos + start_pos],
                # time_section,
                self.direction
                )
        extra_resp_made = np.linspace(
                fit_i_0 - 0.1,
                np.round(
                    fit_pk0 * np.exp((time_section[0] - avg_time[mean_peak_pos + start_pos]) / fit_t0) + fit_i_0
                    ).astype(int),
                # np.round(fit_pk0 * np.exp(time_section[0] / fit_t0) + fit_i_0).astype(int),
                # np.round(np.abs(fit_pk0)).astype(int)
                np.round(np.abs(response_section[0])).astype(int)
                )
        extra_time_made = fit_t0 * np.log((extra_resp_made - fit_i_0) / fit_pk0) + avg_time[mean_peak_pos + start_pos]
        extra_resp = fit_pk0 * np.exp((extra_time_made - avg_time[mean_peak_pos + start_pos]) / fit_t0) + fit_i_0
        plt.plot(time_section, response_section, "bo")
        plt.plot(extra_time_made, extra_resp, "go", markersize=12)
        plt.plot(
                time_section,
                fit_pk0 * np.exp((time_section - avg_time[mean_peak_pos + start_pos]) / fit_t0) + fit_i_0,
                "g"
                )

        bins = [(start <= time_section) & (time_section < end) for start, end in pairwise(np.flip(extra_time_made))]
        binned_time = remove_nan(np.array([np.nanmean(time_section[section]) for section in bins]))
        binned_resp = remove_nan(np.array([np.nanmean(response_section[section]) for section in bins]))
        binned_var = remove_nan(np.array([np.nanmean(variance_section[section]) for section in bins]))
        # plt.plot(binned_time, binned_resp, "ko")
        # plt.plot(binned_time, binned_var, "ko")
        # plt.axhline(0.0, color="k", linestyle='--')
        # plt.axhline(fit_i_0, color="b", linestyle='--')
        # plt.axvline(time_section[0], color="b", linestyle='--')
        # plt.axvline(time_section[-1], color="b", linestyle='--')
        # plt.title(f"Average and Var around the mean {self.time[0]:4.2f} {self.time[-1]:4.2f}")
        # plt.xlim(-0.005, 0.02)  # Set x-axis limits
        # plt.ylim(-80, 40)  # Set y-axis limits
        # plt.show(block=False)

        # plt.figure(figsize=(3, 2.5))
        # plt.axhline(0.0, color="k", linestyle='--')
        # plt.axvline(0.0, color="k", linestyle='--')
        # plt.axvline(mean_peak, color="r", linestyle='--')
        # plt.axvline(self.std * 3.0 * self.direction, color="r", linestyle='--')
        # plt.axvline(self.std * 2.0 * self.direction, color="r", linestyle='--')
        # plt.axvline(self.std * self.direction, color="r", linestyle='--')
        # # TODO make this for positive going too
        binned_resp_clean = binned_resp[binned_resp < self.std * self.direction * n_limit]  # removing background noise
        binned_var_clean = binned_var[binned_resp < self.std * self.direction * n_limit]  # removing background noise
        # plt.plot(
        #         binned_resp_clean,
        #         binned_var_clean,
        #         "ko"
        #         )
        coefficients = parabolic_fit(
                binned_resp_clean,
                binned_var_clean
                )
        intercept = coefficients[0]
        unitary_current = coefficients[1]
        channel_count = -1 / coefficients[2]
        p_0 = np.min(binned_resp_clean) / (unitary_current * channel_count)  # TODO make this for positive going too
        self.ps_nsfa_values = {
                "intercept"      : intercept,
                "i"              : unitary_current,
                "N"              : channel_count,
                "p_0"            : p_0,
                "binned_current" : binned_resp_clean,
                "binned_variance": binned_var_clean,
                "#events"        : n_e
                }
        # plt.plot(
        #         binned_resp,
        #         unitary_current * binned_resp - np.power(binned_resp, 2) / channel_count,
        #         "r:",
        #         label=f"0:{intercept:2.1f},  i:{unitary_current:2.1f},  N:{channel_count:2.1f},  P0:{p_0:1.2f}",
        #         )
        # plt.axhline(intercept, color="r", linestyle='--')
        # plt.xlim(-50, 1)  # Set x-axis limits from 0 to 6
        # plt.ylim(-1, 40)  # Set y-axis limits from 5 to 35
        # plt.title(f"Average and Var around the mean {self.time[0]:4.2f} {self.time[-1]:4.2f}")
        # plt.legend(loc='upper left')
        # plt.show(block=False)

    @timing
    def get_auc(self, already_adjusted=True, min_auc=0.02):  # TODO optimize this method, too slow
        b_amp = 0
        # plt.figure()  # delete me
        evts_attrs_copy = copy.deepcopy(self.events_attrs)
        for evt_pos in evts_attrs_copy.keys():
            if not already_adjusted:
                b_amp = self.events_attrs[evt_pos]["b_amp"]
            if self.events_attrs[evt_pos]["ap_threshold"] is not None:
                start_pos = np.argmax(self.events_attrs[evt_pos]["threshold_segm"])
            else:
                start_pos = np.argmin(self.events_attrs[evt_pos]["z_segm"])
            peak_pos = np.argmax(self.events_attrs[evt_pos]["p_segm"])
            from_peak_segm = self.events_attrs[evt_pos]["r_segm"][peak_pos:] - b_amp
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
                    self.events_attrs[evt_pos]["t_segm"][start_pos:end_pos],
                    self.events_attrs[evt_pos]["r_segm"][start_pos:end_pos] - b_amp
                    )
            if area * self.direction > min_auc * self.direction:
                self.events_attrs[evt_pos]["r_auc"] = area
            else:
                print(
                        f"{evt_pos} {self.time[evt_pos]:12.4f}[s] rejected, low AUC"
                        f" {area=:3.3f}  {min_auc=}"
                        )
                self.events_attrs.pop(evt_pos, f"{evt_pos = } not found")
                continue
        #     plt.axvline(self.events_attrs[evt_pos]["t_segm"][peak_pos])  # delete me
        #     plt.plot(  # delete me
        #             self.events_attrs[evt_pos]["t_segm"][start_pos:end_pos],
        #             self.events_attrs[evt_pos]["r_segm"][start_pos:end_pos] - b_amp
        #             )  # delete me
        # plt.title("Assessing AUC , delete me after...")  # delete me
        # plt.show(block=False)  # delete me
        print(f" Accepted events: {len(self.events_attrs)}, Rejected: {len(evts_attrs_copy) - len(self.events_attrs)}")

    @timing
    def get_threshold(self):
        for evt_pos in self.events_attrs:
            start_pos = np.argmin(self.events_attrs[evt_pos]["z_segm"])
            peak_pos = np.argmax(self.events_attrs[evt_pos]["p_segm"])
            end_pos = peak_pos
            r_section = self.events_attrs[evt_pos]["r_segm"][start_pos:end_pos]
            d1_section = self.events_attrs[evt_pos]["d_segm"][start_pos:end_pos]
            d2_section = differentiate(d1_section, 1.0)
            d3_section = differentiate(d2_section, 1.0)
            threshold_pos = np.argmax(d3_section)
            ap_threshold = r_section[threshold_pos]
            self.events_attrs[evt_pos]["ap_threshold"] = ap_threshold
            # t_section = self.events_attrs[evt_pos]["t_segm"][start_pos:end_pos]
            # ap_threshold_time = t_section[threshold_pos]
            threshold_segm = reset_array(self.events_attrs[evt_pos]["p_segm"], start_pos + threshold_pos)
            self.events_attrs[evt_pos]["threshold_segm"] = threshold_segm

    @timing
    def get_half_width(self):
        print(f"Not implemented!!! {self}")  # TODO implement this method

    @timing
    def show_all_events(
            self, title: str = 'Recording with selected events', show_events: bool = True, adjust=True
            ) -> None:
        plt.figure(figsize=(5, 2.5))
        plt.rcParams.update({'font.size': 8})
        plt.axhline(y=0.0, color="k", linestyle='--')
        plt.plot(self.time, self.resp, "k", linewidth=3)
        if show_events:
            for evt_pos in self.events_attrs:
                t_o_p = self.events_attrs[evt_pos]["t_o_p"]
                t_segm = self.events_attrs[evt_pos]["t_segm"]
                p_segm = self.events_attrs[evt_pos]["p_segm"]
                s_segm = self.events_attrs[evt_pos]["s_segm"]
                z_segm = self.events_attrs[evt_pos]["z_segm"]
                amplitude = self.events_attrs[evt_pos]["amplitude"]
                b_amp = self.events_attrs[evt_pos]["b_amp"]
                threshold_segm = self.events_attrs[evt_pos]["threshold_segm"]
                if not adjust:
                    amplitude -= b_amp
                if threshold_segm is not None:
                    ap_threshold = self.events_attrs[evt_pos]["ap_threshold"]
                    plt.plot(t_segm + t_o_p, b_amp + threshold_segm * (ap_threshold - b_amp), "b")
                plt.plot(t_segm + t_o_p, b_amp + z_segm * amplitude / 4, "g")
                plt.plot(t_segm + t_o_p, b_amp + s_segm * amplitude / 2, "b")
                plt.plot(t_segm + t_o_p, b_amp + p_segm * amplitude, "r", linewidth=2)
            plt.title(f"{title} {len(self.events_attrs) = }")
        plt.show(block=False)
