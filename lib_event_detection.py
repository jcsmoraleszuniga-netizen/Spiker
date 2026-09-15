import copy
import csv
import gc
import io
import os
import random
import warnings
from itertools import pairwise, repeat
from typing import List, Any
from lib_gui import get_record_from_dialog, manage_settings, show_plot
from numpy import ndarray, dtype
from pyabf import ABF
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import find_objects, label
from scipy.signal import fftconvolve, correlate

from lib_utility import (
    conv_vector, down_sample_function_t, exp_decay, exp_fit, find_peaks_and_boundaries, get_names,
    get_real_kernel_sd,
    lin_fit, make_name,
    parabolic_fit,
    remove_nan,
    sort_vectors_by_first, vtp, differentiate,
    find_over_threshold,
    crossing_point, extender, apply_by, mse, timing, prev_change, down_sample_function, smoothing, reset_array,
    remove_shift, file_info, calculate_area, correct_bound, vtp_relative
    )
from copy import deepcopy
from scipy import fft
from matplotlib.ticker import FuncFormatter

precision: int = 8


def test_main(main):
    """
    Primary test entry point.
    """
    # 1. Load the data using the updated helper.
    # We unpack the tuple: (record, path)
    # We pass None as parent since there is no GUI window in this test script.
    original, file_path = get_record_from_dialog(parent=None, location=1)

    # 2. Execute plotting and main logic if a file was selected
    if original:
        show_plot(original, title=f"Testing: {os.path.basename(file_path)}")

        # Define boundaries and execute callback
        bound = original.time[-1]
        main(original, 0, bound, bound)
    else:
        print("File selection cancelled or failed. Exiting test.")


def setup_workspace(ori_inst, dunder_file, const_dict):
    """
    Standardizes directory creation, file naming, and constants loading for all analysis scripts.
    """
    file_name, file_number, file_parent, script_name = get_names(ori_inst, dunder_file)

    # Append the file_name to the parent path and create the folder
    file_parent = os.path.join(file_parent, file_name) + os.sep
    os.makedirs(file_parent, exist_ok=True)

    common_name = [file_name, script_name]
    const_file = str(file_parent + make_name(common_name + ["const"], ".json"))

    # Update the passed dictionary in place
    const_dict.update(manage_settings(const_file, const_dict))

    return file_parent, common_name, const_file


def teardown_workspace(*objects_to_delete):
    """
    Standardizes memory cleanup at the end of scripts.
    """
    for obj in objects_to_delete:
        del obj
    collected = gc.collect()
    print(f"Garbage collector collected {collected} objects.")
    print("Memory should now be freed.")


@timing
def copy_instance(ori_inst: 'EvtPro', direction: int) -> 'EvtPro':
    temp_obj = deepcopy(ori_inst)  # Instantiation of the recordings
    temp_obj.direction = direction
    return temp_obj


@timing
def make_instances(ori_inst: 'EvtPro', direction, start: float, end: float, copies: int) -> List['EvtPro']:
    temp_obj = deepcopy(ori_inst)  # Instantiation of the recordings
    temp_obj.section(start, end)
    if temp_obj.mode == "continuous":
        temp_obj.find_pulses()
    return [copy_instance(temp_obj, direction) for _ in repeat(None, copies)]


def _get_sparse_indices(arr):
    """
    Helper function to extract indices of 1s, their immediate neighbors,
    and the start/end bounds. Reduces mostly-zero arrays by 99%+.
    """
    ones = np.where(arr > 0.1)[0]
    if ones.size == 0:
        return np.array([0, len(arr) - 1])

    # Combine the 1s, the index before, the index after, and the bookends
    idx = np.concatenate((ones - 1, ones, ones + 1, [0, len(arr) - 1]))

    # Ensure no out-of-bounds indices, then return unique (sorted) indices
    return np.unique(np.clip(idx, 0, len(arr) - 1))


@timing
def plot_rec(rec, title="No Title.", values_x=(0.0, 0.0), max_slope=0.0):
    """
    Displays two vertically stacked plots sharing the X-axis.
    Plots all points in the recording without decimation, while retaining
    exact indices and vlines for accurate, fast peak rendering using
    blended transforms to keep markers viewport-centered on Y.

    Color Conventions:
      - Top Plot (Response): Trace in Black ('k'), Peaks & Noise in Red ('r'), Boundaries in Green ('g')
      - Bottom Plot (Derivative):
        * Rise/Original Trace in Black ('k')
        * Softer Decay Trace in Light Gray ('lightgray')
        * Rise phase: Peaks, Zero-crossings, & Threshold in Red ('r')
        * Decay phase: Peaks, Zero-crossings, & Threshold in Blue ('b')
    """
    fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True, figsize=(9, 7),
            gridspec_kw={'height_ratios': [1, 1]}
            )

    # ---------------------------------------------------------
    # DYNAMIC SCALING (NORMALIZED TO AXES FRACTION 0.0 - 0.5)
    # ---------------------------------------------------------
    # Response Peaks
    max_abs_peak = np.max(np.abs(rec.peaks)) if hasattr(rec, 'peaks') and len(rec.peaks) > 0 else 1.0
    if max_abs_peak == 0: max_abs_peak = 1.0
    norm_peaks = (np.abs(rec.peaks) / max_abs_peak) * 0.3

    # Derivative Rise Peaks
    has_rise = hasattr(rec, 'der_peaks_rise') and len(rec.der_peaks_rise) > 0
    max_abs_der_rise = np.max(np.abs(rec.der_peaks_rise)) if has_rise else 1.0
    if max_abs_der_rise == 0: max_abs_der_rise = 1.0
    norm_der_rise = (np.abs(rec.der_peaks_rise) / max_abs_der_rise) * 0.3 if has_rise else np.array([])

    # Derivative Decay Peaks
    has_decay = hasattr(rec, 'der_peaks_decay') and len(rec.der_peaks_decay) > 0
    max_abs_der_decay = np.max(np.abs(rec.der_peaks_decay)) if has_decay else 1.0
    if max_abs_der_decay == 0: max_abs_der_decay = 1.0
    norm_der_decay = (np.abs(rec.der_peaks_decay) / max_abs_der_decay) * 0.3 if has_decay else np.array([])

    # Zero Pass Normalizations
    has_zp_rise = hasattr(rec, 'zero_pass_rise') and len(rec.zero_pass_rise) > 0
    max_zp_rise = np.max(np.abs(rec.zero_pass_rise)) if has_zp_rise else 1.0
    if max_zp_rise == 0: max_zp_rise = 1.0
    norm_zp_rise = (np.abs(rec.zero_pass_rise) / max_zp_rise) * 0.15 if has_zp_rise else np.array([])

    has_zp_decay = hasattr(rec, 'zero_pass_decay') and len(rec.zero_pass_decay) > 0
    max_zp_decay = np.max(np.abs(rec.zero_pass_decay)) if has_zp_decay else 1.0
    if max_zp_decay == 0: max_zp_decay = 1.0
    norm_zp_decay = (np.abs(rec.zero_pass_decay) / max_zp_decay) * 0.15 if has_zp_decay else np.array([])

    # --- Array Indexing Setup ---
    idx_p = np.nonzero(rec.peaks)[0] if hasattr(rec, 'peaks') and len(rec.peaks) > 0 else np.array([], dtype=int)
    idx_dp_rise = np.nonzero(rec.der_peaks_rise)[0] if has_rise else np.array([], dtype=int)
    idx_dp_decay = np.nonzero(rec.der_peaks_decay)[0] if has_decay else np.array([], dtype=int)

    idx_zp_rise = np.nonzero(rec.zero_pass_rise)[0] if has_zp_rise else np.array([], dtype=int)
    idx_zp_decay = np.nonzero(rec.zero_pass_decay)[0] if has_zp_decay else np.array([], dtype=int)

    idx_bounds = []
    if hasattr(rec, 'peak_boundaries') and rec.peak_boundaries:
        for _, start, end in rec.peak_boundaries:
            idx_bounds.extend([start, min(end - 1, len(rec.time) - 1)])
    idx_bounds = np.array(idx_bounds, dtype=int)

    # --- Formatting both axes ---
    for ax in [ax1, ax2]:
        ax.axhline(y=0.0, color="k", linestyle='--', alpha=0.5)
        for x_val in values_x:
            ax.axvline(x=x_val, color="r", linestyle='--', alpha=0.6)

    # ---------------------------------------------------------
    # TOP PLOT (RESPONSE)
    # ---------------------------------------------------------
    ax1.plot(rec.time, rec.resp, "k", linewidth=2.5, label="Response")

    if len(idx_p) > 0:
        ax1.vlines(
                x=rec.time[idx_p],
                ymin=0.5 - norm_peaks[idx_p],
                ymax=0.5 + norm_peaks[idx_p],
                colors="r",
                linewidth=2.0,
                alpha=1.0,
                label="Response Peaks",
                transform=ax1.get_xaxis_transform()
                )

    if len(idx_bounds) > 0:
        ax1.vlines(
                x=rec.time[idx_bounds],
                ymin=0.4,
                ymax=0.6,
                colors="g",
                linewidth=2.0,
                alpha=0.5,
                linestyle='--',
                label="Boundaries",
                transform=ax1.get_xaxis_transform()
                )

    if hasattr(rec, 'peak_noise') and rec.peak_noise is not None:
        ax1.plot(rec.time, rec.peak_noise, "r:", alpha=0.75, label="Peak Noise Threshold")

    ax1.set_ylabel("Response (rec)")
    ax1.set_title(title)

    # ---------------------------------------------------------
    # BOTTOM PLOT (DERIVATIVE)
    # ---------------------------------------------------------
    ax2.axhline(y=max_slope, color="k", linestyle='--', alpha=0.5)
    ax2.plot(rec.time, rec.derivative, "k", linewidth=2.5, label="Derivative (Rise)")

    if hasattr(rec, 'derivative_decay') and rec.derivative_decay is not None:
        ax2.plot(rec.time, rec.derivative_decay, color="lightgray", linewidth=1.8, label="Derivative (Decay)")

    # --- RISE KINETICS (RED) ---
    if len(idx_dp_rise) > 0:
        ax2.vlines(
                x=rec.time[idx_dp_rise],
                ymin=0.5 - norm_der_rise[idx_dp_rise],
                ymax=0.5 + norm_der_rise[idx_dp_rise],
                colors="r",
                linewidth=2.0,
                alpha=1.0,
                label="Rise Slope Peaks",
                transform=ax2.get_xaxis_transform()
                )

    if len(idx_zp_rise) > 0:
        ax2.vlines(
                x=rec.time[idx_zp_rise],
                ymin=0.5 - norm_zp_rise[idx_zp_rise],
                ymax=0.5 + norm_zp_rise[idx_zp_rise],
                colors="r",
                linewidth=2.0,
                alpha=0.5,
                linestyle='--',
                label="Rise Zero Crossings",
                transform=ax2.get_xaxis_transform()
                )

    if hasattr(rec, 'der_peak_noise_rise') and rec.der_peak_noise_rise is not None:
        ax2.plot(rec.time, rec.der_peak_noise_rise, "r:", alpha=0.75, label="Rise Threshold")

    # --- DECAY KINETICS (BLUE) ---
    if len(idx_dp_decay) > 0:
        ax2.vlines(
                x=rec.time[idx_dp_decay],
                ymin=0.5 - norm_der_decay[idx_dp_decay],
                ymax=0.5 + norm_der_decay[idx_dp_decay],
                colors="b",
                linewidth=2.0,
                alpha=1.0,
                label="Decay Slope Peaks",
                transform=ax2.get_xaxis_transform()
                )

    if len(idx_zp_decay) > 0:
        ax2.vlines(
                x=rec.time[idx_zp_decay],
                ymin=0.5 - norm_zp_decay[idx_zp_decay],
                ymax=0.5 + norm_zp_decay[idx_zp_decay],
                colors="b",
                linewidth=2.0,
                alpha=0.5,
                linestyle='--',
                label="Decay Zero Crossings",
                transform=ax2.get_xaxis_transform()
                )

    if hasattr(rec, 'der_peak_noise_decay') and rec.der_peak_noise_decay is not None:
        ax2.plot(rec.time, rec.der_peak_noise_decay, "b:", alpha=0.75, label="Decay Threshold")

    ax2.set_ylabel("Derivative (rec.derivative)")
    ax2.set_xlabel("Time")

    plt.tight_layout()

    # --- Matplotlib Event Loop & RAM Clearing ---
    plt.show(block=False)
    fig.canvas.draw()

    def on_close(event):
        event.canvas.stop_event_loop()

    cid = fig.canvas.mpl_connect('close_event', on_close)
    fig.canvas.start_event_loop(timeout=0)

    fig.canvas.mpl_disconnect(cid)
    fig.clear()
    plt.close(fig)
    plt.close('all')
    del fig, ax1, ax2
    gc.collect()


# @timing
# def plot_rec(rec, title="No Title.", values_x=(0.0, 0.0), max_slope=0.0):
#     """
#     Displays two vertically stacked plots sharing the X-axis.
#     Plots all points in the recording without decimation, while retaining
#     exact indices and vlines for accurate, fast peak rendering using
#     blended transforms to keep markers viewport-centered on Y.
#     """
#     fig, (ax1, ax2) = plt.subplots(
#             2, 1, sharex=True, figsize=(9, 7),
#             gridspec_kw={'height_ratios': [1, 1]}
#             )
#
#     # ---------------------------------------------------------
#     # DYNAMIC SCALING (NORMALIZED TO AXES FRACTION 0.0 - 0.5)
#     # ---------------------------------------------------------
#     # Max peak line extent will occupy up to 30% above/below center (0.5 +/- 0.3)
#     max_abs_peak = np.max(np.abs(rec.peaks)) if len(rec.peaks) > 0 else 1.0
#     if max_abs_peak == 0: max_abs_peak = 1.0
#     norm_peaks = (np.abs(rec.peaks) / max_abs_peak) * 0.3
#
#     max_abs_der_peak = np.max(np.abs(rec.der_peaks_rise)) if len(rec.der_peaks_rise) > 0 else 1.0
#     if max_abs_der_peak == 0: max_abs_der_peak = 1.0
#     norm_der_peaks = (np.abs(rec.der_peaks_rise) / max_abs_der_peak) * 0.3
#
#     # --- Array Indexing Setup ---
#     idx_p = np.nonzero(rec.peaks)[0]
#     idx_dp = np.nonzero(rec.der_peaks_rise)[0]
#     idx_zp = np.nonzero(rec.zero_pass_rise)[0]
#
#     idx_bounds = []
#     if hasattr(rec, 'peak_boundaries') and rec.peak_boundaries:
#         for _, start, end in rec.peak_boundaries:
#             idx_bounds.extend([start, min(end - 1, len(rec.time) - 1)])
#     idx_bounds = np.array(idx_bounds, dtype=int)
#
#     # --- Formatting both axes ---
#     for ax in [ax1, ax2]:
#         ax.axhline(y=0.0, color="k", linestyle='--', alpha=0.5)
#         for x_val in values_x:
#             ax.axvline(x=x_val, color="r", linestyle='--', alpha=0.6)
#
#     # ---------------------------------------------------------
#     # TOP PLOT (REC)
#     # ---------------------------------------------------------
#     ax1.plot(rec.time, rec.resp, "k", linewidth=2.5, label="Response")
#
#     # Draw vertical lines centered at y=0.5 in Axes space (viewport plane)
#     ax1.vlines(
#             x=rec.time[idx_p],
#             ymin=0.5 - norm_peaks[idx_p],
#             ymax=0.5 + norm_peaks[idx_p],
#             colors="r",
#             linewidth=2.0,
#             alpha=1.0,
#             transform=ax1.get_xaxis_transform()
#             )
#
#     # Draw boundary markers centered at y=0.5 with fixed fractional height
#     if len(idx_bounds) > 0:
#         ax1.vlines(
#                 x=rec.time[idx_bounds],
#                 ymin=0.4,
#                 ymax=0.6,
#                 colors="g",
#                 linewidth=2.0,
#                 alpha=0.5,
#                 linestyle='--',
#                 transform=ax1.get_xaxis_transform()
#                 )
#
#     ax1.plot(rec.time, rec.peak_noise, "r:", alpha=0.75)
#     ax1.set_ylabel("Response (rec)")
#     ax1.set_title(title)
#
#     # ---------------------------------------------------------
#     # BOTTOM PLOT (DER)
#     # ---------------------------------------------------------
#     ax2.axhline(y=max_slope, color="k", linestyle='--', alpha=0.5)
#     ax2.plot(rec.time, rec.derivative, "b", linewidth=2.5, label="Derivative")
#
#     # Draw derivative peak lines centered at y=0.5 in Axes space
#     ax2.vlines(
#             x=rec.time[idx_dp],
#             ymin=0.5 - norm_der_peaks[idx_dp],
#             ymax=0.5 + norm_der_peaks[idx_dp],
#             colors="r",
#             linewidth=2.0,
#             alpha=1.0,
#             transform=ax2.get_xaxis_transform()
#             )
#
#     # Zero pass markers centered at y=0.5 with relative scale
#     max_zp = np.max(np.abs(rec.zero_pass_rise)) if len(rec.zero_pass_rise) > 0 else 1.0
#     if max_zp == 0: max_zp = 1.0
#     norm_zp = (np.abs(rec.zero_pass_rise) / max_zp) * 0.15
#
#     ax2.vlines(
#             x=rec.time[idx_zp],
#             ymin=0.5 - norm_zp[idx_zp],
#             ymax=0.5 + norm_zp[idx_zp],
#             colors="g",
#             linewidth=2.0,
#             alpha=0.5,
#             linestyle='--',
#             transform=ax2.get_xaxis_transform()
#             )
#
#     ax2.plot(rec.time, rec.der_peak_noise_rise, "r:", alpha=0.75)
#     ax2.set_ylabel("Derivative (rec.derivative)")
#     ax2.set_xlabel("Time")
#
#     plt.tight_layout()
#
#     # --- Matplotlib Event Loop & RAM Clearing ---
#     plt.show(block=False)
#     fig.canvas.draw()
#
#     def on_close(event):
#         event.canvas.stop_event_loop()
#
#     cid = fig.canvas.mpl_connect('close_event', on_close)
#     fig.canvas.start_event_loop(timeout=0)
#
#     fig.canvas.mpl_disconnect(cid)
#     fig.clear()
#     plt.close(fig)
#     plt.close('all')
#     del fig, ax1, ax2
#     gc.collect()
#

def plot_smooth(temp_rec, temp_smooth_rise, temp_smooth_decay, title="Original versus smoothed recordings."):
    """
    Displays a comparison plot of original data against rise and decay smoothed data.
    Implements strict memory management and local event loops to prevent RAM leaks.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.axhline(y=0.0, color="k", linestyle='--')

    # Signal traces
    ax.plot(temp_rec.time, temp_rec.resp, "k", label="Original", alpha=0.6)
    ax.plot(temp_smooth_rise.time, temp_smooth_rise.resp, "r", label="Smoothed (Rise)", alpha=0.85)
    ax.plot(temp_smooth_decay.time, temp_smooth_decay.resp, "b", label="Smoothed (Decay)", alpha=0.85)

    ax.set_title(title)
    ax.legend(loc="upper right")

    plt.show(block=False)
    fig.canvas.draw()

    def on_close(event):
        event.canvas.stop_event_loop()

    cid = fig.canvas.mpl_connect('close_event', on_close)
    fig.canvas.start_event_loop(timeout=0)

    # Scorch earth memory cleanup
    fig.canvas.mpl_disconnect(cid)
    fig.clear()
    plt.close(fig)
    plt.close('all')

    del fig, ax
    gc.collect()


# def plot_smooth(temp_rec, temp_rec_smooth, title="Original versus smoothed recording."):
#     """
#     Displays a comparison plot of the original and smoothed data.
#     Implements strict memory management and local event loops to prevent RAM leaks.
#     """
#     # Create figure using the object-oriented approach
#     fig, ax = plt.subplots(figsize=(9, 7))
#
#     ax.axhline(y=0.0, color="k", linestyle='--')
#
#     # Plot full resolution signals with slight transparency for readability
#     ax.plot(temp_rec.time, temp_rec.resp, "k", label="Original", alpha=0.75)
#
#     # Calculate the difference array once
#     diff = temp_rec.resp - temp_rec_smooth.resp
#     ax.plot(temp_rec.time, diff, "b", label="original - smoothed", alpha=0.6)
#
#     ax.plot(temp_rec_smooth.time, temp_rec_smooth.resp, "r", label="Smoothed", alpha=0.9)
#
#     ax.set_title(title)
#     ax.legend(loc="upper right")
#
#     # 1. Show the window without triggering the global block
#     plt.show(block=False)
#     fig.canvas.draw()
#
#     # 2. Pass 'event' and use 'event.canvas' to avoid closure circular reference
#     def on_close(event):
#         event.canvas.stop_event_loop()
#
#     cid = fig.canvas.mpl_connect('close_event', on_close)
#
#     # 3. Start Matplotlib's internal loop (Pauses script until window is closed)
#     fig.canvas.start_event_loop(timeout=0)
#
#     # ---------------------------------------------------------
#     # SCORCHED EARTH RAM CLEARING
#     # ---------------------------------------------------------
#     fig.canvas.mpl_disconnect(cid)
#     fig.clear()
#     plt.close(fig)
#     plt.close('all')
#
#     # Explicitly delete the local variables holding the plot objects and arrays
#     del fig, ax, diff
#
#     # Force Python's Garbage Collector to reclaim RAM
#     gc.collect()
#

def plot_adaptive_threshold(
        temp_rec,
        temp_smooth,
        temp_der,
        temp_sder,
        sder_kr_sd,
        diff_derv_g_conv_derv,
        diff_sderv_g_conv_sderv,
        frame_data,
        run_const,
        title="Adaptive Threshold Analysis"
        ):
    """
    Displays 2x2 grid comparing smoothed vs raw responses and derivatives
    alongside adaptive noise threshold trajectories.
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    ax1, ax2 = axs[0, 0], axs[0, 1]
    ax3, ax4 = axs[1, 0], axs[1, 1]

    if title:
        fig.suptitle(title, fontsize=14)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.axhline(y=0.0, color='k', linestyle='dashed', linewidth=1, alpha=0.5)

    # ---------------------------------------------------------
    # BASELINE RAW VS SMOOTH PLOTS
    # ---------------------------------------------------------
    ax1.plot(temp_rec.time, temp_rec.resp, 'r', label="temp_rec.resp", alpha=0.5, linewidth=2)
    ax1.plot(temp_smooth.time, temp_smooth.resp, 'k', label="temp_smooth.resp", linewidth=2)

    ax2.plot(temp_rec.time, temp_rec.derivative, 'r', label="temp_rec.derivative", alpha=0.5, linewidth=2)
    ax2.plot(temp_sder.time, temp_sder.resp, 'k', label="temp_sder.resp", linewidth=2)

    # Transparency decrements across frame steps
    alpha_var_smooth = 1.0
    alpha_var_raw = 0.5
    n_frames = len(frame_data)
    decrement_smooth = alpha_var_smooth / n_frames if n_frames > 0 else 0
    decrement_raw = alpha_var_raw / n_frames if n_frames > 0 else 0

    # ---------------------------------------------------------
    # DERIVATIVE ADAPTIVE THRESHOLDS (STATIC)
    # ---------------------------------------------------------
    ax2.plot(
            temp_sder.time,
            temp_sder.adaptive_sresp,
            'b', alpha=alpha_var_smooth,
            label=f"Frame={run_const['noise_smooth_frame_der']:3.3f} SD={sder_kr_sd:3.5f}"
            )
    ax2.legend(loc="lower right")
    ax2.set_title("Derivative (temp_sder.resp)")

    ax4.plot(temp_der.time, diff_derv_g_conv_derv, 'r', alpha=alpha_var_raw)
    ax4.plot(
            temp_sder.time,
            diff_sderv_g_conv_sderv,
            'k', alpha=alpha_var_smooth,
            label=f"Frame={run_const['noise_smooth_frame_der']:3.3f} SD={sder_kr_sd:3.5f}"
            )
    ax4.plot(temp_sder.time, temp_sder.direction * temp_sder.interpolated_sd, 'g', alpha=alpha_var_smooth)
    ax4.legend(loc="lower right")
    ax4.set_title("temp_sder.resp - temp_sder.adaptive_sresp")
    ax4.set_xlabel("Time")

    # ---------------------------------------------------------
    # RESPONSE FRAME INCREMENTS
    # ---------------------------------------------------------
    for data in frame_data:
        frame = data["frame"]
        smooth_kr_sd = data["smooth_kr_sd"]
        adaptive_sresp = data["adaptive_sresp"]
        interpolated_sd = data["interpolated_sd"]
        diff_resp = data["diff_resp"]
        diff_sresp = data["diff_sresp"]

        ax1.plot(
                temp_smooth.time,
                adaptive_sresp,
                'b', alpha=alpha_var_smooth,
                label=f"Frame={frame:3.3f} SD={smooth_kr_sd:3.5f}"
                )

        ax3.plot(temp_rec.time, diff_resp, 'r', alpha=alpha_var_raw)
        ax3.plot(
                temp_smooth.time,
                diff_sresp,
                'k', alpha=alpha_var_smooth,
                label=f"Frame={frame:3.3f} SD={smooth_kr_sd:3.5f}"
                )
        ax3.plot(temp_smooth.time, temp_smooth.direction * interpolated_sd, 'g', alpha=alpha_var_smooth)

        alpha_var_smooth -= decrement_smooth
        alpha_var_raw -= decrement_raw

    ax1.legend(loc="lower right")
    ax1.set_title("Response (temp_smooth.resp)")

    ax3.legend(loc="lower right")
    ax3.set_title("temp_smooth.resp - temp_smooth.adaptive_sresp")
    ax3.set_xlabel("Time")

    plt.tight_layout()

    # ---------------------------------------------------------
    # EVENT LOOP & RAM CLEARING
    # ---------------------------------------------------------
    plt.show(block=False)
    fig.canvas.draw()

    def on_close(event):
        event.canvas.stop_event_loop()

    cid = fig.canvas.mpl_connect('close_event', on_close)
    fig.canvas.start_event_loop(timeout=0)

    fig.canvas.mpl_disconnect(cid)
    fig.clear()
    plt.close(fig)
    plt.close('all')
    del fig, axs, ax1, ax2, ax3, ax4
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
        self.t_delta: np.floating = np.round(self.sweepX[1] - self.sweepX[0], decimals=precision)
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
                self.record[0][1] - self.record[0][0], decimals=precision
                )  # time increment
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
        self.sweep_index = 0
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

    @timing
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        cls.instance_number += 1
        result._instance_number = cls.instance_number
        return result

    @timing
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for key, value in self.__dict__.items():
            # Skip or shallow-assign file streams and BufferedReader instances
            if isinstance(value, io.IOBase) or type(value).__name__ in ('BufferedReader', 'BufferedWriter'):
                setattr(result, key, None)
            else:
                try:
                    setattr(result, key, copy.deepcopy(value, memo))
                except TypeError:
                    # Fallback for any other non-copyable internal objects
                    setattr(result, key, value)

        cls.instance_number += 1
        result._instance_number = cls.instance_number
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
        # Standard Python list boundaries: from -length to length-1
        if -len(self.sweeps) <= index < len(self.sweeps):
            self.sweep_index = index
            self.resp = self.sweeps[self.sweep_index]
        else:
            # Use the passed 'index' in the error message, not the old 'self.sweep_index'
            raise IndexError(
                    f"The index ({index}) is out of bounds. Choose between {-len(self.sweeps)} and {len(self.sweeps) - 1}."
                    )

    @timing
    def transfer(self, other, sweep_index: int = None):
        """
        Transfers data from another instance or array.
        If sweep_index is provided, updates that specific sweep in self.sweeps.
        """
        # 1. Extract the data to be transferred
        if isinstance(other, loadRecord):
            new_time = other.time
            new_resp = other.resp
            new_cdac = other.cdac
        elif isinstance(other, (list, tuple, np.ndarray)):
            new_time = other[0]
            new_resp = other[1]
            new_cdac = other[2]
        else:
            print(f"Type not supported.")
            return

        # 2. Update active buffers
        self.time = new_time
        self.resp = new_resp
        self.cdac = new_cdac

        # 3. CRITICAL: Update the specific sweep in the collection
        if sweep_index is not None:
            if 0 <= sweep_index < len(self.sweeps):
                # We use copy() to ensure the reference is independent
                self.sweeps[sweep_index] = np.copy(new_resp)
                print(f"Sweep {sweep_index} updated in self.sweeps.")
            else:
                print(f"Sweep index {sweep_index} out of range.")

    @timing
    def section(self, start, end):
        print(f"{self.time[0] = }  {self.time[-1] = }")

        # 1. Find start position
        try:
            start_pos = np.where(start == self.time)[0][0]
        except IndexError:
            # FIXED: Was previously checking against 'end' instead of 'start'
            closest_pos = vtp_relative(start, self.time)
            # closest_pos = np.argmin(np.abs(self.time - start))
            print(f"Using {len(self.time) = } {self.time[-1] = }  {closest_pos = }  {self.time[closest_pos]}")
            start_pos = closest_pos

        # 2. Find end position
        try:
            end_pos = np.where(end == self.time)[0][0]
        except IndexError:
            closest_pos = vtp_relative(end, self.time)
            # closest_pos = np.argmin(np.abs(self.time - end))
            print(f"Using {len(self.time) = } {self.time[-1] = }  {closest_pos = }  {self.time[closest_pos]}")
            end_pos = closest_pos

        # ---------------------------------------------------------
        # THE FIX: Prevent dropping the final point of the recording
        # ---------------------------------------------------------
        # If end_pos is the very last index of the entire array,
        # add 1 so Python's exclusive slicing captures it.
        if end_pos == len(self.time) - 1:
            end_pos += 1

        print(f"{start_pos = }  {end_pos = }")

        # 3. Slice the data
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
        plt.plot(plot_x, plot_y, linewidth=0.5)
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
        self.interpolated_sd = np.array([])  # Initialization
        self.adaptive_sresp = np.array([])  # Initialization
        self.peak_noise = np.array([])  # Initialization
        self.der_peak_noise_rise = np.array([])  # Initialization
        self.der_peak_noise_decay = np.array([])
        self.std = 0.0  # Initialization
        self.derivative = np.array([])  # Initialization
        self.derivative_decay = np.array([])  # Initialization
        self.der_peaks_rise = np.array([])  # Initialization
        self.der_peaks_decay = np.array([])
        self.o_thresh = np.array([])  # Initialization
        self.peaks = np.array([])  # Initialization
        self.peak_boundaries = np.array([])  # Initialization
        self.zero_pass_rise = np.array([])  # Initialization
        self.zero_pass_decay = np.array([])
        self.raw_zero_pass_rise = np.array([])
        self.raw_zero_pass_decay = np.array([])
        self.inp_res = np.array([])  # Initialization
        self.mem_cap = np.array([])  # Initialization
        self.acc_res = np.array([])  # Initialization
        self.area: dict[str, float | ndarray[Any, dtype]] = {"Area": 0.0, "Amplitude": 0.0, "rTTP": 0.0}
        self.voltage_ires = None  # Store to instance for external plotting
        # Dictionary to store extracted IV attributes, keyed by pulse position
        self.iv_attrs = {}
        # Default dictionary template for IV parameters
        self.default_iv = {
                "t_o_p"  : 0.0,
                "acc_res": 0.0,
                "iv_res" : np.array([]),
                "iv_time": np.array([])
                }

    @timing
    def get_smooth(self, smooth_width=0.001, sharpness=4):
        n_p = vtp(smooth_width, self.t_delta)
        self.resp, kernel_sd = smoothing(self.resp, n_p, sharpness, 'g', 1 / self.t_delta)
        return kernel_sd

    @timing
    def get_section_area(self, baseline_start=371, baseline_end=376, response_end=441, linear_fit=False):
        """Calculates the area, peak and rTTP"""
        b_s_p = np.where(self.time == baseline_start)[0][0]
        b_e_p = np.where(self.time == baseline_end)[0][0]
        r_e_p = np.where(self.time == response_end)[0][0]
        points = vtp(0.002, self.t_delta)

        shift = np.nanmean(self.resp[b_s_p:b_e_p])

        resp, kernel_sd = smoothing(
                copy.deepcopy(self.resp[b_e_p:r_e_p]) - shift, points, 2, 'g', 1 / self.t_delta
                )
        r_time = copy.deepcopy(self.time[b_e_p:r_e_p])
        base, kernel_sd = smoothing(
                copy.deepcopy(self.resp[b_s_p:b_e_p]) - shift, points, 2, 'g', 1 / self.t_delta
                )
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
        # Check if either structure already contains data; if so, exit early
        if len(self.pulses_peaks) > 0 and len(self.pul_attrs) > 0:
            return
        der_test_resp = differentiate(self.cdac, self.t_delta)
        der_test_resp = prev_change(der_test_resp)
        threshold = der_test_resp * 0.0 + self.direction * 9 * np.std(der_test_resp)
        o_thresh = find_over_threshold(der_test_resp, threshold, direction)
        self.pulses_peaks, _ = find_peaks_and_boundaries(o_thresh, der_test_resp, direction)
        self.pul_attrs = {evt_pos: {"t_o_p": self.time[evt_pos]} for evt_pos, val in enumerate(self.pulses_peaks) if
                          val}

    @timing
    def del_pulses(self, del_length=0.75, target_val=2000.0):
        resp_copy = copy.deepcopy(self.resp)
        time_copy = copy.deepcopy(self.time)

        for evt_data in self.pul_attrs.values():
            t_start = evt_data["t_o_p"]

            # Resolve start and end indices dynamically in the current self.time array
            start_idx = vtp_relative(t_start, self.time)
            end_idx = vtp_relative(t_start + del_length, self.time)

            if end_idx > start_idx:
                resp_copy[start_idx:end_idx] = target_val

        index = np.where(resp_copy == target_val)[0]
        self.time = np.delete(time_copy, index)
        self.resp = np.delete(resp_copy, index)

    # def del_pulses(self, del_length=0.75, target_val=2000.0):
    #     peaks = self.pulses_peaks
    #     resp_copy = copy.deepcopy(self.resp)
    #     time_copy = copy.deepcopy(self.time)
    #     del_range = vtp(del_length, self.t_delta)
    #     for pos, val in enumerate(peaks):
    #         if val:
    #             resp_copy[pos:pos + del_range] = target_val
    #     index = np.argwhere(resp_copy == target_val)
    #     self.time = np.delete(time_copy, index)
    #     self.resp = np.delete(resp_copy, index)

    @timing
    def del_artifacts(self, artifacts_at, artifact_width, target_val=2000.0):
        resp_copy = copy.deepcopy(self.resp)
        time_copy = copy.deepcopy(self.time)

        for val in artifacts_at:
            start_idx = vtp_relative(val, self.time)
            end_idx = vtp_relative(val + artifact_width, self.time)

            if end_idx > start_idx:
                resp_copy[start_idx:end_idx] = target_val

        index = np.where(resp_copy == target_val)[0]
        self.time = np.delete(time_copy, index)
        self.resp = np.delete(resp_copy, index)

    # def del_artifacts(self, artifacts_at, artifact_width, target_val=2000.0):
    #     resp_copy = copy.deepcopy(self.resp)
    #     time_copy = copy.deepcopy(self.time)
    #     del_range = vtp(artifact_width, self.t_delta)
    #     for val in artifacts_at:
    #         pos = vtp_relative(val, self.time)
    #         resp_copy[pos:pos + del_range] = target_val
    #     index = np.argwhere(resp_copy == target_val)
    #     self.time = np.delete(time_copy, index)
    #     self.resp = np.delete(resp_copy, index)

    @timing
    def get_pk_noise(self, time_frame=0.2, n_deviations=3, resp_increment=0.5, std_increment=10, sharpness=2):
        n_p = vtp(time_frame, self.t_delta)
        self.adaptive_sresp, kernel_sd = smoothing(self.resp, n_p, sharpness, 'g', 1 / self.t_delta)
        # 1. Use a separate variable name for the dictionary
        resp_inc_dict = {"start": self.time[0], "end": self.time[-1], "increment": resp_increment}
        std_resp = apply_by(np.std, np.array([self.time, self.resp]), resp_inc_dict, True)
        # Safety Check: Did resp_increment fail?
        if std_resp.size == 0 or len(std_resp[0]) == 0:
            print("Warning: std_resp is empty. Falling back to 0 noise.")
            self.std = 0.0
            self.peak_noise = self.adaptive_sresp
            return kernel_sd

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

        std_min = apply_by(np.min, std_resp, std_inc_dict, True)

        # 4. The Safety Net: If apply_by STILL returns empty, don't crash.
        if std_min.size == 0 or len(std_min[0]) == 0:
            print("Warning: apply_by returned an empty array for std_min. Using global minimum instead.")
            self.std = np.nanmean(std_resp[1])  # Fallback to the mean of the whole section
            global_min = np.nanmin(std_resp[1])
            self.peak_noise = self.adaptive_sresp + (global_min * n_deviations * self.direction)
        else:
            # Standard successful execution
            self.std = np.nanmean(std_min[1])
            self.interpolated_sd = np.interp(
                    self.time, std_min[0],
                    std_min[1] * n_deviations
                    )
            self.peak_noise = self.adaptive_sresp + self.direction * self.interpolated_sd
        return kernel_sd

    @timing
    def get_derv(self):
        self.derivative = differentiate(self.resp, self.t_delta)

    @timing
    def get_o_thresh(self):
        self.o_thresh = find_over_threshold(self.resp, self.peak_noise, self.direction)

    @timing
    def get_peaks(self, shift_time=0.001):
        self.get_o_thresh()
        self.peaks, self.peak_boundaries = find_peaks_and_boundaries(self.o_thresh, self.resp, self.direction)
        # Vectorized artifact removal
        if np.any(self.pulses_peaks):
            shift = vtp(shift_time, self.t_delta)
            has_artifact = (self.pulses_peaks != 0).astype(int)
            kernel_size = 2 * shift
            if kernel_size < 1: kernel_size = 1
            kernel = np.ones(kernel_size, dtype=int)
            artifact_mask = np.convolve(has_artifact, kernel, mode='same') > 0
            # Zero out peaks that fall inside the danger zone
            self.peaks[artifact_mask] = 0.0
            # KEEP BOUNDARIES IN SYNC:
            # Only keep boundaries if their peak index (b[0]) does NOT fall in the artifact_mask
            self.peak_boundaries = [b for b in self.peak_boundaries if not artifact_mask[b[0]]]
        else:
            print("No pulse peaks detected.")

    @timing
    def get_z_pass(self, delete_peaks: bool = True) -> None:
        """
        Globally locates all zero crossings in self.resp and isolates the ones
        that directly flank valid peaks. Matches the farthest points from the peak,
        or locks exactly onto the point if the value is precisely 0.0.
        """
        self.zero_pass_rise = np.zeros(len(self.peaks))

        if not np.any(self.peaks) or not np.any(self.resp):
            return

        peak_idx = np.where(self.peaks > 0)[0]

        # 1. Use np.sign to correctly handle exact 0.0 values
        # np.sign returns -1, 0, or 1. A non-zero diff catches EVERY boundary.
        sign_array = np.sign(self.resp)
        diffs = np.diff(sign_array)

        # change_lefts: The exact indices before a sign change / zero hit
        change_lefts = np.where(diffs != 0)[0]

        if change_lefts.size == 0:
            if delete_peaks:
                self.peaks = np.zeros_like(self.peaks)
            return

        # change_rights: The exact indices after a sign change / zero hit
        change_rights = change_lefts + 1

        # 2. Map peaks to the correct boundaries
        # Left boundary: The LAST change_lefts index strictly before the peak
        idx_left = np.searchsorted(change_lefts, peak_idx, side='left') - 1

        # Right boundary: The FIRST change_rights index strictly after the peak
        idx_right = np.searchsorted(change_rights, peak_idx, side='right')

        # 3. Safely mask peaks that fall outside the global boundaries
        valid_mask = (idx_left >= 0) & (idx_right < len(change_rights))

        if delete_peaks:
            invalid_peaks = peak_idx[~valid_mask]
            self.peaks[invalid_peaks] = 0

        valid_idx_left = idx_left[valid_mask]
        valid_idx_right = idx_right[valid_mask]

        if valid_idx_left.size == 0:
            return

        # 4. Extract the exact bounding indices
        left_crossings = change_lefts[valid_idx_left]
        right_crossings = change_rights[valid_idx_right]

        # 5. Deduplicate and identify shared crossings
        unique_left = np.unique(left_crossings)
        unique_right = np.unique(right_crossings)

        # self.zero_pass[unique_left] = -1
        # self.zero_pass[unique_right] = 1

        shared = np.intersect1d(unique_left, unique_right)
        only_left = np.setdiff1d(unique_left, shared)
        only_right = np.setdiff1d(unique_right, shared)

        self.zero_pass_rise[only_left] = -1
        self.zero_pass_rise[only_right] = 1
        self.zero_pass_rise[shared] = 2  # 2 acts as a simultaneous 'End & Start'

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
                # self.pul_attrs[pos]["t_o_p"] = self.time[pos]
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
                # self.pul_attrs[pos]["t_o_p"] = self.time[pos]
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
        """
        Extracts Current-Voltage (I-V) characteristics and access resistance from control pulses.

        Acts as the primary analytical routine for voltage-clamp pulse protocols. The method
        first verifies the presence of control pulses, exiting early if none are detected to
        prevent empty array operations. For each valid pulse, it calculates the access resistance
        using the amplitude of the capacitive transient, and isolates the steady-state current
        response to derive the input/membrane resistance.

        Extracted metrics, including calculated resistances and raw waveform slices, are
        stored directly within the `self.iv_attrs` dictionary.

        Args:
            beg_ar (float): Start time (in seconds) relative to the pulse onset to calculate the access resistance baseline.
            end_ar (float): Time window (in seconds) to capture the peak capacitive transient for access resistance.
            beg_ir (float): Start time (in seconds) to measure the steady-state current for input resistance.
            end_ir (float): End time (in seconds) for the steady-state current measurement.
            holding (float): The holding potential of the cell (in mV), used as an offset for the I-V graph.
        """
        if not np.any(self.pulses_peaks):
            print("No control pulses detected. Skipping I-V calculation.")
            return

        np_average = np.average
        np_min = np.min
        np_abs = np.abs
        bas_acc = vtp(beg_ar - 0.001, self.t_delta)
        beg_acc = vtp(beg_ar, self.t_delta)
        end_acc = vtp(end_ar, self.t_delta)
        beg_res = vtp(beg_ir, self.t_delta)
        end_res = vtp(end_ir, self.t_delta)

        voltage_pulse = None

        # Reset attributes for the current analysis section
        self.iv_attrs = {}
        self.voltage_ires = np.array([])

        for pos, value in enumerate(self.pulses_peaks):
            if value:
                # Initialize default IV template for this position
                self.iv_attrs[pos] = self.default_iv.copy()
                self.iv_attrs[pos]["t_o_p"] = self.time[pos]

                # Access resistance calculations
                pulse_slice = slice(pos + end_acc, pos + beg_acc)
                pulse_base_slice = slice(pos - beg_acc, pos - end_acc)
                if voltage_pulse is None:
                    voltage_pulse = np_average(self.cdac[pulse_slice]) - np_average(self.cdac[pulse_base_slice])

                current_acc = self.resp[pos - beg_acc:pos + end_acc]
                base_current = np_average(current_acc[:bas_acc])
                c_diff = base_current - np_min(current_acc)

                if c_diff == 0:
                    c_diff = 1e-12  # Prevent division by zero

                # Input resistance calculations and fitting
                ir_slice = slice(pos + beg_res, pos + end_res)

                # Directly populate the instance variable initialized in __init__
                if len(self.voltage_ires) == 0:
                    self.voltage_ires = self.cdac[ir_slice]

                current_ires = self.resp[ir_slice]
                current_time = self.time[ir_slice]

                # Storing into the iv_attrs dictionary
                self.iv_attrs[pos]["acc_res"] = np_abs(voltage_pulse / c_diff) * 1000  # In Mega Ohms
                self.iv_attrs[pos]["iv_res"] = current_ires
                self.iv_attrs[pos]["iv_time"] = current_time

    @staticmethod
    def _filter_peaks_by_zero_pass(der_peaks: np.ndarray, zero_pass: np.ndarray) -> np.ndarray:
        """
        Core state machine: extracts strict [start, end] zero-pass intervals
        and retains the maximum peak within each interval window.
        """
        if not np.any(der_peaks) or not np.any(zero_pass):
            return np.zeros_like(der_peaks)

        # 1. Safely extract strict [start, end] windows from zero_pass
        zc_indices = np.where(zero_pass != 0)[0]
        intervals = []
        in_event = False
        start_idx = 0

        for idx in zc_indices:
            val = zero_pass[idx]
            if val == -1 and not in_event:
                start_idx = idx
                in_event = True
            elif val == 1 and in_event:
                intervals.append((start_idx, idx))
                in_event = False
            elif val == 2:
                if in_event:
                    intervals.append((start_idx, idx))  # Close previous event
                start_idx = idx  # Immediately open new event
                in_event = True

        # Close the final event if the recording ended before a +1 marker
        if in_event:
            intervals.append((start_idx, len(zero_pass) - 1))

        # 2. Isolate the maximum slope peak in each window
        new_der_peaks = np.zeros_like(der_peaks)

        for start, end in intervals:
            window = der_peaks[start:end + 1]
            local_peak_indices = np.where(window > 0)[0]

            if local_peak_indices.size > 0:
                best_local_idx = local_peak_indices[np.argmax(window[local_peak_indices])]
                global_idx = start + best_local_idx
                new_der_peaks[global_idx] = der_peaks[global_idx]

        return new_der_peaks

    @timing
    def align_slopes_to_zero_crossings(self, phase: str = "both") -> None:
        """
        Refines slope peaks (rise, decay, or both) by retaining the maximum peak
        value between each consecutive zero crossing pair (-1 and 1).

        Parameters
        ----------
        phase : str, optional
            Target phase to align: 'rise', 'decay', or 'both' (default is 'both').
        """
        phase = phase.lower()

        if phase in ("rise", "both"):
            if hasattr(self, "der_peaks_rise") and hasattr(self, "zero_pass_rise"):
                self.der_peaks_rise = self._filter_peaks_by_zero_pass(
                        self.der_peaks_rise, self.zero_pass_rise
                        )

        if phase in ("decay", "both"):
            if hasattr(self, "der_peaks_decay") and hasattr(self, "zero_pass_decay"):
                self.der_peaks_decay = self._filter_peaks_by_zero_pass(
                        self.der_peaks_decay, self.zero_pass_decay
                        )

    # @timing
    # def align_slopes_to_zero_crossings(self) -> None:
    #     """
    #     Refines self.der_peaks by retaining the slope peak with the maximum value
    #     between each consecutive zero crossing pair (-1 and 1).
    #     Uses a strict interval state machine to handle overlapping/duplicate zero-pass markers.
    #     """
    #     if not np.any(self.der_peaks_rise) or not np.any(self.zero_pass_rise):
    #         self.der_peaks_rise = np.zeros_like(self.der_peaks_rise)
    #         return
    #
    #     # 1. Safely extract strict [start, end] windows from zero_pass
    #     zc_indices = np.where(self.zero_pass_rise != 0)[0]
    #     intervals = []
    #     in_event = False
    #     start_idx = 0
    #
    #     # This parses markers left-to-right, absorbing duplicate -1s or 1s
    #     for idx in zc_indices:
    #         val = self.zero_pass_rise[idx]
    #         if val == -1 and not in_event:
    #             start_idx = idx
    #             in_event = True
    #         elif val == 1 and in_event:
    #             intervals.append((start_idx, idx))
    #             in_event = False
    #         elif val == 2:
    #             if in_event:
    #                 intervals.append((start_idx, idx))  # Close previous event
    #             start_idx = idx  # Immediately open new event
    #             in_event = True
    #
    #     # Close the final event if the recording ended before a +1 marker
    #     if in_event:
    #         intervals.append((start_idx, len(self.zero_pass_rise) - 1))
    #
    #     # 2. Iterate through each strict window and isolate the maximum slope peak
    #     new_der_peaks = np.zeros_like(self.der_peaks_rise)
    #
    #     for start, end in intervals:
    #         # Extract the actual values of der_peaks in this closed interval [start, end]
    #         window = self.der_peaks_rise[start:end + 1]
    #
    #         # Find local indices where a derivative peak actually exists
    #         local_peak_indices = np.where(window > 0)[0]
    #
    #         if local_peak_indices.size > 0:
    #             # Find the index of the max value among these candidates
    #             best_local_idx = local_peak_indices[np.argmax(window[local_peak_indices])]
    #
    #             # Map back to global index and store the exact value
    #             global_idx = start + best_local_idx
    #             new_der_peaks[global_idx] = self.der_peaks_rise[global_idx]
    #
    #     # 3. Normalize array to strictly 1.0 and 0.0
    #     # self.der_peaks = np.where(new_der_peaks > 0.0, 1.0, 0.0)
    #     self.der_peaks_rise = new_der_peaks

    # def _extract_slope_intervals(
    #         self, sig_len: int, zero_pass: np.ndarray, der_peaks: np.ndarray
    #         ) -> list[tuple[int, int]]:
    #     """Parses zero-crossings into discrete slope windows [start, end]."""
    #     slope_intervals = []
    #     if np.any(zero_pass):
    #         zc_indices = np.where(zero_pass != 0)[0]
    #         in_event = False
    #         start_idx = 0
    #
    #         for idx in zc_indices:
    #             val = zero_pass[idx]
    #             if val == -1 and not in_event:
    #                 start_idx = idx
    #                 in_event = True
    #             elif val == 1 and in_event:
    #                 slope_intervals.append((start_idx, idx))
    #                 in_event = False
    #             elif val == 2:
    #                 if in_event:
    #                     slope_intervals.append((start_idx, idx))
    #                 start_idx = idx
    #                 in_event = True
    #
    #         if in_event:
    #             slope_intervals.append((start_idx, sig_len - 1))
    #     else:
    #         ms_idx = np.where(der_peaks > 0)[0]
    #         for slope_idx in ms_idx:
    #             slope_intervals.append((slope_idx, slope_idx))
    #
    #     return slope_intervals
    #
    # def _clear_all_attributes(self) -> None:
    #     """Wipes all peak, derivative, boundary, and zero-pass attributes."""
    #     self.peaks = np.zeros_like(self.peaks)
    #     self.der_peaks_rise = np.zeros_like(self.der_peaks_rise)
    #     self.zero_pass_rise = np.zeros_like(self.zero_pass_rise)
    #     self.der_peaks_decay = np.zeros_like(self.der_peaks_decay)
    #     self.zero_pass_decay = np.zeros_like(self.zero_pass_decay)
    #     self.peak_boundaries = []
    #
    # @timing
    # def align_peaks_to_slopes(self) -> None:
    #     """
    #     Symmetrically aligns response peaks with rise and decay slopes.
    #     Retains an event if and only if a response peak overlaps with BOTH
    #     a valid rise slope and a valid decay slope.
    #     """
    #     if (
    #             not np.any(self.peaks)
    #             or not np.any(self.der_peaks_rise)
    #             or not np.any(self.der_peaks_decay)
    #             or not self.peak_boundaries
    #     ):
    #         self._clear_all_attributes()
    #         return
    #
    #     sig_len = len(self.peaks)
    #
    #     # 1. Precompute Rise Slopes ONCE
    #     rise_intervals = self._extract_slope_intervals(
    #             sig_len, self.zero_pass_rise, self.der_peaks_rise
    #             )
    #     precomputed_rise = []
    #     for r_start, r_end in rise_intervals:
    #         r_indices = np.where(self.der_peaks_rise[r_start: r_end + 1] > 0)[0] + r_start
    #         if r_indices.size > 0:
    #             r_peak_idx = r_indices[np.argmax(self.der_peaks_rise[r_indices])]
    #             precomputed_rise.append((r_start, r_end, r_peak_idx, self.der_peaks_rise[r_peak_idx]))
    #
    #     # 2. Precompute Decay Slopes ONCE
    #     decay_intervals = self._extract_slope_intervals(
    #             sig_len, self.zero_pass_decay, self.der_peaks_decay
    #             )
    #     precomputed_decay = []
    #     for d_start, d_end in decay_intervals:
    #         d_indices = np.where(self.der_peaks_decay[d_start: d_end + 1] > 0)[0] + d_start
    #         if d_indices.size > 0:
    #             d_peak_idx = d_indices[np.argmax(self.der_peaks_decay[d_indices])]
    #             precomputed_decay.append((d_start, d_end, d_peak_idx, self.der_peaks_decay[d_peak_idx]))
    #
    #     if not precomputed_rise or not precomputed_decay:
    #         self._clear_all_attributes()
    #         return
    #
    #     # 3. Commutative Intersection Evaluation
    #     valid_events = []
    #
    #     for p_idx, p_start, p_end in self.peak_boundaries:
    #         # Evaluate Rise Overlap
    #         best_r_val, best_r_idx, best_r_interval = -1.0, -1, None
    #         for r_start, r_end, r_peak_idx, r_val in precomputed_rise:
    #             if max(p_start, r_peak_idx) <= min(p_idx, r_end):
    #                 if r_val > best_r_val:
    #                     best_r_val, best_r_idx, best_r_interval = r_val, r_peak_idx, (r_start, r_end)
    #
    #         if best_r_idx == -1:
    #             continue
    #
    #         # Evaluate Decay Overlap
    #         best_d_val, best_d_idx, best_d_interval = -1.0, -1, None
    #         for d_start, d_end, d_peak_idx, d_val in precomputed_decay:
    #             if max(p_idx, d_start) <= min(p_end, d_peak_idx):
    #                 if d_val > best_d_val:
    #                     best_d_val, best_d_idx, best_d_interval = d_val, d_peak_idx, (d_start, d_end)
    #
    #         if best_d_idx == -1:
    #             continue
    #
    #         valid_events.append(
    #                 (p_idx, p_start, p_end, best_r_idx, best_r_interval, best_d_idx, best_d_interval)
    #                 )
    #
    #     if not valid_events:
    #         self._clear_all_attributes()
    #         return
    #
    #     # 4. Spatial Island Grouping
    #     combined_mask = np.zeros(sig_len, dtype=bool)
    #     for _, p_start, p_end, _, _, _, _ in valid_events:
    #         combined_mask[p_start: p_end + 1] = True
    #
    #     padded = np.zeros(sig_len + 2, dtype=bool)
    #     padded[1:-1] = combined_mask
    #     changes = np.diff(padded.astype(int))
    #
    #     island_starts = np.where(changes == 1)[0]
    #     island_ends = np.where(changes == -1)[0]
    #
    #     surviving_peaks, surviving_rise_slopes, surviving_decay_slopes = [], [], []
    #     surviving_rise_intervals, surviving_decay_intervals = [], []
    #
    #     for i_start, i_end in zip(island_starts, island_ends):
    #         island_candidates = [ev for ev in valid_events if i_start <= ev[0] < i_end]
    #         if island_candidates:
    #             winner = max(island_candidates, key=lambda ev: self.peaks[ev[0]])
    #             p_idx, _, _, r_idx, r_int, d_idx, d_int = winner
    #
    #             surviving_peaks.append(p_idx)
    #             surviving_rise_slopes.append(r_idx)
    #             surviving_decay_slopes.append(d_idx)
    #             surviving_rise_intervals.append(r_int)
    #             surviving_decay_intervals.append(d_int)
    #
    #     # 5. Direct Attribute Updates
    #     new_peaks = np.zeros_like(self.peaks)
    #     new_peaks[surviving_peaks] = self.peaks[surviving_peaks]
    #     self.peaks = new_peaks
    #
    #     new_der_rise = np.zeros_like(self.der_peaks_rise)
    #     new_der_rise[surviving_rise_slopes] = self.der_peaks_rise[surviving_rise_slopes]
    #     self.der_peaks_rise = new_der_rise
    #
    #     new_der_decay = np.zeros_like(self.der_peaks_decay)
    #     new_der_decay[surviving_decay_slopes] = self.der_peaks_decay[surviving_decay_slopes]
    #     self.der_peaks_decay = new_der_decay
    #
    #     surviving_peak_set = set(surviving_peaks)
    #     self.peak_boundaries = [b for b in self.peak_boundaries if b[0] in surviving_peak_set]
    #
    #     new_zp_rise = np.zeros_like(self.zero_pass_rise)
    #     for r_start, r_end in surviving_rise_intervals:
    #         new_zp_rise[r_start] = self.zero_pass_rise[r_start]
    #         new_zp_rise[r_end] = self.zero_pass_rise[r_end]
    #     self.zero_pass_rise = new_zp_rise
    #
    #     new_zp_decay = np.zeros_like(self.zero_pass_decay)
    #     for d_start, d_end in surviving_decay_intervals:
    #         new_zp_decay[d_start] = self.zero_pass_decay[d_start]
    #         new_zp_decay[d_end] = self.zero_pass_decay[d_end]
    #     self.zero_pass_decay = new_zp_decay
    @timing
    def align_peaks_to_slopes_decay(self) -> None:
        """
        Aligns peaks and decay slopes by identifying combined overlapping intervals.
        Selects the maximum peak per interval, the maximum decay slope after it,
        and filters boundaries and zero crossings to strictly match the survivors.
        """
        if (
                not np.any(self.peaks)
                or not np.any(self.der_peaks_decay)
                or not self.peak_boundaries
                    ):
            self.peaks = np.zeros_like(self.peaks)
            self.der_peaks_decay = np.zeros_like(self.der_peaks_decay)
            self.peak_boundaries = []
            self.zero_pass_decay = np.zeros_like(self.zero_pass_decay)
            return

        sig_len = len(self.peaks)

        # ---------------------------------------------------------
        # 1. Extract Decay Slope Intervals using State Machine
        # ---------------------------------------------------------
        slope_intervals = []
        # Use raw, unmutated zero passes to preserve state machine syntax across multiple calls
        source_zero_pass = getattr(self, "raw_zero_pass_decay", self.zero_pass_decay)

        if np.any(source_zero_pass):
            zc_indices = np.where(source_zero_pass != 0)[0]
            in_event = False
            start_idx = 0

            for idx in zc_indices:
                val = source_zero_pass[idx]
                if val == -1 and not in_event:
                    start_idx = idx
                    in_event = True
                elif val == 1 and in_event:
                    slope_intervals.append((start_idx, idx))
                    in_event = False
                elif val == 2:
                    if in_event:
                        slope_intervals.append((start_idx, idx))
                    start_idx = idx
                    in_event = True

            if in_event:
                slope_intervals.append((start_idx, sig_len - 1))
        else:
            ms_idx = np.where(self.der_peaks_decay > 0)[0]
            for slope_idx in ms_idx:
                slope_intervals.append((slope_idx, slope_idx))
        # ---------------------------------------------------------
        # 2. Build the Base Regional Masks & Find Overlaps
        # ---------------------------------------------------------
        peak_mask = np.zeros(sig_len, dtype=bool)
        for p_idx, p_start, p_end in self.peak_boundaries:
            # ONLY mask the response decay: peak to end
            peak_mask[p_idx: p_end + 1] = True

        slope_mask = np.zeros(sig_len, dtype=bool)
        slope_rises = []  # Cache the slope rise regions (start to derivative peak)

        for start, end in slope_intervals:
            # Find the peak of the decay slope inside this zero-cross interval
            s_indices = np.where(self.der_peaks_decay[start: end + 1] > 0)[0] + start
            if s_indices.size > 0:
                s_peak_idx = s_indices[np.argmax(self.der_peaks_decay[s_indices])]
                slope_rises.append((start, s_peak_idx))

                # ONLY mask the decay slope rise phase: start to derivative peak
                slope_mask[start: s_peak_idx + 1] = True

        overlap_mask = peak_mask & slope_mask

        # ---------------------------------------------------------
        # 3. Construct the Combined Event Footprint Mask
        # ---------------------------------------------------------
        combined_event_mask = np.zeros(sig_len, dtype=bool)

        for p_idx, p_start, p_end in self.peak_boundaries:
            # Check overlap and build island strictly using the decay region
            if np.any(overlap_mask[p_idx: p_end + 1]):
                # Shift end by 1 to prevent fusing at the shared right boundary
                safe_end = p_end - 1 if p_end > p_idx else p_end
                combined_event_mask[p_idx: safe_end + 1] = True

        for start, s_peak_idx in slope_rises:
            # Check overlap and build island strictly using the slope rise region
            if np.any(overlap_mask[start: s_peak_idx + 1]):
                # Shift start by 1 to prevent fusing at the shared left boundary
                safe_start = start + 1 if start < s_peak_idx else start
                combined_event_mask[safe_start: s_peak_idx + 1] = True

        # ---------------------------------------------------------
        # 4. Find Contiguous Islands of the Combined Mask
        # ---------------------------------------------------------
        padded = np.zeros(sig_len + 2, dtype=bool)
        padded[1:-1] = combined_event_mask
        changes = np.diff(padded.astype(int))

        island_starts = np.where(changes == 1)[0]
        island_ends = np.where(changes == -1)[0]

        # ---------------------------------------------------------
        # 5. Iterate Through Intervals and Select Winners
        # ---------------------------------------------------------
        best_peaks = []
        best_slopes = []

        for start, end in zip(island_starts, island_ends):
            peak_indices = np.where(self.peaks[start:end] > 0)[0] + start

            if peak_indices.size > 0:
                max_p_idx = peak_indices[np.argmax(self.peaks[peak_indices])]
                best_peaks.append(max_p_idx)

                # Search strictly AFTER or AT the response peak up to the island end
                slope_indices = (
                        np.where(self.der_peaks_decay[max_p_idx:end] > 0)[0] + max_p_idx
                                    )

                if slope_indices.size > 0:
                    max_s_idx = slope_indices[np.argmax(self.der_peaks_decay[slope_indices])]
                    best_slopes.append(max_s_idx)

        # ---------------------------------------------------------
        # 6. Reconstruct the Target Arrays
        # ---------------------------------------------------------
        new_peaks = np.zeros_like(self.peaks)
        if best_peaks:
            new_peaks[best_peaks] = self.peaks[best_peaks]
        self.peaks = new_peaks

        new_der_peaks = np.zeros_like(self.der_peaks_decay)
        if best_slopes:
            new_der_peaks[best_slopes] = self.der_peaks_decay[best_slopes]
        self.der_peaks_decay = new_der_peaks

        # ---------------------------------------------------------
        # 7. Clean up Boundaries
        # ---------------------------------------------------------
        valid_peak_set = set(best_peaks)
        self.peak_boundaries = [
                b for b in self.peak_boundaries if b[0] in valid_peak_set
                ]

        # ---------------------------------------------------------
        # 8. Clean up Zero Passes
        # ---------------------------------------------------------
        new_zero_pass = np.zeros_like(self.zero_pass_decay)
        if best_slopes and slope_intervals:
            best_slopes_arr = np.array(best_slopes)
            slope_starts = np.array([s for s, e in slope_intervals])
            slope_ends = np.array([e for s, e in slope_intervals])

            # Map surviving slopes back to their decay zero-cross interval
            interval_idx = np.searchsorted(slope_ends, best_slopes_arr)

            valid_mask = (interval_idx < len(slope_starts)) & (slope_starts[interval_idx] <= best_slopes_arr)

            valid_interval_idx = interval_idx[valid_mask]
            valid_starts = slope_starts[valid_interval_idx]
            valid_ends = slope_ends[valid_interval_idx]

            # Restore original zero-crossing values directly from source_zero_pass
            new_zero_pass[valid_starts] = source_zero_pass[valid_starts]
            new_zero_pass[valid_ends] = source_zero_pass[valid_ends]

        self.zero_pass_decay = new_zero_pass

    @timing
    def align_peaks_to_slopes_rise(self) -> None:
        """
        Aligns peaks and slopes by identifying combined overlapping intervals.
        Selects the maximum peak per interval, the maximum slope before it,
        and filters boundaries and zero crossings to strictly match the survivors.
        """
        if not np.any(self.peaks) or not np.any(self.der_peaks_rise) or not self.peak_boundaries:
            self.peaks = np.zeros_like(self.peaks)
            self.der_peaks_rise = np.zeros_like(self.der_peaks_rise)
            self.peak_boundaries = []
            self.zero_pass_rise = np.zeros_like(self.zero_pass_rise)
            return

        sig_len = len(self.peaks)

        # ---------------------------------------------------------
        # 1. Extract Slope Intervals using State Machine
        # ---------------------------------------------------------
        slope_intervals = []
        # Use raw, unmutated zero passes to preserve state machine syntax across multiple calls
        source_zero_pass = getattr(self, "raw_zero_pass_rise", self.zero_pass_rise)

        if np.any(source_zero_pass):
            zc_indices = np.where(source_zero_pass != 0)[0]
            in_event = False
            start_idx = 0

            for idx in zc_indices:
                val = source_zero_pass[idx]
                if val == -1 and not in_event:
                    start_idx = idx
                    in_event = True
                elif val == 1 and in_event:
                    slope_intervals.append((start_idx, idx))
                    in_event = False
                elif val == 2:
                    if in_event:
                        slope_intervals.append((start_idx, idx))
                    start_idx = idx
                    in_event = True

            if in_event:
                slope_intervals.append((start_idx, sig_len - 1))

        # ---------------------------------------------------------
        # 2. Build the Base Regional Masks & Find Overlaps
        # ---------------------------------------------------------
        peak_mask = np.zeros(sig_len, dtype=bool)
        for p_idx, p_start, p_end in self.peak_boundaries:
            # ONLY mask the response rise: start to peak
            peak_mask[p_start:p_idx + 1] = True

        slope_mask = np.zeros(sig_len, dtype=bool)
        slope_decays = []  # Cache the decay regions to use in Step 3

        for start, end in slope_intervals:
            # Find the peak of the slope inside this specific zero-cross interval
            s_indices = np.where(self.der_peaks_rise[start:end + 1] > 0)[0] + start
            if s_indices.size > 0:
                s_peak_idx = s_indices[np.argmax(self.der_peaks_rise[s_indices])]
                slope_decays.append((s_peak_idx, end))

                # ONLY mask the slope decay: peak to end
                slope_mask[s_peak_idx:end + 1] = True

        overlap_mask = peak_mask & slope_mask

        # ---------------------------------------------------------
        # 3. Construct the Combined Event Footprint Mask
        # ---------------------------------------------------------
        combined_event_mask = np.zeros(sig_len, dtype=bool)

        for p_idx, p_start, p_end in self.peak_boundaries:
            # Check overlap and build island strictly using the rise region
            if np.any(overlap_mask[p_start:p_idx + 1]):
                # Shift start by 1 to prevent fusing at the shared left boundary
                safe_start = p_start + 1 if p_start < p_idx else p_start
                combined_event_mask[safe_start:p_idx + 1] = True

        for s_peak_idx, end in slope_decays:
            # Check overlap and build island strictly using the decay region
            if np.any(overlap_mask[s_peak_idx:end + 1]):
                # Shift end by 1 to prevent fusing at the shared right boundary
                safe_end = end - 1 if end > s_peak_idx else end
                combined_event_mask[s_peak_idx:safe_end + 1] = True

        # ---------------------------------------------------------
        # 4. Find Contiguous Islands of the Combined Mask
        # ---------------------------------------------------------
        padded = np.zeros(sig_len + 2, dtype=bool)
        padded[1:-1] = combined_event_mask
        changes = np.diff(padded.astype(int))

        island_starts = np.where(changes == 1)[0]
        island_ends = np.where(changes == -1)[0]

        # ---------------------------------------------------------
        # 5. Iterate Through Intervals and Select Winners
        # ---------------------------------------------------------
        best_peaks = []
        best_slopes = []

        for start, end in zip(island_starts, island_ends):
            peak_indices = np.where(self.peaks[start:end] > 0)[0] + start

            if peak_indices.size > 0:
                max_p_idx = peak_indices[np.argmax(self.peaks[peak_indices])]
                best_peaks.append(max_p_idx)

                # ADDED '+ 1' to include the peak index itself in case they happen simultaneously
                slope_indices = np.where(self.der_peaks_rise[start:max_p_idx + 1] > 0)[0] + start

                # RESTORED defensive check to prevent np.argmax() from crashing on empty arrays
                if slope_indices.size > 0:
                    max_s_idx = slope_indices[np.argmax(self.der_peaks_rise[slope_indices])]
                    best_slopes.append(max_s_idx)

        # ---------------------------------------------------------
        # 6. Reconstruct the Target Arrays
        # ---------------------------------------------------------
        new_peaks = np.zeros_like(self.peaks)
        if best_peaks:
            new_peaks[best_peaks] = self.peaks[best_peaks]
        self.peaks = new_peaks

        new_der_peaks = np.zeros_like(self.der_peaks_rise)
        if best_slopes:
            new_der_peaks[best_slopes] = self.der_peaks_rise[best_slopes]
        self.der_peaks_rise = new_der_peaks

        # ---------------------------------------------------------
        # 7. Clean up Boundaries
        # ---------------------------------------------------------
        valid_peak_set = set(best_peaks)
        self.peak_boundaries = [
                b for b in self.peak_boundaries if b[0] in valid_peak_set
                ]

        # ---------------------------------------------------------
        # 8. Clean up Zero Passes
        # ---------------------------------------------------------
        new_zero_pass = np.zeros_like(self.zero_pass_rise)
        if best_slopes and slope_intervals:
            best_slopes_arr = np.array(best_slopes)
            slope_starts = np.array([s for s, e in slope_intervals])
            slope_ends = np.array([e for s, e in slope_intervals])

            # Map each surviving slope back to its specific [start, end] zero-cross interval
            interval_idx = np.searchsorted(slope_ends, best_slopes_arr)

            # Verify the index falls strictly within the interval bounds
            valid_mask = (interval_idx < len(slope_starts)) & (slope_starts[interval_idx] <= best_slopes_arr)

            valid_interval_idx = interval_idx[valid_mask]
            valid_starts = slope_starts[valid_interval_idx]
            valid_ends = slope_ends[valid_interval_idx]

            # Keep only the original zero passes that bounded the surviving slopes
            new_zero_pass[valid_starts] = source_zero_pass[valid_starts]
            new_zero_pass[valid_ends] = source_zero_pass[valid_ends]

        self.zero_pass_rise = new_zero_pass

    @timing
    def align_peaks_to_slopes(self) -> None:
        # Save unmutated originals ONCE before running alignment passes
        self.raw_zero_pass_rise = self.zero_pass_rise.copy()
        self.raw_zero_pass_decay = self.zero_pass_decay.copy()
        # This needs to be sequential
        self.align_peaks_to_slopes_rise()
        self.align_peaks_to_slopes_decay()
        self.align_peaks_to_slopes_rise()

    @timing
    def get_pulse_arr(self, name):
        return np.array([values[name] for values in self.pul_attrs.values()])


class EvtPro(Analyzer):
    """Event detection class"""

    def __init__(self, path_to_file="", initialize=True, location=0):
        super().__init__(path_to_file, initialize, location)
        self.burst_threshold = None
        self.burst_smoothed_ifreq = None
        self.burst_continuous_ifreq = None
        self.burst_continuous_time = None
        self.default_event = {
                # --- Original Keys ---
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
                "sd_segm"         : None,  # max decay slope location
                "z_segm"          : None,  # zero pass locations, critical points
                "b_amp"           : None,  # Baseline amplitude
                "amplitude"       : None,  # Absolute amplitude of the response
                "r_ifreq"         : None,  # Instantaneous frequency of the response
                "r_inter"         : None,  # Interval between the response and the previous response
                # I(t) = pk0 * exp(-t / t0)
                "fit_min"         : 0.0,  # Fit of i0
                "fit_peak"        : None,  # Fit of pk0
                "tau"             : None,  # Exp. decay fit constant, t0
                "mse_fit"         : None,  # Minimal standard error of the fit
                "pearson_r"       : 0.01,  # Pearson's R of the fit
                "fit_valid"       : False,  # Validity of the fit
                "r_auc"           : None,  # Area under the curve of the response
                "threshold_segm"  : None,  # threshold segment
                "ap_threshold"    : None,  # threshold value
                # --- Newly Added Keys (From screen_events logic) ---
                "rise_slope_val"  : None,  # RISE velocity value used in constraints
                "decay_slope_val" : None,  # DECAY velocity value used in constraints
                "peak_error"      : None,  # Deviation between original peak time and zero-pass peak time
                "rise_time_peak"  : None,  # Activation rise time (original peak time - zero pass start)
                "rise_time_der"   : None,
                # Rise time calculated via derivative crossings (zero pass peak - zero pass start)
                "slope_pos_delta" : None,  # Index delta from max slope to peak
                "start_pos_delta" : None,  # Index delta from zero pass start to peak
                "peak_pos_delta"  : None,  # Index delta from zero-pass peak to alignment peak
                }
        self.events_attrs = {}
        self.rejected_counts = {}
        self.ps_nsfa_values = {}
        self.burst_attrs = {}
        self.ifreq_blocks = np.array([])
        self.mfreq_blocks = np.array([])
        self._events_positions = np.array([])
        self.common_time = np.array([])

    def _reject_event(self, evt_time: float, reason: str, msg: str = ""):
        """
        Handles cleanup, metrics tracking, and logging for rejected events
        across the entire processing pipeline.
        """
        # 1. Calculate and cache the file path on the first call
        if not hasattr(self, '_rejection_log_path'):
            # Pass `self` (the EvtPro instance) and `__file__` (the current script)
            file_name, _, file_parent, script_name = get_names(self, __file__)
            csv_filename = f"{file_name}_{script_name}_rejected_events.csv"
            self._rejection_log_path = os.path.join(file_parent, csv_filename)

        log_file = self._rejection_log_path

        # 2. Check if file exists to write headers on the first pass
        file_exists = os.path.isfile(log_file)

        # 3. Append the rejection data to the CSV
        with open(log_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Event_Time", "Reason", "Message"])
            writer.writerow([evt_time, reason, msg])

        # 4. Track the rejection reason globally
        self.rejected_counts[reason] = self.rejected_counts.get(reason, 0) + 1

        # 5. Evict the event from the main dictionary using its absolute time key
        self.events_attrs.pop(evt_time, None)

    # def _reject_event(self, evt_time: float, reason: str, msg: str = ""):
    #     """
    #     Handles cleanup, metrics tracking, and logging for rejected events
    #     across the entire processing pipeline.
    #     """
    #     if msg:
    #         print(msg)
    #
    #     # Track the rejection reason globally
    #     self.rejected_counts[reason] = self.rejected_counts.get(reason, 0) + 1
    #
    #     # Evict the event from the main dictionary using its absolute time key
    #     self.events_attrs.pop(evt_time, None)

    @timing
    def _select_events(self):
        """
        Extracts foundational kinetic properties for detected peaks and initializes valid putative events.

        Iterates over all identified peak indices, utilizing the interval between consecutive
        peaks to define regional boundaries. Isolates the maximum rising and decay slopes,
        calculates the absolute start time of the event rise, and refines the peak location.
        """
        self.events_attrs = {}

        # 1. Evaluate the condition ONCE and store the indices
        peak_indices = np.where(self.peaks > 0)[0]
        self.peaks[peak_indices] = 1.0
        num_peaks = len(peak_indices)

        if num_peaks == 0:
            print("No peaks detected.")
            return

        # 2. Extract marker indices ONCE for fast bounded lookups
        slope_indices = np.where(self.der_peaks_rise > 0)[0]
        decay_slope_indices = np.where(self.der_peaks_decay > 0)[0]
        zc_indices = np.where(self.zero_pass_rise != 0)[0]

        for i in range(num_peaks):
            evt_pos = peak_indices[i]
            evt_time = float(self.time[evt_pos])

            # Define regional boundaries
            prev_pos = peak_indices[i - 1] if i > 0 else 0
            next_pos = peak_indices[i + 1] if i < num_peaks - 1 else len(self.time) - 1

            # ---------------------------------------------------------
            # 1. Bounded Rise Slope Lookup
            # ---------------------------------------------------------
            s_start = np.searchsorted(slope_indices, prev_pos, side="left")
            s_end = np.searchsorted(slope_indices, evt_pos, side="right")

            if s_start == s_end:
                msg = f"{evt_time:12.4f}[s] rejected (No slope found in region)"
                self._reject_event(evt_time, "no_slope_region", msg)
                continue

            abs_slope_pos = slope_indices[s_end - 1]
            slope_value = float(self.derivative[abs_slope_pos])
            slope_time = float(self.time[abs_slope_pos])

            # ---------------------------------------------------------
            # 2. Bounded Decay Slope Lookup
            # ---------------------------------------------------------
            sd_start = np.searchsorted(decay_slope_indices, evt_pos, side="left")
            sd_end = np.searchsorted(decay_slope_indices, next_pos, side="right")

            if sd_start == sd_end:
                msg = f"{evt_time:12.4f}[s] rejected (No decay slope found in region)"
                self._reject_event(evt_time, "no_decay_slope_region", msg)
                continue

            abs_decay_slope_pos = decay_slope_indices[sd_start]  # First decay slope after peak
            decay_slope_value = float(self.derivative_decay[abs_decay_slope_pos])
            decay_slope_time = float(self.time[abs_decay_slope_pos])

            # ---------------------------------------------------------
            # 3. Bounded Zero-Crossings Lookup
            # ---------------------------------------------------------
            zs_limit = max(0, prev_pos - 1)
            zc_s_start = np.searchsorted(zc_indices, zs_limit, side="left")
            zc_s_end = np.searchsorted(zc_indices, abs_slope_pos, side="right")

            if zc_s_start == zc_s_end:
                msg = f"{evt_time:12.4f}[s] rejected (crossings not found)"
                self._reject_event(evt_time, "crossings_not_found", msg)
                continue

            abs_zs_pos = zc_indices[zc_s_end - 1]

            zc_p_start = np.searchsorted(zc_indices, abs_slope_pos, side="left")
            zc_p_end = np.searchsorted(zc_indices, next_pos, side="right")

            if zc_p_start == zc_p_end:
                msg = f"{evt_time:12.4f}[s] rejected (crossings not found)"
                self._reject_event(evt_time, "crossings_not_found", msg)
                continue

            abs_zp_pos = zc_indices[zc_p_start]

            # ---------------------------------------------------------
            # 4. Direct Absolute Assignment
            # ---------------------------------------------------------
            t_o_zs = float(self.time[abs_zs_pos])
            t_o_zp = float(self.time[abs_zp_pos])

            self.events_attrs[evt_time] = self.default_event.copy()
            self.events_attrs[evt_time].update(
                    {
                            "t_o_s"                : slope_time,
                            "t_o_sd"               : decay_slope_time,
                            "slope_peak_delta_time": evt_time - slope_time,
                            "rise_slope_val"       : slope_value,
                            "decay_slope_val"      : decay_slope_value,
                            "t_o_zs"               : t_o_zs,
                            "t_o_zp"               : t_o_zp,
                            "peak_error_time"      : evt_time - t_o_zp,
                            "rise_time_peak"       : evt_time - t_o_zs,
                            "rise_time_der"        : t_o_zp - t_o_zs,
                            }
                    )

        print(
                f" Accepted events: {len(self.events_attrs)}, Rejected: {num_peaks - len(self.events_attrs)}"
                )

    @timing
    def _event_sections(self, t_aft, baseline_time, peak_to_peak=0.001):
        """
        Extracts and isolates the specific waveform data arrays for each validated event
        using absolute time boundaries.
        """
        evt_times_list = list(self.events_attrs.keys())
        total_initial_events = len(evt_times_list)

        peak_times = self.time[self.peaks > 0]
        pulse_times = (self.time[self.pulses_peaks > 0] if self.cdac.size else np.array([]))

        for i, evt_time in enumerate(evt_times_list):
            # =========================================================
            # 1. TIME-BASED BOUNDARY CALCULATIONS
            # =========================================================
            if i + 1 < len(evt_times_list):
                next_evt_time = evt_times_list[i + 1]
                next_evt_start_time = self.events_attrs[next_evt_time]["t_o_zs"]
            else:
                next_evt_start_time = float(self.time[-1])

            p_idx = np.searchsorted(
                    peak_times, evt_time + peak_to_peak, side="right"
                    )
            if (
                    p_idx < len(peak_times)
                    and peak_times[p_idx] < next_evt_start_time
                    ):
                inter_time = peak_times[p_idx]
            else:
                inter_time = next_evt_start_time

            if pulse_times.size > 0:
                pl_idx = np.searchsorted(pulse_times, evt_time, side="right")
                if (
                        pl_idx < len(pulse_times)
                        and pulse_times[pl_idx] < next_evt_start_time
                        ):
                    pulse_time = pulse_times[pl_idx]
                else:
                    pulse_time = next_evt_start_time
            else:
                pulse_time = next_evt_start_time

            end_roi_time = min(
                    evt_time + t_aft, inter_time, pulse_time, next_evt_start_time
                    )

            t_o_zs = self.events_attrs[evt_time]["t_o_zs"]
            t_o_zp = self.events_attrs[evt_time]["t_o_zp"]
            start_roi_time = max(
                    float(self.time[0]), evt_time - t_aft, t_o_zs - baseline_time
                    )

            # =========================================================
            # 2. DYNAMIC SLICING VIA vtp_relative
            # =========================================================
            start_roi_idx = vtp_relative(start_roi_time, self.time)
            end_roi_idx = vtp_relative(end_roi_time, self.time)

            roi_slice = slice(start_roi_idx, end_roi_idx + 1)

            if end_roi_idx - start_roi_idx < 2:
                msg = f"{evt_time:12.4f}[s] rejected, short response. Indices: {start_roi_idx} to {end_roi_idx}"
                self._reject_event(evt_time, "short_response", msg)
                continue

            t_segm = self.time[roi_slice]
            if t_segm.size > 0 and (t_segm[-1] - t_segm[0]) > 1.0:
                print(
                        f"Event at {evt_time:.4f}s exceeds 1s: length = {t_segm[-1] - t_segm[0]:.4f}s"
                        )

            if len(t_segm) == 0:
                self._reject_event(
                        evt_time,
                        "empty_segment",
                        f"{evt_time:12.4f}[s] rejected, completely empty segment.",
                        )
                continue

            if not (t_segm[0] <= t_o_zs <= t_segm[-1]) or not (t_segm[0] <= t_o_zp <= t_segm[-1]):
                msg = (
                        f"{evt_time:12.4f}[s] rejected, zero crossings out of bounds by t_aft constraint. "
                        f"Segment: [{t_segm[0]:.4f}, {t_segm[-1]:.4f}], zs: {t_o_zs:.4f}, zp: {t_o_zp:.4f}"
                        )
                self._reject_event(evt_time, "zero_pass_error", msg)
                continue

            # =========================================================
            # 3. SEGMENT EXTRACTION & FORMATTING
            # =========================================================
            z_segm = np.zeros_like(self.time[roi_slice])
            z_segm[vtp_relative(t_o_zp, t_segm)] = 1
            z_segm[vtp_relative(t_o_zs, t_segm)] = -1

            p_segm = self.peaks[roi_slice]
            num_peaks_in_segm = np.count_nonzero(p_segm > 0)

            if num_peaks_in_segm > 1:
                p_segm = reset_array(
                        self.peaks[roi_slice], vtp_relative(evt_time, t_segm)
                        )
            elif num_peaks_in_segm == 0:
                self._reject_event(
                        evt_time,
                        "no_peak_detected",
                        f"{evt_time:12.4f}[s] rejected, No peaks detected in final slice.",
                        )
                continue

            r_segm = self.resp[roi_slice]
            d_segm = self.derivative[roi_slice]

            t_o_s = self.events_attrs[evt_time]["t_o_s"]
            s_segm = reset_array(
                    self.der_peaks_rise[roi_slice], vtp_relative(t_o_s, t_segm)
                    )

            t_o_sd = self.events_attrs[evt_time]["t_o_sd"]
            sd_segm = reset_array(
                    self.der_peaks_decay[roi_slice], vtp_relative(t_o_sd, t_segm)
                    )

            self.events_attrs[evt_time].update(
                    {
                            "end_time": float(t_segm[-1] - evt_time),
                            "t_segm"  : t_segm,
                            "r_segm"  : r_segm,
                            "p_segm"  : p_segm,
                            "d_segm"  : d_segm,
                            "s_segm"  : s_segm,
                            "sd_segm" : sd_segm,
                            "z_segm"  : z_segm,
                            }
                    )

        print(
                f" Accepted events: {len(self.events_attrs)}, Rejected: {total_initial_events - len(self.events_attrs)}"
                )

    # @timing
    # def _select_events(self):
    #     """
    #     Extracts foundational kinetic properties for detected peaks and initializes valid putative events.
    #
    #     Iterates over all identified peak indices, utilizing the interval between consecutive
    #     peaks to define regional boundaries. Isolates the maximum rising slope,
    #     calculates the absolute start time of the event rise, and refines the peak location.
    #     """
    #     self.events_attrs = {}
    #
    #     # 1. Evaluate the condition ONCE and store the indices
    #     peak_indices = np.where(self.peaks > 0)[0]
    #     self.peaks[peak_indices] = 1.0
    #     num_peaks = len(peak_indices)
    #
    #     if num_peaks == 0:
    #         print("No peaks detected.")
    #         return
    #
    #     # 2. Extract marker indices ONCE for fast bounded lookups
    #     slope_indices = np.where(self.der_peaks_rise > 0)[0]
    #     zc_indices = np.where(self.zero_pass_rise != 0)[0]  # Restored to match crossing_point (!= 0) behavior
    #
    #     for i in range(num_peaks):
    #         evt_pos = peak_indices[i]
    #         evt_time = float(self.time[evt_pos])
    #
    #         # Define regional boundaries exactly as before
    #         prev_pos = peak_indices[i - 1] if i > 0 else 0
    #         next_pos = peak_indices[i + 1] if i < num_peaks - 1 else len(self.time) - 1
    #
    #         # ---------------------------------------------------------
    #         # 1. Restored Bounded Slope Lookup
    #         # Equivalent to: np.where(self.der_peaks[prev_pos:evt_pos + 1] > 0)
    #         # ---------------------------------------------------------
    #         s_start = np.searchsorted(slope_indices, prev_pos, side='left')
    #         s_end = np.searchsorted(slope_indices, evt_pos, side='right')
    #
    #         if s_start == s_end:
    #             msg = f"{evt_time:12.4f}[s] rejected (No slope found in region)"
    #             self._reject_event(evt_time, "no_slope_region", msg)
    #             continue
    #
    #         abs_slope_pos = slope_indices[s_end - 1]  # [-1] grabs the exact same slope as the old code
    #         slope_value = float(self.derivative[abs_slope_pos])
    #         slope_time = float(self.time[abs_slope_pos])
    #
    #         # ---------------------------------------------------------
    #         # 2. Restored Bounded Zero-Crossings Lookup
    #         # ---------------------------------------------------------
    #         # zs_p equivalent: Find the LAST non-zero pass in [max(0, prev_pos - 1), abs_slope_pos]
    #         zs_limit = max(0, prev_pos - 1)
    #         zc_s_start = np.searchsorted(zc_indices, zs_limit, side='left')
    #         zc_s_end = np.searchsorted(zc_indices, abs_slope_pos, side='right')
    #
    #         if zc_s_start == zc_s_end:
    #             msg = f"{evt_time:12.4f}[s] rejected (crossings not found)"
    #             self._reject_event(evt_time, "crossings_not_found", msg)
    #             continue
    #
    #         abs_zs_pos = zc_indices[zc_s_end - 1]
    #
    #         # zp_p equivalent: Find the FIRST non-zero pass in [abs_slope_pos, next_pos]
    #         zc_p_start = np.searchsorted(zc_indices, abs_slope_pos, side='left')
    #         zc_p_end = np.searchsorted(zc_indices, next_pos, side='right')
    #
    #         if zc_p_start == zc_p_end:
    #             msg = f"{evt_time:12.4f}[s] rejected (crossings not found)"
    #             self._reject_event(evt_time, "crossings_not_found", msg)
    #             continue
    #
    #         abs_zp_pos = zc_indices[zc_p_start]
    #
    #         # ---------------------------------------------------------
    #         # 3. Direct Absolute Assignment
    #         # ---------------------------------------------------------
    #         t_o_zs = float(self.time[abs_zs_pos])
    #         t_o_zp = float(self.time[abs_zp_pos])
    #
    #         self.events_attrs[evt_time] = self.default_event.copy()
    #         self.events_attrs[evt_time].update(
    #                 {
    #                         "t_o_s"                : slope_time,
    #                         "slope_peak_delta_time": evt_time - slope_time,
    #                         "rise_slope_val"       : slope_value,
    #                         "t_o_zs"               : t_o_zs,
    #                         "t_o_zp"               : t_o_zp,
    #                         "peak_error_time"      : evt_time - t_o_zp,
    #                         "rise_time_peak"       : evt_time - t_o_zs,
    #                         "rise_time_der"        : t_o_zp - t_o_zs
    #                         }
    #                 )
    #
    #     print(f" Accepted events: {len(self.events_attrs)}, Rejected: {num_peaks - len(self.events_attrs)}")
    #
    # @timing
    # def _event_sections(self, t_aft, baseline_time, peak_to_peak=0.001):
    #     """
    #     Extracts and isolates the specific waveform data arrays for each validated event
    #     using absolute time boundaries.
    #     """
    #     evt_times_list = list(self.events_attrs.keys())
    #     total_initial_events = len(evt_times_list)
    #
    #     # Pre-extract global event times to eliminate O(N) slicing inside the loop
    #     peak_times = self.time[self.peaks > 0]
    #     pulse_times = self.time[self.pulses_peaks > 0] if self.cdac.size else np.array([])
    #
    #     for i, evt_time in enumerate(evt_times_list):
    #         # =========================================================
    #         # 1. TIME-BASED BOUNDARY CALCULATIONS
    #         # =========================================================
    #         if i + 1 < len(evt_times_list):
    #             next_evt_time = evt_times_list[i + 1]
    #             next_evt_start_time = self.events_attrs[next_evt_time]["t_o_zs"]
    #         else:
    #             next_evt_start_time = float(self.time[-1])
    #
    #         # Fast binary lookup for trailing peaks (interferences)
    #         # Replaces: decay_peaks_idx = np.where(decay_peaks > 0)[0]
    #         p_idx = np.searchsorted(peak_times, evt_time + peak_to_peak, side='right')
    #         if p_idx < len(peak_times) and peak_times[p_idx] < next_evt_start_time:
    #             inter_time = peak_times[p_idx]
    #         else:
    #             inter_time = next_evt_start_time
    #
    #         # Fast binary lookup for pulse artifacts
    #         # Replaces: pulse_idx = np.argmax(self.pulses_peaks[decay_region])
    #         if pulse_times.size > 0:
    #             pl_idx = np.searchsorted(pulse_times, evt_time, side='right')
    #             if pl_idx < len(pulse_times) and pulse_times[pl_idx] < next_evt_start_time:
    #                 pulse_time = pulse_times[pl_idx]
    #             else:
    #                 pulse_time = next_evt_start_time
    #         else:
    #             pulse_time = next_evt_start_time
    #
    #         # Boundaries structurally identical to original
    #         end_roi_time = min(evt_time + t_aft, inter_time, pulse_time, next_evt_start_time)
    #
    #         t_o_zs = self.events_attrs[evt_time]["t_o_zs"]
    #         t_o_zp = self.events_attrs[evt_time]["t_o_zp"]
    #         start_roi_time = max(float(self.time[0]), evt_time - t_aft, t_o_zs - baseline_time)
    #
    #         # =========================================================
    #         # 2. DYNAMIC SLICING VIA vtp_relative
    #         # =========================================================
    #         start_roi_idx = vtp_relative(start_roi_time, self.time)
    #         end_roi_idx = vtp_relative(end_roi_time, self.time)
    #
    #         roi_slice = slice(start_roi_idx, end_roi_idx + 1)
    #
    #         # Rejections kept identical
    #         if end_roi_idx - start_roi_idx < 2:
    #             msg = f"{evt_time:12.4f}[s] rejected, short response. Indices: {start_roi_idx} to {end_roi_idx}"
    #             self._reject_event(evt_time, "short_response", msg)
    #             continue
    #
    #         t_segm = self.time[roi_slice]
    #         if t_segm.size > 0 and (t_segm[-1] - t_segm[0]) > 1.0:
    #             print(f"Event at {evt_time:.4f}s exceeds 1s: length = {t_segm[-1] - t_segm[0]:.4f}s")
    #
    #         if len(t_segm) == 0:
    #             self._reject_event(
    #                     evt_time, "empty_segment", f"{evt_time:12.4f}[s] rejected, completely empty segment."
    #                     )
    #             continue
    #
    #         if not (t_segm[0] <= t_o_zs <= t_segm[-1]) or not (t_segm[0] <= t_o_zp <= t_segm[-1]):
    #             msg = (f"{evt_time:12.4f}[s] rejected, zero crossings out of bounds by t_aft constraint. "
    #                    f"Segment: [{t_segm[0]:.4f}, {t_segm[-1]:.4f}], zs: {t_o_zs:.4f}, zp: {t_o_zp:.4f}")
    #             self._reject_event(evt_time, "zero_pass_error", msg)
    #             continue
    #
    #         # =========================================================
    #         # 3. SEGMENT EXTRACTION & FORMATTING
    #         # =========================================================
    #         # z_segm = reset_array(self.zero_pass[roi_slice], vtp_relative(t_o_zp, t_segm))
    #         z_segm = np.zeros_like(self.time[roi_slice])  # Blank canvas for the segment
    #         z_segm[vtp_relative(t_o_zp, t_segm)] = 1  # Safely force end to 1
    #         z_segm[vtp_relative(t_o_zs, t_segm)] = -1
    #
    #         p_segm = self.peaks[roi_slice]
    #         num_peaks_in_segm = np.count_nonzero(p_segm > 0)
    #
    #         if num_peaks_in_segm > 1:
    #             p_segm = reset_array(self.peaks[roi_slice], vtp_relative(evt_time, t_segm))
    #         elif num_peaks_in_segm == 0:
    #             self._reject_event(
    #                     evt_time, "no_peak_detected", f"{evt_time:12.4f}[s] rejected, No peaks detected in final slice."
    #                     )
    #             continue
    #
    #         r_segm = self.resp[roi_slice]
    #         d_segm = self.derivative[roi_slice]
    #
    #         t_o_s = self.events_attrs[evt_time]["t_o_s"]
    #         s_segm = reset_array(self.der_peaks_rise[roi_slice], vtp_relative(t_o_s, t_segm))
    #
    #         # Consolidated dictionary updates
    #         self.events_attrs[evt_time].update(
    #                 {
    #                         "end_time": float(t_segm[-1] - evt_time),
    #                         "t_segm"  : t_segm,
    #                         "r_segm"  : r_segm,
    #                         "p_segm"  : p_segm,
    #                         "d_segm"  : d_segm,
    #                         "s_segm"  : s_segm,
    #                         "z_segm"  : z_segm
    #                         }
    #                 )
    #
    #     print(f" Accepted events: {len(self.events_attrs)}, Rejected: {total_initial_events - len(self.events_attrs)}")

    @timing
    def get_evt(self, peak_to_peak: float = 0.01, t_aft=0.04, baseline_time=0.002):
        """
            Orchestrates the two-pass event detection and waveform segmentation pipeline.

            Serves as the primary internal entry point for capturing raw events. It sequentially
            executes the foundational zero-pass property extraction (`_select_events`) and follows
            up by dynamically slicing and validating the localized data arrays for each captured
            event (`_event_sections`).

            Args:
                peak_to_peak (float): Minimum time interval (in seconds) required between consecutive
                                      peaks to avoid truncating the decay phase.
                t_aft (float): Maximum allowed decay window (in seconds) to capture after the peak.
                baseline_time (float): Duration of the baseline to capture prior to the event onset.
            """
        self._select_events()
        self._event_sections(t_aft, baseline_time, peak_to_peak)

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

        for evt_pos in evts_attrs_copy.keys():
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

    @timing
    def get_alig(self, alignment='p'):
        t_o_p = 0
        for evt_time in list(self.events_attrs.keys()):
            match alignment:
                case 'p':
                    t_o_p = evt_time  # Directly use the absolute peak timestamp
                case 'z':
                    t_o_p = self.events_attrs[evt_time]["t_o_zp"]
                case 's':
                    t_o_p = self.events_attrs[evt_time]["t_o_s"]

            t_segm = self.events_attrs[evt_time]["t_segm"]
            t_segm = np.round(t_segm - t_o_p, decimals=precision)
            self.events_attrs[evt_time]["t_o_p"] = t_o_p
            self.events_attrs[evt_time]["t_segm"] = t_segm

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
            self, baseline_time: float = 0.005,
            peak_radius: float = 0.0, peak_type: str = 'p', adjust=True
            ) -> None:
        self._get_adj(baseline_time, adjust)
        p_r_p = vtp(peak_radius, self.t_delta)

        for evt_time in list(self.events_attrs.keys()):
            r_segm = self.events_attrs[evt_time]["r_segm"]
            match peak_type:
                case "p":
                    p_segm = self.events_attrs[evt_time]["p_segm"]
                    p_t_p = np.argmax(p_segm)
                case "z":
                    z_segm = self.events_attrs[evt_time]["z_segm"]
                    p_t_p = np.argmax(z_segm)
                case _:
                    raise ValueError(f"Wrong peak type ('p' or 'z')")

            if peak_radius > 0.0:
                amplitude = np.nanmean(r_segm[p_t_p - p_r_p: p_t_p + p_r_p + 1])
            else:
                amplitude = r_segm[p_t_p]

            # Unconditionally assign the extracted amplitude
            self.events_attrs[evt_time]["amplitude"] = amplitude

        print(f"Extracted amplitudes for {len(self.events_attrs)} putative events.")

    def get_arr(self, name, element_type="evt"):
        match element_type:
            case "evt":
                return np.array([values[name] for values in self.events_attrs.values()])
            case "burst_evt":
                # Returns the attribute ONLY for events tagged as being inside a burst
                return np.array([values[name] for values in self.events_attrs.values() if values.get("in_burst")])
            case "isolated_evt":
                # Returns the attribute ONLY for events tagged as NOT being inside a burst
                return np.array([values[name] for values in self.events_attrs.values() if not values.get("in_burst")])
            case "pul":
                return np.array([values[name] for values in self.pul_attrs.values()])
            case "burst":
                return np.array([values[name] for values in self.burst_attrs.values()])
            case _:
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

    @timing
    def get_intervals(self):
        for evt_pos, r_inter in zip(self.events_attrs, np.append([0.0], np.diff(self.get_arr("t_o_p")))):
            self.events_attrs[evt_pos]["r_inter"] = r_inter

    @timing
    def get_replaced(self, substitute: 'EvtPro', baseline_time: float = 0.005) -> None:
        self.transfer(substitute)
        for evt_pos in self.events_attrs:
            # pre = int(np.where(self.events_attrs[evt_pos]["t_segm"] == 0.0)[0])
            pre = vtp_relative(0.0, self.events_attrs[evt_pos]["t_segm"])
            post = len(self.events_attrs[evt_pos]["t_segm"]) - pre
            # To preserve alignment
            p_o_p = vtp_relative(self.events_attrs[evt_pos]["t_o_p"], self.time)
            # p_o_p = vtp(self.events_attrs[evt_pos]["t_o_p"], self.t_delta) - vtp(self.time[0], self.t_delta)
            # To preserve alignment
            self.events_attrs[evt_pos]["r_segm"] = self.resp[p_o_p - pre: p_o_p + post]
        self._get_adj(baseline_time)

    @timing
    def fit_events(self, gaussian_window=0.002, fit_beg=0.00, fit_end=0.015, sharpness=2, min_length=0.003):
        """
        Fits the decay phase of events to an exponential curve.
        Operates strictly in data-gathering mode; does not reject events.
        Dynamically truncates the fitting window when the trace drops to `amp_threshold`.
        Assumes traces are already baseline-adjusted to 0.
        """
        n_p = vtp(gaussian_window, self.t_delta)
        min_len_pts = vtp(min_length, self.t_delta)
        pos_fit_start = vtp(fit_beg, self.t_delta)
        pos_fit_end = vtp(fit_end, self.t_delta)
        dire = self.direction
        # fig, ax = plt.subplots(figsize=(8, 5))  # testing... delete me after
        # counter = 0  # testing... delete me after
        for evt_time, evt_data in self.events_attrs.copy().items():
            evt_data.update(
                    {
                            "fit_min": np.nan, "fit_peak": np.nan, "tau": np.nan,
                            "mse_fit": np.nan, "pearson_r": np.nan, "decay_slope": np.nan,
                            }
                    )

            r_segm = evt_data["r_segm"]
            t_segm = evt_data["t_segm"]
            p_segm = evt_data["p_segm"]

            peak_pos = np.argmax(p_segm)
            smoothed_resp, kernel_sd = smoothing(r_segm, n_p, sharpness, 'g', 1 / self.t_delta)

            abs_fit_start = peak_pos + pos_fit_start
            abs_fit_end = peak_pos + pos_fit_end
            s_resp_s = smoothed_resp[abs_fit_start:abs_fit_end]

            if s_resp_s.size == 0:
                msg = f"Event at {evt_time:.4f}[s] rejected: empty_slice s_resp_s.size == 0."
                self._reject_event(evt_time, "empty_slice", msg)
                continue

            # =========================================================
            # DYNAMIC DECAY TRUNCATION (POLARITY & ZERO-CROSSING)
            # =========================================================
            if dire == 1:
                # Positive-going: keep strictly positive values (> 0)
                invalid_pts = np.where(s_resp_s <= 0)[0]
                zero_cross_end = invalid_pts[0] if invalid_pts.size > 0 else s_resp_s.size
                local_ds_end = min(np.argmin(s_resp_s), zero_cross_end)
            elif dire == -1:
                # Negative-going: keep strictly negative values (< 0)
                invalid_pts = np.where(s_resp_s >= 0)[0]
                zero_cross_end = invalid_pts[0] if invalid_pts.size > 0 else s_resp_s.size
                local_ds_end = min(np.argmax(s_resp_s), zero_cross_end)
            else:
                msg = f"Event at {evt_time:.4f}[s] rejected: wrong_direction dire={dire}."
                self._reject_event(evt_time, "wrong_direction", msg)
                continue

            # =========================================================
            # RESTRICT SLICE & FIT
            # =========================================================
            fit_region_restricted = slice(abs_fit_start, abs_fit_start + local_ds_end)

            r_s_short = smoothed_resp[fit_region_restricted]
            t_short = t_segm[fit_region_restricted]

            if local_ds_end <= 0 or r_s_short.size < min_len_pts or r_s_short.size != t_short.size:
                msg = f"Event at {evt_time:.4f}[s] rejected: fit_length_problem size={r_s_short.size} (min={min_len_pts})."
                self._reject_event(evt_time, "fit_length_problem", msg)
                continue
            # =========================================================
            # Exponential fitting
            fit_i_0, fit_pk0, fit_t0, pearson_r = exp_fit(r_s_short, t_short, dire)
            # =========================================================
            r_short = r_segm[fit_region_restricted]
            fit_curve = exp_decay(t_short, fit_i_0, fit_pk0, fit_t0)  # testing... delete me after
            mse_fit = mse(r_short, fit_curve)

            # testing... delete me after
            # if np.isnan(pearson_r) and counter < 10:  # testing... delete me after
            #     ax.plot(t_short, r_s_short, color="black", alpha=0.1)  # testing... delete me after
            #     lin_resp = exp_to_lin(r_s_short, dire)
            #     ax.plot(t_short, lin_resp, color="blue", alpha=0.3)  # testing... delete me after
            #     # print(f"{'T' * 20} Testing delete me after... "
            #     #       f"{len(t_short)=} {len(r_s_short)=} {fit_i_0=} {fit_pk0=} {fit_t0=} {pearson_r=}")
            #     ax.plot(t_short, fit_curve, color="red", linewidth=1.5, alpha=0.1)  # testing... delete me after
            #     counter += 1

            evt_data.update(
                    {
                            "fit_min"    : fit_i_0,
                            "fit_peak"   : fit_pk0,
                            "tau"        : fit_t0,
                            "decay_slope": -fit_pk0 / fit_t0,
                            "mse_fit"    : mse_fit,
                            "pearson_r"  : pearson_r,
                            }
                    )

        # testing... delete me after
        # ax.set_title("testing... delete me after")  # testing... delete me after
        # ax.set_xlabel("Time (s)")  # testing... delete me after
        # ax.set_ylabel("Amplitude")  # testing... delete me after
        # plt.show()  # testing... delete me after

    @timing
    def inspect_fits(self, max_events=9, randomize=True):
        """
        Generates a grid plot of event traces overlaid with their exponential fits.

        Parameters:
        - max_events: Maximum number of events to plot (creates an NxN grid).
        - show_only_invalid: If True, skips plotting events that passed all constraints.
        - randomize: If True, picks a random subset of events instead of the first N.
        """
        # 1. Filter the events based on the user's criteria
        events_to_plot = []

        # Extract items to allow for randomization
        event_items = list(self.events_attrs.items())
        if randomize:
            random.shuffle(event_items)

        for evt_time, evt_data in event_items:

            # Only plot events that actually generated fit data (didn't early-continue)
            if not np.isnan(evt_data.get("tau", np.nan)):
                events_to_plot.append((evt_time, evt_data))

            if len(events_to_plot) >= max_events:
                break

        if not events_to_plot:
            print("No fitted events found matching your criteria.")
            return

        # 2. Dynamically calculate grid dimensions
        cols = int(np.ceil(np.sqrt(len(events_to_plot))))
        rows = int(np.ceil(len(events_to_plot) / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
        axes = axes.flatten()

        # 3. Plotting loop
        for idx, (evt_time, evt_data) in enumerate(events_to_plot):
            ax = axes[idx]

            t_segm = evt_data["t_segm"]
            r_segm = evt_data["r_segm"]
            peak_pos = np.argmax(evt_data["p_segm"])

            # Plot the raw trace snippet
            ax.plot(t_segm, r_segm, label="Raw Trace", color="lightgray", linewidth=2)
            ax.axvline(t_segm[peak_pos], color="gray", linestyle=":", alpha=0.6, label="Peak")

            # Plot the exponential fit on the post-peak segment
            fit_i0 = evt_data["fit_min"]
            fit_pk0 = evt_data["fit_peak"]
            fit_t0 = evt_data["tau"]

            if not np.isnan(fit_t0):
                # Evaluate the fit over the portion of time starting at the peak
                t_fit = t_segm[peak_pos:]

                # Reusing your external exp_decay function
                fit_curve = exp_decay(t_fit, fit_i0, fit_pk0, fit_t0)

                # Color code based on whether it passed your constraints
                line_color = "black"
                ax.plot(t_fit, fit_curve, color=line_color, label="Exp Fit", linewidth=2, alpha=0.8)

            # Formatting
            title_color = "black"
            ax.set_title(
                    f"Event: {evt_time:.3f}s]\n"
                    f"r:{evt_data.get('pearson_r', 0):.4f}, MSE:{evt_data.get('mse_fit', 0):.4f}, tau:{fit_t0:.4f}",
                    color=title_color, fontsize=10
                    )

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            if idx == 0:
                ax.legend(loc="upper right", fontsize=8)

        # 4. Cleanup unused axes in the grid
        for i in range(len(events_to_plot), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        # ---------------------------------------------------------
        # DYNAMIC AUTO-SAVE BLOCK
        # ---------------------------------------------------------
        start_s = int(self.time[0])
        end_s = int(self.time[-1])
        events_num = len(events_to_plot)

        try:
            file_name = self.get_info('file', 'name').replace(".", "_")
            file_parent_base = self.get_info('file', 'parent')

            target_dir = os.path.join(file_parent_base, file_name) + os.sep
            os.makedirs(target_dir, exist_ok=True)

            out_name = f"{target_dir}{file_name}_{self.sweep_index:0>2}_{start_s:0>4}_{end_s:0>4}_inspect_fits_{events_num}.png"
        except Exception as e:
            print(f"Name resolution failed: {e}. Using fallback name.")
            out_name = f"recording_{self.sweep_index:0>2}_{start_s:0>4}_{end_s:0>4}_inspect_fits_{events_num}.png"

        fig.savefig(out_name, dpi=300, bbox_inches='tight')

        # ---------------------------------------------------------
        # ASYNC RAM CLEARING BLOCK
        # ---------------------------------------------------------
        def on_close(event):
            event.canvas.figure.clear()
            plt.close(event.canvas.figure)
            import gc

            gc.collect()

        fig.canvas.mpl_connect('close_event', on_close)
        plt.show(block=False)
        fig.canvas.draw()

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
            self, fit_start: float = 0.001, fit_end: float = 0.01,
            n_limit: float = 1.0, peak_radius: float = 0.0
            ):
        """
        Performs Peak-Scaled Non-Stationary Fluctuation Analysis (ps-NSFA)
        using pre-calculated exponential fits.
        """
        end_pos = vtp(fit_end, self.t_delta)

        avg_resp = np.nanmean(self.get_arr("r_segm"), axis=0)
        avg_time = np.nanmean(self.get_arr("t_segm"), axis=0)

        p_r_p = vtp(peak_radius, self.t_delta)
        p_t_p = np.argmax(avg_resp * self.direction)

        if peak_radius > 0.0:
            mean_peak = np.nanmean(avg_resp[p_t_p - p_r_p: p_t_p + p_r_p + 1])
        else:
            mean_peak = avg_resp[p_t_p]

        mean_resp_diff_pow2 = []

        # Calculate variance around the scaled mean
        for evt_pos in list(self.events_attrs.keys()):
            r_segm = self.events_attrs[evt_pos]["r_segm"]
            amplitude = self.events_attrs[evt_pos]["amplitude"]
            end_time = self.events_attrs[evt_pos]["end_time"]

            if avg_time[p_t_p + end_pos] <= end_time:
                scaled_mean = avg_resp * (amplitude / mean_peak)
                diff = r_segm - scaled_mean
                mean_resp_diff_pow2.append(np.power(diff, 2))

        mean_resp_diff_pow2 = np.array(mean_resp_diff_pow2)
        n_e = len(mean_resp_diff_pow2)

        if n_e == 0:
            print("NSFA aborted: No events remained long enough for the fit window.")
            return

        var_resp = np.sum(mean_resp_diff_pow2, axis=0) / n_e

        start_pos = vtp(fit_start, self.t_delta)
        slice_section = slice(p_t_p + start_pos, p_t_p + end_pos)

        time_section = avg_time[slice_section]
        response_section = avg_resp[slice_section]
        variance_section = var_resp[slice_section]

        # =====================================================================
        # Macroscopic Exponential Fit
        # =====================================================================
        # Perform a single fit on the smooth average trace.
        # This prevents NaN generation in np.log and restores performance.
        fit_i_0, fit_pk0, fit_t0, _ = exp_fit(
                response_section,
                time_section - avg_time[p_t_p + start_pos],
                self.direction
                )

        # Generate iso-amplitude bins
        extra_resp_made = np.linspace(
                fit_i_0 - 0.1,
                np.round(
                        exp_decay((time_section[0] - avg_time[p_t_p + start_pos]), fit_i_0, fit_pk0, fit_t0)
                        ).astype(int),
                np.round(np.abs(response_section[0])).astype(int)
                )

        extra_time_made = fit_t0 * np.log((extra_resp_made - fit_i_0) / fit_pk0) + avg_time[p_t_p + start_pos]

        bins = [(start <= time_section) & (time_section < end) for start, end in pairwise(np.flip(extra_time_made))]
        binned_resp = remove_nan(np.array([np.nanmean(response_section[section]) for section in bins]))
        binned_var = remove_nan(np.array([np.nanmean(variance_section[section]) for section in bins]))

        # Apply noise background limit
        noise_threshold = self.std * self.direction * n_limit
        if self.direction == -1:
            clean_mask = binned_resp < noise_threshold
        else:
            clean_mask = binned_resp > noise_threshold

        binned_resp_clean = binned_resp[clean_mask]
        binned_var_clean = binned_var[clean_mask]

        # Parabolic variance-mean fit
        coefficients = parabolic_fit(binned_resp_clean, binned_var_clean)
        intercept = coefficients[0]
        unitary_current = coefficients[1]
        channel_count = -1 / coefficients[2]

        peak_I = np.min(binned_resp_clean) if self.direction == -1 else np.max(binned_resp_clean)
        p_0 = peak_I / (unitary_current * channel_count)

        # Store clean results
        self.ps_nsfa_values = {
                "intercept"      : intercept,
                "i"              : unitary_current,
                "N"              : channel_count,
                "p_0"            : p_0,
                "binned_current" : binned_resp_clean,
                "binned_variance": binned_var_clean,
                "#events"        : n_e
                }

        print(f"ps-NSFA complete: i={unitary_current:.2f}pA, N={channel_count:.0f}, P_open={p_0:.2f}")

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

        # Applied direction-agnostic logic for the endpoint
        end_curr = np.min(current) if self.direction == -1 else np.max(current)
        artificial_current = np.linspace(
                0.0,
                end_curr,
                np.round(np.abs(end_curr)).astype(int)
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
    def get_auc(self, already_adjusted=True):
        b_amp = 0
        # fs = 1/self.t_delta
        # points = vtp(0.001,self.t_delta)
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
                    # smoothing(evt["r_segm"][start_pos:end_pos], points, sharpness=8, c_type='g', fs=fs)[0] - b_amp
                    evt["r_segm"][start_pos:end_pos] - b_amp
                    )

            evt["r_auc"] = area
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
            max_slope_pos = np.argmax(d1_section)
            # --- DYNAMIC DERIVATIVE CALCULATION ---
            # Start with the 1st derivative
            current_deriv = d1_section
            # Loop to calculate the 2nd, 3rd, 4th, ..., Nth derivative dynamically
            for i in range(2, derivative_order + 1):
                current_deriv = differentiate(current_deriv, 1)
            # current_deriv is now your target derivative (e.g., d4, d5, d10)
            # Calculate threshold using the final target derivative
            threshold_pos = np.argmax(current_deriv[:max_slope_pos])
            ap_threshold = r_section[threshold_pos]
            evt["ap_threshold"] = ap_threshold
            # RAM/SPEED FIX: Replaced python's copy.deepcopy with Numpy's native .copy()
            threshold_segm = reset_array(evt["p_segm"], start_pos + threshold_pos)
            evt["threshold_segm"] = threshold_segm

    @timing
    def get_half_width(self):
        print(f"Not implemented!!! {self}")

    @timing
    def screen_events(
            self,
            max_slope: float = -15000,
            zp_to_pp: float = 0.002,
            max_rise_time: float = 0.010,
            slope_peak_time: float = 0.005,
            min_auc: float = 0.0005,
            min_pearson_r: float = 0.9,
            min_ampl: float = -1.48,
            use_fit: bool = True,
            fit_tau_max: float = 0.002,
            fit_tau_min: float = 0.0005
            ):
        if not self.events_attrs:
            print("Screening cancelled: No raw putative events available to process.")
            return

        dire = self.direction
        initial_event_times = list(self.events_attrs.keys())
        print(f"Starting downstream screening pass on {len(initial_event_times)} putative events...")

        def fmt(key: str, spec: str = ".4f") -> str:
            val = evt.get(key)
            if isinstance(val, (int, float)) and not np.isnan(val):
                return f"{key}={val:{spec}}"
            return f"{key}={val}"

        def check_rule(
                key: str,
                failed: bool,
                reason_: str,
                thresh_name: str,
                thresh_val: float,
                spec: str = ".4f"
                ) -> bool:
            if failed:
                msg = (f" Event at {evt_time:12.4f}[s] rejected: {reason_} "
                       f"{fmt(key, spec)} ({thresh_name}={thresh_val:{spec}}).")
                self._reject_event(evt_time, reason_, msg)
                return True
            return False

        for evt_time in initial_event_times:
            evt = self.events_attrs[evt_time]

            # =====================================================================
            # 1. STRUCTURAL & ZERO-PASS INTEGRITY CONSTRAINTS
            # =====================================================================
            peak_err = evt.get("peak_error_time")
            bad_peak_err = peak_err is None or abs(peak_err) > zp_to_pp
            if check_rule(
                    "peak_error_time", bad_peak_err,
                    "peak_alignment_error", "zp_to_pp", zp_to_pp, ".6f"
                    ):
                continue

            # =====================================================================
            # 2. DERIVATIVE & VELOCITY CONSTRAINTS
            # =====================================================================
            rise_slope = evt.get("rise_slope_val")
            bad_slope = rise_slope is None or (rise_slope <= max_slope if dire == -1 else rise_slope >= max_slope)
            if check_rule(
                    "rise_slope_val", bad_slope,
                    "slope_threshold", "max_slope", max_slope, ".2f"
                    ):
                continue

            decay_slope = evt.get("decay_slope_val")
            bad_slope_ratio = decay_slope is None or decay_slope / rise_slope >= 0.0 or abs(decay_slope) > abs(
                    rise_slope
                    )
            if check_rule(
                    "rise_slope_val", bad_slope_ratio,
                    "wrong_slope_ratio", "max_decay_slope", decay_slope, ".2f"
                    ):
                continue

            # =====================================================================
            # 3. KINETIC & PHENOTYPIC SORTING CONSTRAINTS
            # =====================================================================
            rise_time = evt.get("rise_time_peak")
            bad_rise_time = rise_time is None or rise_time > max_rise_time
            if check_rule(
                    "rise_time_peak", bad_rise_time,
                    "rise_time_limit", "max_rise_time", max_rise_time, ".6f"
                    ):
                continue

            slope_delta = evt.get("slope_peak_delta_time")
            bad_slope_delta = slope_delta is None or slope_delta > slope_peak_time
            if check_rule(
                    "slope_peak_delta_time", bad_slope_delta,
                    "slope_peak_delta", "slope_peak_time", slope_peak_time, ".6f"
                    ):
                continue

            r_auc = evt.get("r_auc")
            bad_auc = r_auc is None or dire * r_auc < dire * min_auc
            if check_rule(
                    "r_auc", bad_auc,
                    "low_auc", "min_auc", min_auc, ".6f"
                    ):
                continue

            # =====================================================================
            # 4. FIT QUALITY CONSTRAINTS
            # =====================================================================
            if use_fit:
                pearson_r = evt.get("pearson_r")
                bad_pearson_r = pearson_r is None or np.isnan(pearson_r) or dire * pearson_r < dire * min_pearson_r
                if check_rule(
                        "pearson_r", bad_pearson_r,
                        "poor_pearson_r", "min_pearson_r", min_pearson_r, ".6f"
                        ):
                    continue

                tau = evt.get("tau")
                bad_tau = tau is None or np.isnan(tau) or not (fit_tau_min < tau < fit_tau_max)
                if check_rule(
                        "tau", bad_tau,
                        "invalid_tau", "fit_tau_max", fit_tau_max, ".4f"
                        ):
                    continue

            # =====================================================================
            # 5. AMPLITUDE CONSTRAINTS
            # =====================================================================
            amp_val = evt.get("amplitude")
            bad_amp = amp_val is None or not (amp_val * dire > min_ampl * dire)
            if check_rule(
                    "amplitude", bad_amp,
                    "amplitude_limit", "min_ampl", min_ampl, ".2f"
                    ):
                continue

        current_rejections = sum(self.rejected_counts.values())
        print("\n" + "=" * 50)
        print(" SCREENING PASS COMPLETE COMPLIANCE SUMMARY")
        print("=" * 50)
        print(f" Putative Events Inputted : {len(initial_event_times)}")
        print(f" Validated Events Retained: {len(self.events_attrs)}")
        print(f" Total Lifetime Rejections: {current_rejections}")
        print("-" * 50)
        print(" Cumulative Breakdown of Rejections:")
        for reason, count in self.rejected_counts.items():
            if count > 0:
                print(f"  • {reason:<25}: {count}")
        print("=" * 50 + "\n")

    @timing
    def burst(self, kernel_length=0.5, min_num_evt=2):
        self.burst_attrs = {}

        # 1. Initialize all events as isolated by default
        for evt in self.events_attrs.values():
            evt["in_burst"] = False
            evt["burst_index"] = None

        t_o_p = self.get_arr("t_o_p")
        r_ifreq = self.get_arr("r_ifreq")

        # 2. SAFETY CHECK: Ensure there are enough events to calculate frequency
        if len(t_o_p) < 2:
            print("Fewer than 2 events detected. Burst analysis mathematically impossible. Skipping.")
            self.ifreq_blocks = np.zeros_like(self.time)
            self.mfreq_blocks = np.zeros_like(self.time)
            return

        # 3. RECONSTRUCT CONTINUOUS TIME & FREQUENCY
        r_ifreq_clean = np.nan_to_num(r_ifreq, nan=0.0, posinf=0.0, neginf=0.0)

        start_time = self.time[0]
        end_time = self.time[-1]
        total_points = int(np.round((end_time - start_time) / self.t_delta)) + 1
        continuous_time = np.linspace(start_time, end_time, total_points)
        continuous_ifreq_full = np.zeros(total_points)

        # Calculate exact index placements for the events on the continuous timeline
        event_indices = np.round((t_o_p - start_time) / self.t_delta).astype(int)
        valid_mask = (event_indices >= 0) & (event_indices < total_points)
        continuous_ifreq_full[event_indices[valid_mask]] = r_ifreq_clean[valid_mask]

        # 4. MEMORY-EFFICIENT SMOOTHING & THRESHOLDING
        n_p = vtp(kernel_length, self.t_delta)
        sharpness = 8
        smoothed_ifreq, kernel_sd = smoothing(continuous_ifreq_full, n_p, sharpness, 'g', 1 / self.t_delta)

        kernel = conv_vector(n_p, 'g', sharpness)
        print(f"<----------------->In self.burst: kernel-SD = {get_real_kernel_sd(kernel, 1 / self.t_delta)}")

        single_pulse = np.zeros(2 * n_p)
        single_pulse[n_p] = r_ifreq_clean[0] if len(r_ifreq_clean) > 0 else 1.0
        min_convolved = fftconvolve(single_pulse, kernel, mode='same')
        min_val = min_convolved[min_convolved > 0.0].max()

        # Save parameters for lightweight plotting in show_burst
        self.burst_continuous_time = continuous_time
        self.burst_continuous_ifreq = continuous_ifreq_full
        self.burst_smoothed_ifreq = smoothed_ifreq
        self.burst_threshold = min_val

        smoothed_ifreq_bool = smoothed_ifreq > min_val
        smoothed_ifreq_norm = np.where(smoothed_ifreq_bool, 1.0, 0.0)

        # 'label' returns the labeled array and the number of features found
        labeled_array, section_count = label(smoothed_ifreq_norm)
        print(f"Number of sections (before): {section_count}")

        # OPTIMIZATION: Get slices for every burst in a single pass
        burst_slices = find_objects(labeled_array)

        # Pre-allocate blocks matched to the original (discontinuous) time array size
        self.ifreq_blocks = np.zeros_like(self.time)
        self.mfreq_blocks = np.zeros_like(self.time)

        valid_burst_count = 0

        # 5. EXTRACT BURST FEATURES
        for burst_slice in burst_slices:
            if burst_slice is None:
                continue

            # sl represents the bounds in the strictly continuous arrays
            sl = burst_slice[0]

            # 1. INSTANTLY slice the data on continuous timeline
            temp_burst_ifreq = continuous_ifreq_full[sl]
            temp_burst_time = continuous_time[sl]
            temp_nonzero_ifreq = temp_burst_ifreq > 0
            temp_events_count = np.sum(temp_nonzero_ifreq)

            if temp_nonzero_ifreq.any() and temp_events_count >= min_num_evt:
                valid_burst_count += 1

                tmp_burst_avg_ifreq = np.average(temp_burst_ifreq[temp_nonzero_ifreq][1:])
                tmp_time_arr = temp_burst_time[temp_nonzero_ifreq]
                tmp_min_time = tmp_time_arr.min()
                tmp_max_time = tmp_time_arr.max()

                # 2. Map continuous bounds back to the discontinuous 'self.time' and 'self.resp' arrays
                orig_time_mask = (self.time >= tmp_min_time) & (self.time <= tmp_max_time)

                self.ifreq_blocks[orig_time_mask] = tmp_burst_avg_ifreq
                self.mfreq_blocks[orig_time_mask] = temp_events_count / (tmp_max_time - tmp_min_time)

                resp_mean = np.average(self.resp[orig_time_mask])

                # 3. Calculate integration safely matching the original events array
                evt_mask = (t_o_p >= tmp_min_time) & (t_o_p <= tmp_max_time)
                integration = calculate_area(
                        t_o_p[evt_mask], r_ifreq_clean[evt_mask] ** 2
                        ) / np.abs(resp_mean + 40)

                # 4. Record global position and attributes
                tmp_min_time_pos = vtp_relative(tmp_min_time, temp_burst_time)
                tmp_burst_pos = sl.start + tmp_min_time_pos

                self.burst_attrs[tmp_burst_pos] = dict(
                        burst_start_time=tmp_min_time,
                        burst_end_time=tmp_max_time,
                        burst_length=(tmp_max_time - tmp_min_time),
                        burst_avg_ifreq=tmp_burst_avg_ifreq,
                        burst_max_ifreq=np.max(temp_burst_ifreq[temp_nonzero_ifreq]),
                        burst_mean_freq=temp_events_count / (tmp_max_time - tmp_min_time),
                        burst_freq_power=tmp_burst_avg_ifreq * (temp_events_count - 1),
                        burst_index=valid_burst_count,
                        burst_depolarization=resp_mean,
                        burst_freq_integration=integration,
                        )

                # ADDITION: Tag the specific events that fall within this burst's timeframe
                for evt in self.events_attrs.values():
                    if tmp_min_time <= evt["t_o_p"] <= tmp_max_time:
                        evt["in_burst"] = True
                        evt["burst_index"] = valid_burst_count

        print(f"Number of sections (after): {valid_burst_count}")
    # def burst(self, kernel_length=0.5, min_num_evt=2):
    #     self.burst_attrs = {}
    #     # 1. Initialize all events as isolated by default
    #     for evt in self.events_attrs.values():
    #         evt["in_burst"] = False
    #         evt["burst_index"] = None
    #     # 2. SAFETY CHECK: Ensure there are enough events to calculate frequency
    #     if len(self.get_arr("t_o_p")) < 2:
    #         print("Fewer than 2 events detected. Burst analysis mathematically impossible. Skipping.")
    #         self.ifreq_blocks = np.zeros_like(self.time)
    #         self.mfreq_blocks = np.zeros_like(self.time)
    #         return
    #     ifreq_timecourse = np.array([self.get_arr("t_o_p"), self.get_arr("r_ifreq")])
    #     ifreq_timecourse[1][0] = np.nanmin(ifreq_timecourse[1])
    #     ifreq_timecourse_full = np.zeros_like(self.time)
    #     # 1. Create a boolean mask where the full time array matches the event times
    #     mask = np.isin(self.time, ifreq_timecourse[0])
    #     # 2. Assign the frequencies into those specific 'True' slots
    #     ifreq_timecourse_full[mask] = ifreq_timecourse[1]
    #     n_p = vtp(kernel_length, self.t_delta)
    #     smoothed_ifreq, kernel_sd = smoothing(ifreq_timecourse_full, n_p, 2, 'g', 1 / self.t_delta)
    #     kernel = conv_vector(n_p, 'g', 2)
    #     print(f"<----------------->In self.burst: kernel-SD = {get_real_kernel_sd(kernel, 1 / self.t_delta)}")
    #     single_pulse = np.zeros(2 * n_p)
    #     single_pulse[n_p] = ifreq_timecourse[1][0]
    #     min_convolved = fftconvolve(single_pulse, kernel, mode='same')
    #     min_val = min_convolved[min_convolved > 0.0].max()
    #     smoothed_ifreq_bool = smoothed_ifreq > min_val
    #     smoothed_ifreq_norm = np.where(smoothed_ifreq_bool, 1.0, 0.0)
    #     smoothed_mfreq_norm = np.copy(smoothed_ifreq_norm)
    #     # 'label' returns the labeled array and the number of features found
    #     labeled_array, section_count = label(smoothed_ifreq_norm)
    #     print(f"Number of sections (before): {section_count}")
    #     # OPTIMIZATION: Get slices for every burst in a single pass
    #     burst_slices = find_objects(labeled_array)
    #     # for burst_index in range(1, section_count + 1):
    #     for burst_index, burst_slice in enumerate(burst_slices, 1):
    #         if burst_slice is None:
    #             continue
    #         # burst_slice is a tuple containing a 1D slice: (slice(start, end),)
    #         sl = burst_slice[0]
    #         # 1. INSTANTLY slice the data. No full-array searching.
    #         temp_burst_ifreq = ifreq_timecourse_full[sl]
    #         temp_burst_time = self.time[sl]
    #         temp_nonzero_ifreq = temp_burst_ifreq > 0
    #         temp_events_count = np.sum(temp_nonzero_ifreq)
    #         # 2. Reset the burst area to 0 directly using the slice
    #         smoothed_ifreq_norm[sl] = 0.0
    #         smoothed_mfreq_norm[sl] = 0.0
    #         if temp_nonzero_ifreq.any() and temp_events_count >= min_num_evt:
    #             tmp_burst_avg_ifreq = np.average(temp_burst_ifreq[temp_nonzero_ifreq][1:])
    #             tmp_time_arr = temp_burst_time[temp_nonzero_ifreq]
    #             tmp_min_time = tmp_time_arr.min()
    #             tmp_max_time = tmp_time_arr.max()
    #             # These are local searches on the tiny sliced array (Fast)
    #             tmp_min_time_pos = vtp_relative(tmp_min_time, temp_burst_time)
    #             tmp_max_time_pos = vtp_relative(tmp_max_time, temp_burst_time)
    #             # tmp_min_time_pos = np.where(temp_burst_time == tmp_min_time)[0][0]
    #             # tmp_max_time_pos = np.where(temp_burst_time == tmp_max_time)[0][0]
    #             # 3. Target the exact sub-slice directly using standard math
    #             target_slice = slice(sl.start + tmp_min_time_pos, sl.start + tmp_max_time_pos)
    #             smoothed_ifreq_norm[target_slice] = tmp_burst_avg_ifreq
    #             smoothed_mfreq_norm[target_slice] = temp_events_count / (tmp_max_time - tmp_min_time)
    #             # 4. INSTANTLY calculate the global position using the slice start index
    #             tmp_burst_pos = sl.start + tmp_min_time_pos
    #             mask = (ifreq_timecourse[0] >= tmp_min_time) & (ifreq_timecourse[0] <= tmp_max_time)
    #             self.burst_attrs[tmp_burst_pos] = dict(
    #                     burst_start_time=tmp_min_time,
    #                     burst_end_time=tmp_max_time,
    #                     burst_length=(tmp_max_time - tmp_min_time),
    #                     burst_avg_ifreq=tmp_burst_avg_ifreq,
    #                     burst_max_ifreq=np.max(temp_burst_ifreq[temp_nonzero_ifreq]),
    #                     burst_mean_freq=temp_events_count / (tmp_max_time - tmp_min_time),
    #                     burst_freq_power=tmp_burst_avg_ifreq * (temp_events_count - 1),
    #                     burst_index=burst_index,
    #                     burst_depolarization=np.average(self.resp[sl]),
    #                     burst_freq_integration=calculate_area(
    #                             ifreq_timecourse[0][mask], ifreq_timecourse[1][mask] ** 2
    #                             ) / np.abs(np.average(self.resp[sl]) + 40),
    #                     )
    #             # ADDITION: Tag the specific events that fall within this burst's timeframe
    #             for evt in self.events_attrs.values():
    #                 if tmp_min_time <= evt["t_o_p"] <= tmp_max_time:
    #                     evt["in_burst"] = True
    #                     evt["burst_index"] = burst_index
    #
    #     self.ifreq_blocks = smoothed_ifreq_norm
    #     self.mfreq_blocks = smoothed_mfreq_norm
    #     # Relabeling to ensure continuous numbering if any bursts were disqualified (< 2 APs)
    #     labeled_array, section_count = label(smoothed_ifreq_norm)
    #     for burst_index, (burst_pos, attrs) in enumerate(self.burst_attrs.items(), 1):
    #         attrs['burst_index'] = burst_index
    #     print(f"Number of sections (after): {section_count}")

    @timing
    def show_burst(self, title="Burst Detection Analysis"):
        """
        Visualizes the instantaneous frequency, the calculated threshold,
        and the resulting burst boxes on two synchronized panels,
        using pre-calculated data from the burst() method.
        """
        # Check for the existence of the arrays saved by the updated burst() method
        if not hasattr(self, 'burst_attrs') or not hasattr(self, 'burst_smoothed_ifreq'):
            print("No burst data found. Run burst() first.")
            return

        # SAFETY CHECK: Ensure there are enough events
        if len(self.get_arr("t_o_p")) < 2:
            print("Fewer than 2 events detected. Cannot visualize burst frequency.")
            return

        # Create two subplots stacked vertically, sharing only the X-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        plt.rcParams.update({'font.size': 8})

        # =========================================================
        # 1. RETRIEVE PRE-CALCULATED DATA
        # =========================================================
        continuous_time = self.burst_continuous_time
        continuous_ifreq_full = self.burst_continuous_ifreq
        smoothed_ifreq = self.burst_smoothed_ifreq
        min_val = self.burst_threshold

        # =========================================================
        # 2. TOP PANEL (ax1): Raw Frequencies & Burst Boxes
        # =========================================================
        # Zero (Dashed Line)
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label="Zero Hz")

        # Raw Instantaneous Frequency (Continuous array trace)
        ax1.plot(continuous_time, continuous_ifreq_full, 'k', alpha=0.5, label="Raw Inst. Freq")

        # Burst Boxes (Outline only)
        burst_labeled = False
        for burst_pos, attrs in self.burst_attrs.items():
            start = attrs['burst_start_time']
            end = attrs['burst_end_time']
            avg_ifreq = attrs['burst_avg_ifreq']

            # Trace the perimeter of the box
            ax1.plot(
                    [start, start, end, end, start],
                    [0, avg_ifreq, avg_ifreq, 0, 0],
                    color='orange',
                    alpha=0.75,
                    linewidth=1.5,
                    label="Detected Burst" if not burst_labeled else None
                    )
            burst_labeled = True

        # =========================================================
        # 3. BOTTOM PANEL (ax2): Smoothed Frequency & Threshold
        # =========================================================
        # Zero (Dashed Line)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label="Zero Hz")

        # Smoothed Instantaneous Frequency (Line)
        ax2.plot(continuous_time, smoothed_ifreq, 'b-', linewidth=1.5, label="Smoothed Freq")

        # Threshold (Dashed Line)
        ax2.axhline(y=min_val, color='r', linestyle='--', linewidth=2, label=f"Threshold ({min_val:.2f} Hz)")

        # =========================================================
        # 4. FORMATTING
        # =========================================================
        sweep_idx = getattr(self, 'sweep_index', 0)

        # Top Panel Formatting
        ax1.set_title(f"Sweep {sweep_idx} - {title} ({len(self.burst_attrs)} bursts detected)")
        ax1.set_ylabel("Raw Freq (Hz)")
        ax1.legend(loc='upper right', framealpha=0.8)

        # Bottom Panel Formatting
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Smoothed Freq (Hz)")
        ax2.legend(loc='upper right', framealpha=0.8)

        plt.tight_layout()

        # =========================================================
        # 5. DYNAMIC AUTO-SAVE BLOCK
        # =========================================================
        start_s = int(self.time[0])
        end_s = int(self.time[-1])
        bursts_num = len(self.burst_attrs)

        try:
            file_name = self.get_info('file', 'name').replace(".", "_")
            file_parent_base = self.get_info('file', 'parent')

            target_dir = os.path.join(file_parent_base, file_name) + os.sep
            os.makedirs(target_dir, exist_ok=True)

            out_name = f"{target_dir}{file_name}_{sweep_idx:0>2}_{start_s:0>4}_{end_s:0>4}_burst_analysis_{bursts_num}.png"
        except Exception as e:
            print(f"Name resolution failed: {e}. Using fallback name.")
            out_name = f"burst_analysis_{sweep_idx:0>2}_{start_s:0>4}_{end_s:0>4}_{bursts_num}.png"

        fig.savefig(out_name, dpi=300, bbox_inches='tight')

        # =========================================================
        # 6. ASYNC RAM CLEARING BLOCK
        # =========================================================
        def on_close(event):
            event.canvas.figure.clear()
            plt.close(event.canvas.figure)
            gc.collect()

        fig.canvas.mpl_connect('close_event', on_close)
        plt.show(block=False)
        fig.canvas.draw()
    # def show_burst(self, kernel_length=0.5, title="Burst Detection Analysis"):
    #     """
    #     Visualizes the instantaneous frequency, the calculated threshold,
    #     and the resulting burst boxes on two synchronized panels.
    #     """
    #     if not hasattr(self, 'burst_attrs') or not self.burst_attrs:
    #         print("No burst data found. Run burst() first.")
    #         return
    #
    #     # SAFETY CHECK: Ensure there are enough events to calculate frequency
    #     if len(self.get_arr("t_o_p")) < 2:
    #         print("Fewer than 2 events detected. Cannot visualize burst frequency.")
    #         return
    #
    #     # Create two subplots stacked vertically, sharing only the X-axis
    #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    #     plt.rcParams.update({'font.size': 8})
    #
    #     # =========================================================
    #     # 1. RECONSTRUCT CONTINUOUS TIME & FREQUENCY
    #     # =========================================================
    #     t_o_p = self.get_arr("t_o_p")
    #     r_ifreq = self.get_arr("r_ifreq")
    #
    #     # Safely handle any NaN calculations on the first event without injecting baseline
    #     r_ifreq_clean = np.nan_to_num(r_ifreq, nan=0.0, posinf=0.0, neginf=0.0)
    #
    #     # Reconstruct a strictly continuous time array to bridge gaps
    #     start_time = self.time[0]
    #     end_time = self.time[-1]
    #
    #     # Calculate total points based on t_delta
    #     total_points = int(np.round((end_time - start_time) / self.t_delta)) + 1
    #     continuous_time = np.linspace(start_time, end_time, total_points)
    #
    #     continuous_ifreq_full = np.zeros(total_points)
    #
    #     # Calculate exact index placements for the events on the continuous timeline
    #     event_indices = np.round((t_o_p - start_time) / self.t_delta).astype(int)
    #
    #     # Safety bounds check
    #     valid_mask = (event_indices >= 0) & (event_indices < total_points)
    #     continuous_ifreq_full[event_indices[valid_mask]] = r_ifreq_clean[valid_mask]
    #
    #     n_p = vtp(kernel_length, self.t_delta)
    #     smoothed_ifreq, kernel_sd = smoothing(continuous_ifreq_full, n_p, 2, 'g', 1 / self.t_delta)
    #
    #     kernel = conv_vector(n_p, 'g', 2)
    #     single_pulse = np.zeros(2 * n_p)
    #     single_pulse[n_p] = r_ifreq_clean[0] if len(r_ifreq_clean) > 0 else 1.0
    #     min_convolved = fftconvolve(single_pulse, kernel, mode='same')
    #     min_val = min_convolved[min_convolved > 0.0].max()
    #
    #     # =========================================================
    #     # 2. TOP PANEL (ax1): Raw Frequencies & Burst Boxes
    #     # =========================================================
    #     # Zero (Dashed Line)
    #     ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label=f"Zero Hz)")
    #     # Raw Instantaneous Frequency (Dots)
    #     ax1.plot(continuous_time, continuous_ifreq_full, 'k', alpha=0.5, label="Raw Inst. Freq")
    #
    #     # Burst Boxes (Outline only)
    #     burst_labeled = False
    #     for burst_pos, attrs in self.burst_attrs.items():
    #         start = attrs['burst_start_time']
    #         end = attrs['burst_end_time']
    #         avg_ifreq = attrs['burst_avg_ifreq']
    #
    #         # Trace the perimeter of the box: bottom-left, top-left, top-right, bottom-right, bottom-left
    #         ax1.plot(
    #                 [start, start, end, end, start],
    #                 [0, avg_ifreq, avg_ifreq, 0, 0],
    #                 color='orange',
    #                 alpha=0.75,
    #                 linewidth=1.5,
    #                 label="Detected Burst" if not burst_labeled else None
    #                 )
    #         burst_labeled = True
    #
    #     # =========================================================
    #     # 3. BOTTOM PANEL (ax2): Smoothed Frequency & Threshold
    #     # =========================================================
    #     # Zero (Dashed Line)
    #     ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label=f"Zero Hz)")
    #     # Smoothed Instantaneous Frequency (Line)
    #     ax2.plot(continuous_time, smoothed_ifreq, 'b', linewidth=1.5, label="Smoothed Freq")
    #     # Threshold (Dashed Line)
    #     ax2.axhline(y=min_val, color='r', linestyle='--', linewidth=2, label=f"Threshold ({min_val:.2f} Hz)")
    #
    #     # =========================================================
    #     # 4. FORMATTING
    #     # =========================================================
    #     sweep_idx = getattr(self, 'sweep_index', 0)
    #
    #     # Top Panel Formatting
    #     ax1.set_title(f"Sweep {sweep_idx} - {title} ({len(self.burst_attrs)} bursts detected)")
    #     ax1.set_ylabel("Raw Freq (Hz)")
    #     ax1.legend(loc='upper right', framealpha=0.8)
    #
    #     # Bottom Panel Formatting
    #     ax2.set_xlabel("Time (s)")
    #     ax2.set_ylabel("Smoothed Freq (Hz)")
    #     ax2.legend(loc='upper right', framealpha=0.8)
    #
    #     plt.tight_layout()
    #
    #     # =========================================================
    #     # 5. DYNAMIC AUTO-SAVE BLOCK
    #     # =========================================================
    #     start_s = int(self.time[0])
    #     end_s = int(self.time[-1])
    #     bursts_num = len(self.burst_attrs)
    #
    #     try:
    #         file_name = self.get_info('file', 'name').replace(".", "_")
    #         file_parent_base = self.get_info('file', 'parent')
    #
    #         target_dir = os.path.join(file_parent_base, file_name) + os.sep
    #         os.makedirs(target_dir, exist_ok=True)
    #
    #         out_name = f"{target_dir}{file_name}_{sweep_idx:0>2}_{start_s:0>4}_{end_s:0>4}_burst_analysis_{bursts_num}.png"
    #     except Exception as e:
    #         print(f"Name resolution failed: {e}. Using fallback name.")
    #         out_name = f"burst_analysis_{sweep_idx:0>2}_{start_s:0>4}_{end_s:0>4}_{bursts_num}.png"
    #
    #     fig.savefig(out_name, dpi=300, bbox_inches='tight')
    #
    #     # =========================================================
    #     # 6. ASYNC RAM CLEARING BLOCK
    #     # =========================================================
    #     def on_close(event):
    #         event.canvas.figure.clear()
    #         plt.close(event.canvas.figure)
    #         gc.collect()
    #
    #     fig.canvas.mpl_connect('close_event', on_close)
    #     plt.show(block=False)
    #     fig.canvas.draw()

    @timing
    def get_correlation(self, start=0, end=1800):
        # 1. Extract the 'Gold Standard' Baseline Template
        # Using : for slicing ensures we get a range of values
        idx_start = int(vtp(start, self.t_delta))
        idx_end = int(vtp(end, self.t_delta))
        # freq_bool = self.ifreq_blocks > 0.0
        # freq_norm = np.where(freq_bool, 1.0, 0.0)
        freq_norm = self.ifreq_blocks
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
            self, title: str = 'Recording Events Comparison',
            show_events: bool = True, adjust: bool = True, smooth_params=()
            ) -> None:
        # Stack subplots vertically and synchronize both X and Y axes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, sharey=True)
        plt.rcParams.update({'font.size': 8})

        # =========================================================
        # BOTTOM SUBPLOT: Event-Subtracted Recording
        # =========================================================
        ax2.axhline(y=0.0, color="k", linestyle='--')

        # 1. Create independent copy FIRST to avoid mutating the original instance
        self_copy = deepcopy(self)

        # 2. Subtract events from self_copy.resp instead of self.resp
        for evt_pos, evt in self.events_attrs.items():
            r_segm = evt["r_segm"]
            p_segm = evt["p_segm"]

            if len(r_segm) > 0:
                idx_p = np.nonzero(p_segm)[0]

                if idx_p.size > 0:
                    peak_idx = int(evt_pos) if isinstance(evt_pos, (int, np.integer)) else vtp_relative(
                            evt_pos, self.time
                            )
                    start_idx = peak_idx - idx_p[0]
                else:
                    t_o_p = evt["t_o_p"]
                    t_segm = evt["t_segm"]
                    x_time = t_segm + t_o_p
                    start_idx = vtp_relative(x_time[0], self.time)

                end_idx = min(start_idx + len(r_segm), len(self_copy.resp))
                slice_len = end_idx - start_idx

                if slice_len > 0 and start_idx >= 0:
                    self_copy.resp[start_idx:end_idx] -= r_segm[:slice_len]

        # if smooth_params:  # This block is to remove the noise and make any remaining undetected event more visible
        #     self_copy.get_smooth(*smooth_params)

        ax2.plot(self.time, self_copy.resp, "k", linewidth=1.2, label="Subtracted Trace")
        ax2.set_title(f"Subtracted Recording ({len(self.events_attrs)} events removed)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.legend(loc='best', framealpha=0.7)

        # =========================================================
        # TOP SUBPLOT: Original Recording with Selected Events
        # =========================================================
        ax1.axhline(y=0.0, color="k", linestyle='--')

        if show_events:
            events_labeled = False
            for evt in self.events_attrs.values():
                t_o_p = evt["t_o_p"]
                t_segm = evt["t_segm"]
                r_segm = evt["r_segm"]
                p_segm = evt["p_segm"]
                s_segm = evt["s_segm"]
                z_segm = evt["z_segm"]
                amplitude = evt["amplitude"]
                b_amp = evt["b_amp"]
                threshold_segm = evt["threshold_segm"]

                x_time = t_segm + t_o_p
                if len(x_time) > 0:
                    ax1.hlines(
                            y=b_amp, xmin=x_time[0], xmax=x_time[-1], colors="k",
                            linewidth=1.5, alpha=0.6, label="Event Baseline" if not events_labeled else None
                            )

                if not adjust:
                    amplitude -= b_amp
                    ax1.plot(t_segm + t_o_p, r_segm, "k", label="Event" if not events_labeled else None)
                else:
                    ax1.plot(t_segm + t_o_p, b_amp + r_segm, "k", label="Event" if not events_labeled else None)

                if threshold_segm is not None:
                    ap_threshold = evt["ap_threshold"]
                    idx_t = np.nonzero(threshold_segm)[0]
                    ax1.plot(
                            (t_segm + t_o_p)[idx_t], (b_amp + threshold_segm * (ap_threshold - b_amp))[idx_t], "bo",
                            label="Threshold" if not events_labeled else None
                            )

                idx_z = np.nonzero(z_segm)[0]
                if idx_z.size > 0:
                    ax1.vlines(
                            x=(t_segm + t_o_p)[idx_z], ymin=b_amp, ymax=(b_amp + z_segm * amplitude / 4)[idx_z],
                            colors="g", label="Zero pass" if not events_labeled else None
                            )

                idx_s = np.nonzero(s_segm)[0]
                if idx_s.size > 0:
                    ax1.vlines(
                            x=(t_segm + t_o_p)[idx_s], ymin=b_amp, ymax=(b_amp + s_segm * amplitude / 2)[idx_s],
                            colors="b", label="Slope" if not events_labeled else None
                            )

                idx_p = np.nonzero(p_segm)[0]
                if idx_p.size > 0:
                    ax1.vlines(
                            x=(t_segm + t_o_p)[idx_p], ymin=b_amp, ymax=(b_amp + p_segm * amplitude)[idx_p],
                            colors="r", linewidth=1, label="Peak" if not events_labeled else None
                            )
                events_labeled = True

        # if hasattr(self, 'ifreq_blocks') and self.ifreq_blocks.size > 0:
        #     ax1.plot(self.time, self.ifreq_blocks, "g", lw=2.0, label="Instant Freq")
        #     ax1.plot(self.time, self.mfreq_blocks, "r:", lw=2.0, label="Mean Freq")
        # if self.ifreq_blocks.size:
        #     idx_blocks = np.nonzero(self.ifreq_blocks)[0]
        #     if idx_blocks.size > 0:
        #         ax1.plot(self.time[idx_blocks], self.ifreq_blocks[idx_blocks], "g", lw=2.0, label="Instant Freq")
        #         ax1.plot(self.time[idx_blocks], self.mfreq_blocks[idx_blocks], "r:", lw=2.0, label="Mean Freq")

        # burst_freq_power = np.array(
        #         [self.get_arr("burst_start_time", "burst"),
        #          self.get_arr("burst_freq_power", "burst")]
        #         )
        # ax1.plot(burst_freq_power[0], burst_freq_power[1], "bo", ms=10.0, alpha=0.5, label="Burst Freq Power")

        ax1.set_title(f"{self.sweep_index=} {title} ({len(self.events_attrs)} events)")
        ax1.set_ylabel("Amplitude")
        ax1.legend(loc='best', framealpha=0.7)

        plt.tight_layout()

        # ---------------------------------------------------------
        # DYNAMIC AUTO-SAVE BLOCK
        # ---------------------------------------------------------
        start_s = int(self.time[0])
        end_s = int(self.time[-1])
        events_num = len(self.events_attrs)

        try:
            file_name = self.get_info('file', 'name').replace(".", "_")
            file_parent_base = self.get_info('file', 'parent')

            target_dir = os.path.join(file_parent_base, file_name) + os.sep
            os.makedirs(target_dir, exist_ok=True)

            out_name = f"{target_dir}{file_name}_{self.sweep_index:0>2}_{start_s:0>4}_{end_s:0>4}_comparison_{events_num}.png"
        except Exception as e:
            print(f"Name resolution failed: {e}. Using fallback name.")
            out_name = f"recording_{self.sweep_index:0>2}_{start_s:0>4}_{end_s:0>4}_comparison_{events_num}.png"

        fig.savefig(out_name, dpi=300, bbox_inches='tight')

        # ---------------------------------------------------------
        # ASYNC RAM CLEARING BLOCK
        # ---------------------------------------------------------

        def on_close(event):
            event.canvas.figure.clear()
            plt.close(event.canvas.figure)
            import gc

            gc.collect()

        fig.canvas.mpl_connect('close_event', on_close)
        plt.show(block=False)
        fig.canvas.draw()

    @timing
    def show_events_aligned(self, title):
        r_segm_arr = self.get_arr("r_segm")
        t_segm_arr = self.get_arr("t_segm")
        events_num = len(r_segm_arr)
        if events_num == 0:
            print("No events to plot.")
            return
        alpha_val = 1.0 / (events_num + 1) + 0.03
        fig, ax = plt.subplots(figsize=(5, 2.5))
        for event in r_segm_arr:
            ax.plot(self.common_time, event, "k", alpha=alpha_val)
        ax.plot(
                np.nanmean(t_segm_arr, axis=0),
                np.nanmean(r_segm_arr, axis=0),
                "r:"
                )
        ax.axhline(y=0.0, color='r', linestyle='dashed')
        ax.set_title(f"{self.sweep_index=} {title} {events_num}.")

        # ---------------------------------------------------------
        # DYNAMIC AUTO-SAVE BLOCK (USING PROPER CLASS METHODS)
        # ---------------------------------------------------------
        start_s = int(self.time[0])
        end_s = int(self.time[-1])

        try:
            # Fetch directly from the instance exactly like get_names()
            file_name = self.get_info('file', 'name').replace(".", "_")
            file_parent_base = self.get_info('file', 'parent')

            # Reconstruct the exact directory structure from setup_workspace()
            target_dir = os.path.join(file_parent_base, file_name) + os.sep
            os.makedirs(target_dir, exist_ok=True)  # Failsafe

            # Build the final string ("aligned" modifier)
            out_name = f"{target_dir}{file_name}_{self.sweep_index:0>2}_{start_s:0>4}_{end_s:0>4}_aligned_{events_num}.png"
        except Exception as e:
            print(f"Name resolution failed: {e}. Using fallback name.")
            out_name = f"recording_{self.sweep_index:0>2}_{start_s:0>4}_{end_s:0>4}_aligned_{events_num}.png"

        fig.savefig(out_name, dpi=300, bbox_inches='tight')

        # ---------------------------------------------------------
        # ASYNC RAM CLEARING BLOCK
        # ---------------------------------------------------------
        def on_close(event):
            event.canvas.figure.clear()
            plt.close(event.canvas.figure)
            import gc

            gc.collect()

        fig.canvas.mpl_connect('close_event', on_close)
        plt.show(block=False)
        fig.canvas.draw()
