#!/usr/bin/env python
import copy
import gc
from typing import Any
import numpy as np
from lib_event_detection import EvtPro, make_instances, plot_adaptive_threshold, plot_rec, plot_smooth, setup_workspace, \
    teardown_workspace, test_main
from lib_gui import ConstDialog
from lib_utility import (
    auto_save, average_by, event_count, event_fr, make_name, make_sections, save_dict, save_plot,
    )

const_file = ""
# Variables for every recording
const: dict[str, Any] = dict(
        event_type="EPSC",  # AP, EPSP, EPSC, IPSP, IPSC or Calcium
        units="pA",  # Units of the responses
        direction=-1,  # Is the response going in the positive (+1) or negative direction (-1)?
        n_deviations_peak=3.0,  # threshold deviations for peaks
        n_deviations_slope_rise=3.0,  # threshold deviations for derivative peaks
        n_deviations_slope_decay=1.5,  # threshold deviations for derivative peaks

        rec_smoothed_width_rise=0.005,  # seconds # smooths recordings # smaller values result in noisier results
        rec_sharpness_rise=8,  # Acuity of the gaussian kernel
        rec_smoothed_width_decay=0.01,  # seconds # smooths recordings # smaller values result in noisier results
        rec_sharpness_decay=8,  # Acuity of the gaussian kernel

        noise_smooth_frame=0.05,  # seconds, width of the average
        noise_sharpness=8,  # Acuity of the gaussian kernel
        noise_smooth_frame_der=0.3,  # seconds, width of the average
        noise_sharpness_der=8,  # Acuity of the gaussian kernel

        resp_increment=0.5,  # Minimum interval between two consecutive events necessary to calculate STD
        std_increment=10.0,  # Interval to select the minimum STD

        adjust=True,  # Adjust the amplitude of the events with respect to their baseline
        slope_peak_time=0.0075,  # How far the slope and peak must be to be detected. For fast events
        peak_to_peak=0.002,  # Minimal Interval between two consecutive peaks
        shift_time=0.001,  # seconds to find a pulse's artifact
        zero_peak_to_amp_peak=0.005,  # seconds between the amplitude peak and the derivative calculated peak
        max_rise_time=0.00263,  # max time delta allowed for the events
        min_amplitude=0.0001,  # threshold to accept an event
        min_auc=-0.02,  # Minimal area under the curve accepted for the events
        max_slope=-1000000,  # Use this value for slope based selection, begin with 20k then reduce until is right

        baseline_time=0.002,  # seconds
        peak_radius=0.0002,  # seconds
        t_bef=0.015,  # seconds, It must be at least the size of zero_pass_frame
        t_aft=0.03,  # seconds
        alignment='p',  # Type of alignment: 'p' peak, 'z' zero value slope, and 's' max rising slope
        peak_type='p',  # Type of peak assessment: 'p' around the peak, 'z' around zero-value slope
        plot_increment=60,  # Seconds to make an average
        bins=25,  # Number of bins for the histograms
        show_everything=False,
        plot=True,  # Plot and save, if False the function just saves the analysis
        plot_test=True,  # Plot the thresholds and the identified peaks
        change_config=True,  # If the configuration is set, don't ask to change it again

        use_fit=True,
        gaussian_window=0.002,  # time interval for the gaussian kernel
        fit_sharpness=2,  # Acuity of the gaussian kernel
        fit_beg=0.0,
        fit_end=0.015,
        pearson_r_min=-0.75,  # Minimum Pearson's coefficient required for the fit
        fit_tau_min=0.0015,  # Decay constant, in seconds
        fit_tau_max=0.01,  # Decay constant, in seconds
        normal_mse_fit_max=3.0,  # Maximum MSE normalized by the amplitude of the event
        n_limit=1.0,  # Number of STD to accept the asymptotic limit of the exp decay

        use_psnsfa=False,
        psnsfa_fit_start=0.0,
        psnsfa_fit_end=0.012,
        psnsfa_n_limit=1.0,
        # factor=10,
        psnsfa_x0=-10,
        psnsfa_x1=1,
        psnsfa_y0=1,
        psnsfa_y1=10,

        evoked=False,  # different type of analysis depending on time locked responses
        pair_pulse=False,  # activates pair pulse analysis. 2 pathways as default
        pp1_r1=0.75,
        pp1_r2=0.83,
        pp1_artifact=0.002,  # Artifact delta. The time that the stimulation artifact lasts
        pp2_r1=1.75,
        pp2_r2=1.83,
        pp2_artifact=0.002,  # Artifact delta. The time that the stimulation artifact lasts
        search_resp=0.01,  # max interval to search the response. search_resp < pp1_r2 - pp1_r1

        burst_analysis=False,  # Activate burst_analysis?
        kernel_length=1.0,  # Length in seconds
        min_num_evt=2,  # Minimum number of events to consider an event a real burst
        corr_start=300,  # correlation template start
        corr_end=600,  # correlation template end

        reduce=True,
        background=False,  # Is there a background to subtract? can be a value or an array.
        )


def run_test_block(ori_inst, run_const, start, end):
    print("Running analysis block...")
    temp_rec, temp_smooth_rise, temp_der, temp_sder = make_instances(
            ori_inst, run_const["direction"], start, end, 4
            )

    # ---------------------------------------------------------
    # SMOOTHING INSTANCES (RISE & DECAY)
    # ---------------------------------------------------------
    width_rise = run_const["rec_smoothed_width_rise"]
    sharpness_rise = run_const["rec_sharpness_rise"]
    if width_rise:
        temp_smooth_rise.get_smooth(width_rise, sharpness_rise)
    else:
        temp_smooth_rise = copy.copy(temp_rec)

    temp_smooth_decay = copy.deepcopy(temp_rec)
    width_decay = run_const["rec_smoothed_width_decay"]
    sharpness_decay = run_const["rec_sharpness_decay"]
    if width_decay:
        temp_smooth_decay.get_smooth(width_decay, sharpness_decay)

    # Plot unified comparison showing original, rise (red), and decay (blue) smoothings
    if run_const["plot_test"]:
        plot_smooth(temp_rec, temp_smooth_rise, temp_smooth_decay, title="Original versus smoothed recordings.")

    # Compute derivatives
    temp_smooth_rise.get_derv()
    temp_sder.resp = temp_smooth_rise.derivative

    temp_smooth_decay.get_derv()

    temp_rec.get_derv()
    temp_der.resp = temp_rec.derivative

    # Create deepcopy for decay kinetics with inverted direction & decay derivative
    temp_sder_decay = copy.deepcopy(temp_sder)
    temp_sder_decay.resp = temp_smooth_decay.derivative
    temp_sder_decay.direction = -1 * run_const["direction"]

    # ---------------------------------------------------------
    # NOISE & ADAPTIVE THRESHOLD CALCULATIONS
    # ---------------------------------------------------------
    sder_kr_sd = temp_sder.get_pk_noise(
            run_const["noise_smooth_frame_der"],
            run_const["n_deviations_slope_rise"],
            run_const["resp_increment"],
            run_const["std_increment"],
            run_const["noise_sharpness_der"]
            )
    sder_decay_kr_sd = temp_sder_decay.get_pk_noise(
            run_const["noise_smooth_frame_der"],
            run_const["n_deviations_slope_decay"],
            run_const["resp_increment"],
            run_const["std_increment"],
            run_const["noise_sharpness_der"]
            )
    der_kr_sd = temp_der.get_pk_noise(
            run_const["noise_smooth_frame_der"],
            run_const["n_deviations_slope_rise"],
            run_const["resp_increment"],
            run_const["std_increment"],
            run_const["noise_sharpness_der"]
            )
    print(f"\n{sder_kr_sd=} \n{sder_decay_kr_sd=} \n{der_kr_sd=}")

    diff_derv_g_conv_derv = temp_der.resp - temp_der.adaptive_sresp
    diff_sderv_g_conv_sderv = temp_sder.resp - temp_sder.adaptive_sresp

    # Iterative frame calculations for response traces
    start_f = 0.001
    frame_increments = np.linspace(start_f, run_const["noise_smooth_frame"], 3)
    frame_data = []

    for noise_smooth_frame in frame_increments:
        smooth_kr_sd = temp_smooth_rise.get_pk_noise(
                noise_smooth_frame,
                run_const["n_deviations_peak"],
                run_const["resp_increment"],
                run_const["std_increment"],
                run_const["noise_sharpness"]
                )
        rec_kr_sd = temp_rec.get_pk_noise(
                noise_smooth_frame,
                run_const["n_deviations_peak"],
                run_const["resp_increment"],
                run_const["std_increment"],
                run_const["noise_sharpness"]
                )
        print(f'{noise_smooth_frame=} {run_const["noise_sharpness"]=}')
        print(f"{smooth_kr_sd=} \n{rec_kr_sd=} ")

        # Capture state copies before temp_smooth_rise mutates on the next frame iteration
        frame_data.append(
                {
                        "frame"          : noise_smooth_frame,
                        "smooth_kr_sd"   : smooth_kr_sd,
                        "rec_kr_sd"      : rec_kr_sd,
                        "adaptive_sresp" : temp_smooth_rise.adaptive_sresp.copy(),
                        "interpolated_sd": temp_smooth_rise.interpolated_sd.copy(),
                        "diff_resp"      : temp_rec.resp - temp_rec.adaptive_sresp,
                        "diff_sresp"     : temp_smooth_rise.resp - temp_smooth_rise.adaptive_sresp,
                        }
                )

    # Render adaptive threshold figure if enabled
    if run_const["plot_test"]:
        plot_adaptive_threshold(
                temp_rec,
                temp_smooth_rise,
                temp_der,
                temp_sder,
                sder_kr_sd,
                diff_derv_g_conv_derv,
                diff_sderv_g_conv_sderv,
                frame_data,
                run_const
                )

    # Free temporary calculation arrays
    del frame_data, diff_derv_g_conv_derv, diff_sderv_g_conv_sderv
    gc.collect()

    # ---------------------------------------------------------
    # PEAK & KINETICS EXTRACTION (RISE & DECAY)
    # ---------------------------------------------------------
    temp_smooth_rise.get_peaks(run_const["shift_time"])
    temp_rec.peaks = temp_smooth_rise.peaks
    temp_rec.peak_boundaries = temp_smooth_rise.peak_boundaries
    temp_rec.peak_noise = temp_smooth_rise.peak_noise

    # Rise Kinetics
    temp_sder.get_peaks(run_const["shift_time"])
    temp_sder.get_z_pass()
    temp_rec.derivative = temp_smooth_rise.derivative
    temp_rec.der_peak_noise_rise = temp_sder.peak_noise
    temp_rec.der_peaks_rise = temp_sder.peaks
    temp_rec.zero_pass_rise = temp_sder.zero_pass_rise

    # Decay Kinetics
    temp_sder_decay.get_peaks(run_const["shift_time"])
    temp_sder_decay.get_z_pass()
    temp_rec.derivative_decay = temp_smooth_decay.derivative
    temp_rec.der_peak_noise_decay = temp_sder_decay.peak_noise
    temp_rec.der_peaks_decay = temp_sder_decay.peaks
    temp_rec.zero_pass_decay = temp_sder_decay.zero_pass_rise

    # Alignments
    temp_rec.align_slopes_to_zero_crossings()
    temp_rec.align_peaks_to_slopes()

    if run_const["plot_test"]:
        plot_rec(temp_rec, "Testing config", (0.0, 0.0), run_const["max_slope"])

    del temp_smooth_rise, temp_smooth_decay, temp_sder, temp_sder_decay
    return temp_rec


def body(ori_inst: EvtPro, section: tuple[float, float]) -> EvtPro:
    start, end = section
    print(f"\n{start = } {end = }\n")

    # ---------------------------------------------------------
    # 1. INITIAL DETECTION LOOP
    # ---------------------------------------------------------
    while True:
        print("Testing new parameters... (Close the Matplotlib plot to reopen the Tuner!)")
        rec = run_test_block(ori_inst, const, start, end)
        if not const["change_config"]:
            break

        updated_const = ConstDialog(const, f"Tune Params ({start}s to {end}s) - Cancel to Continue")
        if updated_const is not None:
            const.update(updated_const)
            save_dict(const_file, const)
            print("Updated parameters using 'updated_const'.")
        else:
            save_dict(const_file, const)
            print("User finished tuning. Continuing with the rest of the processing...")
            break

    # ---------------------------------------------------------
    # 2. EVENT EXTRACTION & MEASUREMENT
    # ---------------------------------------------------------
    # Select those events that have a faster rise than decay
    rec.get_evt(const["peak_to_peak"], const["t_aft"], const["baseline_time"])
    rec.get_alig(const["alignment"])
    min_amplitude = const["direction"] * rec.std * const["n_deviations_peak"]
    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~Recommended minimum amplitude: {min_amplitude:.3f}")

    rec.get_amplitudes(
            const["baseline_time"],
            const["peak_radius"],
            const["peak_type"],
            const["adjust"]
            )

    if const["use_fit"]:
        rec.fit_events(const["gaussian_window"], const["fit_beg"], const["fit_end"], const["fit_sharpness"])

    rec.get_extended()  # Extension of the events to use a common time interval

    if const["event_type"] == "AP":
        rec.get_threshold()

    rec.get_auc(const["adjust"])
    rec.get_half_width()

    if const["evoked"]:
        rec.identify_evoked(
                const["pp1_r1"], const["pp1_r2"], const["pp1_artifact"],
                const["pp2_r1"], const["pp2_r2"], const["pp2_artifact"],
                const["search_resp"],
                )

    # ---------------------------------------------------------
    # 3. SCREEN EVENTS TUNING LOOP (OPTIMIZED FIX)
    # ---------------------------------------------------------
    # Back up the raw, pristine dictionaries before screening begins.
    # This avoids deepcopying the whole class, dodging the BufferedReader error!
    initial_events_attrs = copy.deepcopy(rec.events_attrs)
    initial_rejected_counts = copy.deepcopy(getattr(rec, 'rejected_counts', {}))

    while True:
        print("Testing screen_events parameters... (Close the plot to reopen the Tuner!)")
        # 2. Run the screening destructively on the restored dictionaries
        rec.screen_events(
                const["max_slope"],
                const["zero_peak_to_amp_peak"],
                const["max_rise_time"],
                const["slope_peak_time"],
                const["min_auc"],
                const["pearson_r_min"],
                const["min_amplitude"],
                const["use_fit"],
                const["fit_tau_max"],
                const["fit_tau_min"],
                )

        # 3. Visualize the remaining events
        if const["plot_test"]:
            if const["use_fit"]:
                rec.inspect_fits(9)
            rec.show_events_aligned(f"Testing screen_events: ")

        # 4. Check if we should stop
        if not const["change_config"]:
            break

        # 5. Open the Tuner for the screening parameters
        updated_const = ConstDialog(const, f"Tune screen_events ({start}s to {end}s) - Cancel to Apply")

        if updated_const is not None:
            const.update(updated_const)
            save_dict(const_file, const)
            print("Updated screen_events parameters.")
        else:
            save_dict(const_file, const)
            print("User finished tuning screen_events. Applying to recording...")
            # Because we ran screen_events directly on 'rec', we just break and proceed.
            break

        # 1. Restore the events to their pristine state at the start of every loop
        rec.events_attrs = copy.deepcopy(initial_events_attrs)
        rec.rejected_counts = copy.deepcopy(initial_rejected_counts)

    # ---------------------------------------------------------
    # 4. FINAL POST-PROCESSING
    # ---------------------------------------------------------
    if const["use_psnsfa"]:
        rec.ps_nsfa(
                const["psnsfa_fit_start"], const["psnsfa_fit_end"],
                const["psnsfa_n_limit"], const["peak_radius"],
                )
        axis_tuple = ((const["psnsfa_x0"], const["psnsfa_x1"]), (const["psnsfa_y0"], const["psnsfa_y1"]))
        rec.plot_ps_nsfa(axis_tuple)
    rec.get_frequencies()
    rec.get_intervals()

    if const["burst_analysis"]:
        rec.find_bursts(const["kernel_length"], const["min_num_evt"])
        # rec.get_correlation(const["corr_start"], const["corr_end"])

    if const["show_everything"]:
        rec.show_burst(const["kernel_length"])
        title = f"From {start:0>4} to {end:0>4}. Detected {const['event_type']}: "
        rec.show_all_events(title, True, const["adjust"], (const["rec_smoothed_width_rise"], const["rec_sharpness_rise"]))
        rec.show_events_aligned(title)

    return rec

def _create_event_dict(build_const: dict) -> dict:
    """Helper to generate a fresh event dictionary with independent zero_arrs."""
    zero_arr = np.array([[0, 0]])

    d = {
        "Number of events": {
                "value": zero_arr.copy(), "parameter": "amplitude",
                "units": "#", "function": event_count},
        "Amplitude": {
                "value": zero_arr.copy(), "parameter": "amplitude",
                "units": build_const["units"], "function": average_by},
        "AUC": {
                "value": zero_arr.copy(), "parameter": "r_auc",
                "units": build_const["units"] + "*s", "function": average_by},
        "Average Frequency": {
                "value": zero_arr.copy(), "parameter": "r_auc",
                "units": "Hz", "function": event_fr},
        "Rise-slope value": {
                "value": zero_arr.copy(), "parameter": "rise_slope_val",
                "units": build_const["units"] + "/s", "function": average_by},
        "Event baseline value": {
                "value": zero_arr.copy(), "parameter": "b_amp",
                "units": build_const["units"], "function": average_by},
    }
    if build_const["event_type"] in ["AP", "EPSP", "EPSC", "IPSP", "IPSC"]:
        d["Instant Frequency"] = {
                "value": zero_arr.copy(), "parameter": "r_ifreq",
                "units": "Hz", "function": average_by}
    if build_const["use_fit"]:
        d.update({
            "Tau of Fit": {
                    "value": zero_arr.copy(), "parameter": "tau",
                    "units": "s", "function": average_by},
            "R of decay": {
                    "value": zero_arr.copy(), "parameter": "pearson_r",
                    "units": "", "function": average_by},
            "MSE fit": {
                    "value": zero_arr.copy(), "parameter": "mse_fit",
                    "units": build_const["units"], "function": average_by},
        })
    if build_const["event_type"] == "AP":
        d["AP threshold"] = {
                "value": zero_arr.copy(), "parameter": "ap_threshold",
                "units": "mV", "function": average_by}
    return d

def build_analysis_dicts(build_const: dict) -> tuple[dict, dict, dict, dict, dict]:
    """Generates fresh analysis dictionaries for a new sweep."""

    # 1. Standard events
    events_analyses = _create_event_dict(build_const)

    # 2. Burst and Isolated Events (Only initialize if burst analysis is active)
    # burst_evt_analyses = _create_event_dict(build_const) if build_const["burst_analysis"] else {}
    isolated_evt_analyses = _create_event_dict(build_const) if build_const["burst_analysis"] else {}

    # 3. Burst block
    zero_arr = np.array([[0, 0]])
    bursts_analyses = {}
    if build_const["burst_analysis"]:
        bursts_analyses = {
                "Events amplitude": {
                        "value": zero_arr.copy(), "parameter": "burst_evt_amp",
                        "units": build_const["units"], "function": average_by},
                "Events AUC": {
                        "value": zero_arr.copy(), "parameter": "burst_evt_auc",
                        "units": build_const["units"] + '*s', "function": average_by},
                "Events rise slope": {
                        "value": zero_arr.copy(), "parameter": "burst_evt_rslope",
                        "units": build_const["units"] + '/s', "function": average_by},
                "Intra Inst. Freq."    : {
                        "value": zero_arr.copy(), "parameter": "burst_avg_ifreq",
                        "units": "Hz", "function": average_by},
                "Intra Max. Freq."      : {
                        "value": zero_arr.copy(), "parameter": "burst_max_ifreq",
                        "units": "Hz", "function": average_by},
                "Intra Mean Freq."     : {
                        "value": zero_arr.copy(), "parameter": "burst_mean_freq",
                        "units": "Hz", "function": average_by},
                "Number of bursts": {
                        "value": zero_arr.copy(), "parameter": "burst_length",
                        "units": "#", "function": event_count},
                "Number of events": {
                        "value": zero_arr.copy(), "parameter": "burst_n_evts",
                        "units": "#", "function": average_by},
                "Inter Length"               : {
                        "value": zero_arr.copy(), "parameter": "burst_length",
                        "units": "s", "function": average_by},
                "Inter Frequency"      : {
                        "value": zero_arr.copy(), "parameter": "burst_length",
                        "units": "Hz", "function": event_fr},
                "Depolarization"       : {
                        "value"   : zero_arr.copy(), "parameter": "burst_depolarization",
                        "units": build_const["units"], "function": average_by},
                "Position integration": {
                        "value"   : zero_arr.copy(), "parameter": "burst_pos_integration",
                        "units": "AU", "function": average_by},
                "Burst skewness": {
                        "value": zero_arr.copy(), "parameter": "burst_skewness",
                        "units": "", "function": average_by},
                "Burst kurtosis": {
                        "value": zero_arr.copy(), "parameter": "burst_kurtosis",
                        "units": "", "function": average_by},
                }

    # 4. Section block
    section_analyses = {}
    if build_const["use_psnsfa"]:
        section_analyses = {
                "Intercept"       : {
                        "value"   : zero_arr.copy(), "parameter": "intercept",
                        "units": build_const["units"] + "²","function": None},
                "Unitary current" : {
                        "value": zero_arr.copy(), "parameter": "i",
                        "units": build_const["units"], "function": None},
                "Channel count"   : {
                        "value": zero_arr.copy(), "parameter": "N",
                        "units": "", "function": None},
                "Open probability": {
                        "value": zero_arr.copy(), "parameter": "p_0",
                        "units": "", "function": None},
                }

    # Notice the updated return signature
    # return events_analyses, burst_evt_analyses, isolated_evt_analyses, bursts_analyses, section_analyses
    return events_analyses, isolated_evt_analyses, bursts_analyses, section_analyses


def main(ori_inst, start: float = 0, total: float = 1800, interval: float = 600) -> None:
    """
    Main analysis function.

    Args:
        ori_inst: The original evt_pro instance.
        start: Start time of the analysis.
        total: Total time of the analysis.
        interval: Interval for sectioning the data.
    """
    # ---------------------------------------------------------
    # STANDARDIZED SETUP BLOCK
    # ---------------------------------------------------------
    global const_file
    # Call the general setup. It returns the file_parent, common_name, and the const_file path
    file_parent, common_name, const_file = setup_workspace(ori_inst, __file__, const)

    # ---------------------------------------------------------
    # SCRIPT-SPECIFIC LOGIC
    # ---------------------------------------------------------
    common_name += [const["event_type"], const["alignment"]]

    if ori_inst.mode == "sweeps":
        sweep_count = ori_inst.sweeps
        if const["background"]:
            sweep_count = sweep_count[:-1]
    else:
        sweep_count = [1]

    for sweep_number, _ in enumerate(sweep_count):
        if ori_inst.mode == "sweeps":
            ori_inst.set_resp(sweep_number)
        # evts_anlss, burst_evt_anlss, isol_evt_anlss, bursts_anlss, section_anlss = build_analysis_dicts(const)
        evts_anlss, isol_evt_anlss, bursts_anlss, section_anlss = build_analysis_dicts(const)

        for section in make_sections(start, total, interval):
            start_s, end_s = section
            print(f"{section = }")

            # body function perform the analysis
            rec = body(ori_inst, section)  # assess the use of the '+' operator
            rec.common_name = common_name
            times_of_peaks: np.ndarray = rec.get_arr("t_o_p")
            times_of_bursts: np.ndarray = rec.get_arr("burst_start_time", "burst")

            if len(rec.events_attrs) > 0:
                print("...At least one event")
                # Specific for events
                events: np.ndarray = np.concatenate(([rec.common_time], rec.get_arr("r_segm")), axis=0).T
                events_name = common_name + [
                        f"{sweep_number:0>2}_{start_s:0>4}_{end_s:0>4}_events_{len(events.T[1:])}"]
                out_name_evn = file_parent + make_name(events_name)
                print(f"{out_name_evn = }")
                auto_save(events, out_name_evn)  # Events saved for every section

                # Specific for events
                for components in evts_anlss.values():  # appending consecutive the values every iteration
                    components["value"] = np.append(
                            components["value"],
                            np.stack((times_of_peaks, rec.get_arr(components["parameter"])), axis=0).T,
                            axis=0
                            )
                if const["burst_analysis"]:

                    # ---- NEW: Append Burst events ----
                    # times_of_burst_evts = rec.get_arr("t_o_p", "burst_evt")
                    # if times_of_burst_evts is not None and times_of_burst_evts.size > 0:
                    #     for components in burst_evt_anlss.values():
                    #         components["value"] = np.append(
                    #                 components["value"],
                    #                 np.stack(
                    #                         (times_of_burst_evts, rec.get_arr(components["parameter"], "burst_evt")),
                    #                         axis=0
                    #                         ).T,
                    #                 axis=0
                    #                 )

                    # ---- NEW: Append Isolated events ----
                    times_of_isolated_evts = rec.get_arr("t_o_p", "isolated_evt")
                    if times_of_isolated_evts is not None and times_of_isolated_evts.size > 0:
                        for components in isol_evt_anlss.values():
                            components["value"] = np.append(
                                    components["value"],
                                    np.stack(
                                            (times_of_isolated_evts,
                                             rec.get_arr(components["parameter"], "isolated_evt")), axis=0
                                            ).T,
                                    axis=0
                                    )

                    # Specific for bursts
                    for components in bursts_anlss.values():  # appending consecutive the values every iteration
                        components["value"] = np.append(
                                components["value"],
                                np.stack(
                                        (
                                                times_of_bursts,
                                                rec.get_arr(components["parameter"], "burst")
                                                ),
                                        axis=0
                                        ).T,
                                axis=0
                                )

                # Specific for sections
                for components in section_anlss.values():
                    components["value"] = np.append(
                            components["value"],
                            np.array(
                                    [
                                            [
                                                    (rec.time[0] + rec.time[-1]) / 2,
                                                    rec.ps_nsfa_values[components["parameter"]]
                                                    ]
                                            ]
                                    ),
                            axis=0
                            )

                # ---------------------------------------------------------
                # CONSOLIDATED EVENT CSV EXPORT (PER SECTION)
                # ---------------------------------------------------------
                # Standard events
                rec.export_attrs_csv(
                        file_parent,
                        common_name,
                        sweep_number,
                        "events_attrs",
                        "events_consolidated",
                        "evt_time"
                        )

                # Bursts
                if const["burst_analysis"] and rec.burst_attrs:
                    rec.export_attrs_csv(
                            file_parent,
                            common_name,
                            sweep_number,
                            "burst_attrs",
                            "bursts_consolidated",
                            "burst_time"
                            )
                # rec.export_events_csv(file_parent, common_name, sweep_number)
            else:
                print("No events detected!!")

            # ---------------------------------------------------------
            # STANDARDIZED TEARDOWN BLOCK
            # ---------------------------------------------------------
            teardown_workspace(rec)

        actual_plot_increment = {"start": start, "end": total, "increment": const["plot_increment"]}
        const["bins"] = int(const["bins"])  # Making sure that "bins" is of integer type

        # Saving & plotting for events
        for analysis_type, components in evts_anlss.items():
            components["value"] = components["value"][1:].T  # removing zero_arr
            save_plot(
                    components["value"],
                    {
                            "file_parent"  : file_parent, "common_name": common_name,
                            "sweep_number" : f"{sweep_number:0>2}", "parameter": components["parameter"],
                            "analysis_type": f"{const['event_type']} - {analysis_type}"  # Labeled specifically
                            },
                    components["units"],
                    actual_plot_increment,
                    const["bins"],
                    components["function"],
                    const["plot"]
                    )

        if const["burst_analysis"]:

            # ---- NEW: Save/Plot Burst events ----
            # for analysis_type, components in burst_evt_anlss.items():
            #     if components["value"].shape[0] > 1:  # Ensure data exists beyond zero_arr
            #         components["value"] = components["value"][1:].T
            #         save_plot(
            #                 components["value"],
            #                 {
            #                         "file_parent"  : file_parent, "common_name": common_name,
            #                         "sweep_number" : f"{sweep_number:0>2}", "parameter": components["parameter"],
            #                         "analysis_type": f"Burst event-{analysis_type}"  # Labeled specifically
            #                         },
            #                 components["units"], actual_plot_increment, const["bins"], components["function"],
            #                 const["plot"]
            #                 )

            # ---- NEW: Save/Plot Isolated events ----
            for analysis_type, components in isol_evt_anlss.items():
                if components["value"].shape[0] > 1:  # Ensure data exists beyond zero_arr
                    components["value"] = components["value"][1:].T
                    save_plot(
                            components["value"],
                            {
                                    "file_parent"  : file_parent, "common_name": common_name,
                                    "sweep_number" : f"{sweep_number:0>2}", "parameter": components["parameter"],
                                    "analysis_type": f"Isol event-{analysis_type}"  # Labeled specifically
                                    },
                            components["units"], actual_plot_increment, const["bins"], components["function"],
                            const["plot"]
                            )

            # Saving & plotting for bursts
            for analysis_type, components in bursts_anlss.items():
                components["value"] = components["value"][1:].T  # removing zero_arr
                save_plot(
                        components["value"],
                        {
                                "file_parent"  : file_parent, "common_name": common_name,
                                "sweep_number" : f"{sweep_number:0>2}", "parameter": components["parameter"],
                                "analysis_type": f"BURSTS-{analysis_type}"  # Labeled specifically
                                },
                        components["units"],
                        actual_plot_increment,
                        const["bins"],
                        components["function"],
                        const["plot"]
                        )

        # Saving & plotting for sections
        for analysis_type, components in section_anlss.items():
            components["value"] = components["value"][1:].T  # removing zero_arr
            save_plot(
                    components["value"],
                    {
                            "file_parent"  : file_parent, "common_name": common_name,
                            "sweep_number" : f"{sweep_number:0>2}", "parameter": components["parameter"],
                            "analysis_type": analysis_type
                            },
                    components["units"],
                    actual_plot_increment,
                    const["bins"],
                    components["function"],
                    const["plot"]
                    )

    # # ---------------------------------------------------------
    # # STANDARDIZED TEARDOWN BLOCK
    # # ---------------------------------------------------------
    # teardown_workspace(rec)

    if const["reduce"]:
        # ---------------------------------------------------------
        # PIPELINE CHAINING: INJECT CONST & TRIGGER DOWNSAMPLING
        # ---------------------------------------------------------
        print("Events analysis complete. Initiating downsampling...")

        # 1. Import the reducer module
        import cm_reducer

        # 2. Inject the shared GUI-updated const values into the reducer's dictionary
        for key in cm_reducer.const.keys():
            if key in const:
                cm_reducer.const[key] = const[key]

        # 3. Call the reducer's main function
        cm_reducer.main(ori_inst, start=start, total=total, interval=interval)


if __name__ == "__main__":
    test_main(main)
