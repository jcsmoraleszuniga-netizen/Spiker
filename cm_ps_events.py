#!/usr/bin/env python
import gc

import matplotlib.pyplot as plt
import numpy as np
import lib_gui as gui
from lib_event_detection import EvtPro, make_instances, plot_rec
from lib_utility import (
    average_by, event_fr, event_aft, get_names, make_sections, make_name, auto_save, save_plot, file_info, replace
    )
from typing import Any

# Variables for every recording
const: dict[str, Any] = dict(
        event_type="EPSC",  # AP, EPSP, EPSC, IPSP, IPSC or Calcium
        units="pA",  # Units of the responses
        direction=-1,  # Is the response going in the positive (+1) or negative direction (-1)?
        evoked=True,  # different type of analysis depending on time locked responses
        pair_pulse=True,  # activates pair pulse analysis. 2 pathways as default
        n_deviations_peak=3.0,  # threshold deviations for peaks
        n_deviations_slope=3.0,  # threshold deviations for derivative peaks

        search_width_peak=0.003,  # seconds, range to look for the maximum amplitude position +- search_width_peak/2
        slope_peak_time=0.0075,  # How far the slope and peak must be to be detected. For fast events
        zero_pass_frame=0.1,  # How far the start and peak must be to be detected. For fast events
        peak_to_peak=0.002,  # Interval between two consecutive peaks
        shift_time=0.001,  # seconds to find a pulse's artifact
        zero_peak_to_amp_peak=0.005,  # seconds between the amplitude peak and the derivative calculated peak
        max_rise_time=0.00263,  # max time delta allowed for the events

        baseline_time=0.002,  # seconds
        peak_radius=0.0002,  # seconds
        min_amplitude=0.0001,  # threshold to accept an event
        adjust=True,  # Adjust the amplitude of the events with respect to their baseline
        max_slope=-1000000,  # Use this value for slope based selection, begin with 20k then reduce until is right
        t_bef=0.015,  # seconds, It must be at least the size of zero_pass_frame
        t_aft=0.03,  # seconds
        alignment='p',  # Type of alignment: 'p' peak, 'z' zero value slope, and 's' max rising slope
        peak_type='p',  # Type of peak assessment: 'p' around the peak, 'z' around zero-value slope

        noise_smooth_frame=0.1,  # seconds, width of the average
        resp_increment=0.5,
        std_increment=10.0,
        noise_sharpness=2,  # Acuity of the gaussian kernel

        smoothed_width=0.002,  # seconds # smooths recordings # smaller values result in noisier results
        rec_repetitions=4,  # Number of smoothing repetitions
        rec_sharpness=4,  # Acuity of the gaussian kernel

        gaussian_window=0.002,  # time interval for the gaussian kernel
        fit_repetitions=4,  # Number of smoothing repetitions
        fit_sharpness=2,  # Acuity of the gaussian kernel
        fit_beg=0.0,
        fit_end=0.015,
        pearson_r_min=0.75,  # Minimum Pearson's coefficient required for the fit
        fit_tau_min=0.0015,  # Decay constant, in seconds
        fit_tau_max=0.01,  # Decay constant, in seconds
        normal_mse_fit_max=3.0,  # Maximum MSE normalized by the amplitude of the event
        n_limit=1.0,  # Number of STD to accept the asymptotic limit of the exp decay

        min_auc=-0.02,  # Minimal area under the curve accepted for the events

        plot_increment=60,  # Seconds to make an average
        bins=25,  # Number of bins for the histograms

        use_fit=True,
        use_psnsfa=False,
        psnsfa_fit_start=0.0,
        psnsfa_fit_end=0.012,
        psnsfa_n_limit=1.0,
        show_everything=False,
        plot=True,  # Plot and save, if False the function just saves the analysis
        debug=False,
        factor=10,
        psnsfa_x0=-10,
        psnsfa_x1=1,
        psnsfa_y0=1,
        psnsfa_y1=10,

        pp1_r1=0.75,
        pp1_r2=0.83,
        pp1_artifact=0.002,  # Artifact delta. The time that the stimulation artifact lasts
        pp2_r1=1.75,
        pp2_r2=1.83,
        pp2_artifact=0.002,  # Artifact delta. The time that the stimulation artifact lasts
        search_resp=0.01,  # max interval to search the response. search_resp < pp1_r2 - pp1_r1

        # TODO use a different baseline, or assess using 10 ms instead.

        )


def body(ori_inst: EvtPro, section: tuple[float, float]) -> tuple[EvtPro, EvtPro]:
    """
    Analyzes a section of the recording.
    Args:
        ori_inst: The original evt_pro instance.
        section: The start and end times of the section.
    Returns:
        A tuple containing the processed recording and derivative evt_pro instances.
    """
    start, end = section
    print()
    print(f"{start = } {end = }")
    print()
    rec, rec_smooth, der = make_instances(ori_inst, const["direction"], start, end, 3)
    if const["debug"]:
        plot_rec(rec, der, "DEBUG basic", (0.0, 0.0), "basic", const["factor"])
    # Producing a smoothed version of the recording
    rec_smooth.get_smooth(
            const["smoothed_width"],
            const["rec_repetitions"],
            const["rec_sharpness"]
            )
    # Assessing peak random amplitude in noise region, for the entire file
    # Important! You must smooth the response before getting peak noise to increase sensitivity to peaks
    rec.get_pk_noise(
            const["noise_smooth_frame"],
            const["n_deviations_peak"],
            const["resp_increment"],
            const["std_increment"],
            const["noise_sharpness"]
            )
    rec_smooth.get_derv()  # Producing a derivative of the recording
    # Using der instance to analyze the derivative of rec
    der.resp = rec_smooth.derivative  # der.resp is the derivative of rec.resp
    # Derivative: Assessing peak random amplitude in noise region, for the entire file
    der.get_pk_noise(
            const["noise_smooth_frame"],
            const["n_deviations_slope"],
            const["resp_increment"],
            const["std_increment"],
            const["noise_sharpness"]
            )
    if const["debug"]:
        plot_rec(rec, der, "DEBUG derivative", (0.0, 0.0), "derivative", const["factor"])
    rec.get_peaks(const["search_width_peak"], const["shift_time"])  # Event detection process
    der.get_peaks(const["search_width_peak"], const["shift_time"])  # Event detection process for derivative
    rec.derivative = der.resp  # der.resp is the derivative of rec.resp
    rec.der_peaks = der.peaks  # Substitution
    der.get_z_pass(const["zero_pass_frame"])  # Detection of responses' rise-start and peak
    rec.zero_pass = der.zero_pass  # Substitution
    if const["debug"]:
        plot_rec(rec, der, "DEBUG full", (0.0, 0.0), "full", const["factor"])
    # Select those events that have a faster rise than decay
    rec.get_evt(
            const["slope_peak_time"], const["max_slope"], const["peak_to_peak"], const["t_bef"], const["t_aft"],
            const["zero_peak_to_amp_peak"], const["baseline_time"], const["max_rise_time"]
            )

    rec.get_alig(const["alignment"])

    min_amplitude = const["direction"] * rec.std * const["n_deviations_peak"]
    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~Recommended minimum amplitude: {min_amplitude:.3f}")
    rec.get_amplitudes(
            const["min_amplitude"],
            const["baseline_time"],
            const["peak_radius"],
            const["peak_type"],
            const["adjust"]
            )
    if const["use_fit"]:
        rec.fit_events(
                const["gaussian_window"],
                const["fit_beg"],
                const["fit_end"],
                const["pearson_r_min"],
                const["fit_repetitions"],
                const["fit_sharpness"],
                const["fit_tau_min"],
                const["fit_tau_max"],
                const["normal_mse_fit_max"],
                const["n_limit"],
                const["min_amplitude"]
                )

    rec.get_extended()  # Extension of the events to use a common time interval
    if const["event_type"] == "AP":
        rec.get_threshold()
    rec.get_auc(const["adjust"], const["min_auc"])
    if const["use_psnsfa"]:
        rec.ps_nsfa(
                const["psnsfa_fit_start"],
                const["psnsfa_fit_end"],
                const["psnsfa_n_limit"],
                const["peak_radius"],
                )
        axis_tuple = ((const["psnsfa_x0"], const["psnsfa_x1"]), (const["psnsfa_y0"], const["psnsfa_y1"]))
        rec.plot_ps_nsfa(axis_tuple)

    rec.identify_evoked(
            const["pp1_r1"], const["pp1_r2"], const["pp1_artifact"],
            const["pp2_r1"], const["pp2_r2"], const["pp2_artifact"],
            const["search_resp"],
            )

    rec.get_frequencies()
    rec.get_intervals()
    rec.get_half_width()

    if const["show_everything"]:
        title = f"From {start:0>4} to {end:0>4}. Detected {const["event_type"]}: "
        rec.show_all_events(title, True, const["adjust"])
        rec.show_events_aligned(title)

    # RAM release
    del rec_smooth
    del der

    return rec


def main(ori_inst: EvtPro, start: float = 0, total: float = 1800, interval: float = 600) -> None:
    """
    Main analysis function.

    Args:
        ori_inst: The original evt_pro instance.
        start: Start time of the analysis.
        total: Total time of the analysis.
        interval: Interval for sectioning the data.
    """
    # Names and routes of the files
    file_name, file_number, file_parent, script_name = get_names(ori_inst, __file__)
    common_name: list = [file_name, script_name]

    const_file = file_parent + make_name(common_name + ["const"], ".json")
    const.update(gui.manage_settings(const_file, const))
    common_name += [const["event_type"], const["alignment"]]
    # "t_bef" must be at least the size of "zero_pass_frame"
    if const["zero_pass_frame"] > const["t_bef"]:
        print(f"Changing {const["t_bef"] = }, because is smaller than {const["zero_pass_frame"] = }")
        const["t_bef"] = const["zero_pass_frame"]
    else:
        print(f"{const["t_bef"] = } is at least the size of {const["zero_pass_frame"] = }")

    if ori_inst.mode == "sweeps":
        sweep_count = ori_inst.sweeps
    else:
        sweep_count = [1]
    # for sweep_number, _ in enumerate(ori_inst.sweeps):
    zero_arr = np.array([[0, 0]])
    for sweep_number, _ in enumerate(sweep_count):
        # zero_arr = np.array([[0, 0]])
        # For single events
        events_analyses = {  # TODO load this dict from a json file
                "Amplitude"                    : {
                        "value"    : zero_arr,
                        "parameter": "amplitude",
                        "units"    : const["units"],
                        "function" : average_by
                        },
                "AUC"                          : {
                        "value"    : zero_arr,
                        "parameter": "r_auc",
                        "units"    : const["units"] + "*s",
                        "function" : average_by
                        },
                "Average Frequency"            : {
                        "value"    : zero_arr,
                        "parameter": "r_auc",
                        "units"    : "Hz",
                        "function" : event_fr
                        },
                "Rise-slope value"             : {
                        "value"    : zero_arr,
                        "parameter": "rise_slope_val",
                        "units"    : const["units"] + "/s",
                        "function" : average_by
                        },
                "Max slope to peak"            : {
                        "value"    : zero_arr,
                        "parameter": "slope_peak_delta",
                        "units"    : "s",
                        "function" : average_by
                        },
                "Event baseline value"         : {
                        "value"    : zero_arr,
                        "parameter": "b_amp",
                        "units"    : const["units"],
                        "function" : average_by
                        },
                "Peak position error"          : {
                        "value"    : zero_arr,
                        "parameter": "peak_error",
                        "units"    : "s",
                        "function" : average_by
                        },
                "Rise time to peak: amplitude" : {
                        "value"    : zero_arr,
                        "parameter": "rise_time_peak",
                        "units"    : "s",
                        "function" : average_by
                        },
                "Rise time to peak: derivative": {
                        "value"    : zero_arr,
                        "parameter": "rise_time_der",
                        "units"    : "s",
                        "function" : average_by
                        },
                "Real end time"                : {
                        "value"    : zero_arr,
                        "parameter": "end_time",
                        "units"    : "s",
                        "function" : average_by
                        },
                }
        # Analysis for fast or high frequency events
        if const["event_type"] in ["AP", "EPSP", "EPSC", "IPSP", "IPSC"]:
            fast_events_dict = {  # TODO load this dict from a json file
                    "Instant Frequency": {
                            "value"    : zero_arr,
                            "parameter": "r_ifreq",
                            "units"    : "Hz",
                            "function" : average_by
                            },
                    "I_count/I_average": {
                            "value"    : zero_arr,
                            "parameter": "r_auc",
                            "units"    : "",
                            "function" : event_aft
                            },
                    }
            events_analyses.update(fast_events_dict)
        if const["use_fit"]:
            fit_dict = {
                    "Tau of Fit": {
                            "value"    : zero_arr,
                            "parameter": "tau",
                            "units"    : "s",
                            "function" : average_by
                            },
                    "R of decay": {
                            "value"    : zero_arr,
                            "parameter": "r_decay",
                            "units"    : "",
                            "function" : average_by
                            },
                    "MSE fit"   : {
                            "value"    : zero_arr,
                            "parameter": "mse_fit",
                            "units"    : const["units"],
                            "function" : average_by
                            },
                    }
            events_analyses.update(fit_dict)
        if const["event_type"] == "AP":
            threshold_dict = {
                    "AP threshold": {
                            "value"    : zero_arr,
                            "parameter": "ap_threshold",
                            "units"    : "mV",
                            "function" : average_by
                            },
                    }
            events_analyses.update(threshold_dict)

        section_analyses = {}
        if const["use_psnsfa"]:
            psnsfa_dict = {
                    "Intercept"       : {
                            "value"    : zero_arr,
                            "parameter": "intercept",
                            "units"    : const["units"] + "²",
                            "function" : None
                            },
                    "Unitary current" : {
                            "value"    : zero_arr,
                            "parameter": "i",
                            "units"    : const["units"],
                            "function" : None
                            },
                    "Channel count"   : {
                            "value"    : zero_arr,
                            "parameter": "N",
                            "units"    : "",
                            "function" : None
                            },
                    "Open probability": {
                            "value"    : zero_arr,
                            "parameter": "p_0",
                            "units"    : "",
                            "function" : None
                            },
                    }
            section_analyses.update(psnsfa_dict)
        for section in make_sections(start, total, interval):
            start_s, end_s = section
            if ori_inst.mode == "sweeps":
                ori_inst.set_resp(sweep_number)
            print(f"{section = }")
            # body function perform the analysis
            rec = body(ori_inst, section)  # assess the use of the '+' operator
            times_of_peaks: np.ndarray = rec.get_arr("t_o_p")
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
                for components in events_analyses.values():  # appending consecutive the values every iteration
                    # print(f"{components["value"].shape = }  {components["value"] = }")
                    # print(f"{len(times_of_peaks) = }  {times_of_peaks = }")
                    # print(f"{len(rec.get_arr(components["parameter"])) = }  {rec.get_arr(components["parameter"]) = }")
                    components["value"] = np.append(
                            components["value"],
                            np.stack((times_of_peaks, rec.get_arr(components["parameter"])), axis=0).T,
                            axis=0
                            )

                # Specific for sections
                for components in section_analyses.values():
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
            else:
                print(f"No events detected!!")

        actual_plot_increment = {"start": start, "end": total, "increment": const["plot_increment"]}

        # Saving & plotting for events
        for analysis_type, components in events_analyses.items():
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

        # Saving & plotting for sections
        for analysis_type, components in section_analyses.items():
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

    del rec  # RAM release
    # Manually trigger garbage collection
    collected = gc.collect()
    print(f"Garbage collector collected {collected} objects.")
    print("Memory should now be freed (though the OS might not immediately show it).")


if __name__ == "__main__":
    # For testing purposes
    from PyQt6.QtWidgets import QApplication
    import sys
    import os
    from lib_utility import get_previous_folder, save_previous_folder

    app = QApplication(sys.argv)
    previous_folder = get_previous_folder()
    if not previous_folder:
        previous_folder = os.path.expanduser("~")
    file_path_out, _ = gui.open_file_dialog(None, previous_folder, "ABF Files (*.abf);; CSV Files (*.csv *.CSV)")
    if file_path_out:
        save_previous_folder(os.path.dirname(file_path_out))
        original = EvtPro(file_path_out, True)
        gui.show_plot(original, title="Select the time of the sections: ")
        bound: int = int(original.time[-1])
        main(original, 0, bound, 600)
