#!/usr/bin/env python
import copy
import numpy as np
from lib_event_detection import EvtPro, setup_workspace, test_main
from lib_utility import remove_shift, make_sections
from lib_gui import get_record_from_dialog, show_plot
from copy import copy as cp_copy

const_file = ""

# Constants for every recording
const = dict(
        event_type="EPSC",
        direction=-1,
        linear=False,
        baseline_start=300,
        baseline_end=600,
        pulse_length=0.75,  # to remove control pulses from the estimation
        smart_shift=True,
        percentile_threshold=5,  # percentile threshold
        background=False,  # Is there a background to subtract? can be a value or an array.
        )


# def body(ori_inst: EvtPro, section: tuple[int, int]) -> EvtPro:
#     start, end = section
#     print()
#     print(f"{start = } {end = }")
#     print()
#     rec = cp_copy(ori_inst)  # Instantiation of the recordings
#     rec.section(start, end)
#
#     rec_npul = cp_copy(ori_inst)  # Instantiation of the recordings
#     rec_npul.section(start, end)
#
#     # Now we clean pulses from the baseline segment specifically
#     # (Or use rec_npul.del_pulses if your class handles it without changing length)
#     if const["event_type"] in ["AP", "EPSP", "EPSC", "IPSP", "IPSC"]:
#         rec_npul.find_pulses()
#         rec_npul.del_pulses(const["pulse_length"])
#     else:
#         print(f"Pulse removal omitted.")
#
#     b_s_p = np.where(rec.time == const["baseline_start"])[0][0]
#     b_e_p = np.where(rec.time == const["baseline_end"])[0][0]
#
#     base_resp = copy.deepcopy(rec_npul.resp[b_s_p:b_e_p])
#     base_time = copy.deepcopy(rec_npul.time[b_s_p:b_e_p])
#     single_resp_val = np.average(base_resp)
#
#     if const["smart_shift"]:
#         # ---------------------------------------------------------
#         # THE "SMART" MINIMAL VALUES (Robust)
#         # ---------------------------------------------------------
#         # Extracts the lowest 5% of the data points, ignoring single-point noise artifacts
#
#         # 1. Get the indices that sort the response array from lowest to highest
#         sorted_idx = np.argsort(base_resp)
#
#         # 2. Determine how many points make up the bottom 5% (ensuring at least 1 point is selected)
#         percentile_threshold = const["percentile_threshold"]/100
#         n_points = max(1, int(len(base_resp) * percentile_threshold))
#
#         # 3. Extract the indices of those specific minimal points
#         smart_min_indices = sorted_idx[:n_points]
#
#         # 4. Map those indices back to the original arrays to get your times and values
#         base_time = base_time[smart_min_indices]
#         base_resp = base_resp[smart_min_indices]
#
#         # Optional: If you need a single, rock-solid numerical baseline limit from this subset,
#         # use the median of these minimal values. It ignores outliers beautifully.
#         single_resp_val = np.median(base_resp)
#
#     # Adjusts the shift in the response. Can be constant (linear=False) or linear (linear=True)
#     if const["event_type"] == "Calcium":
#         rec.resp = (rec.resp - single_resp_val)/single_resp_val
#     else:
#         if const["linear"]:
#             rec.time, rec.resp = remove_shift(
#                     np.array([base_time, base_resp]),
#                     np.array([rec.time, rec.resp])
#                     )
#         else:
#             rec.resp -= single_resp_val
#
#     return rec
def body(ori_inst: EvtPro, section: tuple[int, int], bg_inst: EvtPro = None) -> EvtPro:
    start, end = section
    print(f"\n{start = } {end = }\n")

    rec = cp_copy(ori_inst)  # Instantiation of the recordings
    rec.section(start, end)

    rec_npul = cp_copy(ori_inst)  # Instantiation of the recordings
    rec_npul.section(start, end)

    # ---------------------------------------------------------
    # BACKGROUND SUBTRACTION
    # ---------------------------------------------------------
    if bg_inst is not None:
        # Create a sectioned copy of the background to match array lengths
        bg_rec = cp_copy(bg_inst)
        bg_rec.section(start, end)

        # Subtract background from both the working data and the baseline estimation data
        rec.resp = rec.resp - bg_rec.resp
        rec_npul.resp = rec_npul.resp - bg_rec.resp

    # Now we clean pulses from the baseline segment specifically
    if const["event_type"] in ["AP", "EPSP", "EPSC", "IPSP", "IPSC"]:
        rec_npul.find_pulses()
        rec_npul.del_pulses(const["pulse_length"])
    else:
        print("Pulse removal omitted.")

    b_s_p = np.where(rec.time == const["baseline_start"])[0][0]
    b_e_p = np.where(rec.time == const["baseline_end"])[0][0]

    base_resp = copy.deepcopy(rec_npul.resp[b_s_p:b_e_p])
    base_time = copy.deepcopy(rec_npul.time[b_s_p:b_e_p])
    single_resp_val = np.average(base_resp)

    if const["smart_shift"]:
        # THE "SMART" MINIMAL VALUES (Robust)
        sorted_idx = np.argsort(base_resp)
        percentile_threshold = const["percentile_threshold"] / 100
        n_points = max(1, int(len(base_resp) * percentile_threshold))

        smart_min_indices = sorted_idx[:n_points]
        base_time = base_time[smart_min_indices]
        base_resp = base_resp[smart_min_indices]

        single_resp_val = np.median(base_resp)

    # Adjusts the shift in the response.
    if const["event_type"] == "Calcium":
        rec.resp = (rec.resp - single_resp_val) / single_resp_val
    else:
        if const["linear"]:
            rec.time, rec.resp = remove_shift(
                    np.array([base_time, base_resp]),
                    np.array([rec.time, rec.resp])
                    )
        else:
            rec.resp -= single_resp_val

    return rec


# def main(ori_inst: EvtPro, start: float = 0, total: float = 1800, interval: float = 600) -> None:
#     """
#     Corrects the shift in of the recording. It can use a constant value or a linear correction.
#     """
#     # ---------------------------------------------------------
#     # STANDARDIZED SETUP BLOCK
#     # ---------------------------------------------------------
#     global const_file
#     # Call the general setup. It returns the file_parent, common_name, and the const_file path
#     file_parent, common_name, const_file = setup_workspace(ori_inst, __file__, const)
#
#     rec = cp_copy(ori_inst)
#     rec.clean()
#     for section in make_sections(start, total, interval):
#         rec + body(ori_inst, section)  # body function perform the analysis!!!!!!!
#     ori_inst.transfer(rec)
#     # show_plot(ori_inst, title="Linearly corrected response.")
#
#     del rec
# def main(ori_inst: EvtPro, start: float = 0, total: float = 1800, interval: float = 600) -> None:
#     """
#     Corrects the shift in the recording. Handles multiple sweeps/cells using
#     the assignment logic from the events analysis script.
#     """
#     # ---------------------------------------------------------
#     # STANDARDIZED SETUP BLOCK
#     # ---------------------------------------------------------
#     global const_file
#     file_parent, common_name, const_file = setup_workspace(ori_inst, __file__, const)
#
#     # ---------------------------------------------------------
#     # SWEEP PROCESSING LOGIC
#     # ---------------------------------------------------------
#     if ori_inst.mode == "sweeps":
#         sweep_count = ori_inst.sweeps
#         if const["background"]:
#             # separate the last column as the background and subtract it from the other columns
#             print(f"{const["background"]=}")
#     else:
#         sweep_count = [1]
#
#     for sweep_number, _ in enumerate(sweep_count):
#         # Initialize the sweep container (cm_ps_events style)
#         rec = cp_copy(ori_inst)
#         rec.clean()
#
#         for section in make_sections(start, total, interval):
#             # Set active column/sweep before processing the section
#             if ori_inst.mode == "sweeps":
#                 ori_inst.set_resp(sweep_number)
#
#             print(f"Processing sweep {sweep_number}, {section = }")
#
#             # Perform correction and add to the sweep container
#             rec = body(ori_inst, section)
#
#             # Now we pass sweep_number so the changes are saved back to the list
#             ori_inst.transfer(rec, sweep_number)
#
#     show_plot(ori_inst, title="Corrected response.")
#     # Final cleanup of the loop container
#     del rec
def main(ori_inst: EvtPro, start: float = 0, total: float = 1800, interval: float = 600) -> None:
    """
    Corrects the shift in the recording. Handles multiple sweeps/cells using
    the assignment logic from the events analysis script.
    """
    # ---------------------------------------------------------
    # STANDARDIZED SETUP BLOCK
    # ---------------------------------------------------------
    global const_file
    file_parent, common_name, const_file = setup_workspace(ori_inst, __file__, const)

    # ---------------------------------------------------------
    # SWEEP PROCESSING LOGIC
    # ---------------------------------------------------------
    bg_inst = None

    if ori_inst.mode == "sweeps":
        sweep_count = list(ori_inst.sweeps) if isinstance(ori_inst.sweeps, (list, tuple, np.ndarray)) else list(
                range(len(ori_inst.sweeps))
                )

        if const["background"]:
            print(f"Background subtraction enabled. Using last column.")
            # 1. Isolate the background sweep
            bg_inst = cp_copy(ori_inst)
            bg_inst.set_resp(-1)  # Target the final sweep

            # 2. Remove the background sweep from the iteration list
            sweep_count = sweep_count[:-1]
    else:
        sweep_count = [1]

    for sweep_number, _ in enumerate(sweep_count):
        # Initialize the sweep container (cm_ps_events style)
        rec = cp_copy(ori_inst)
        rec.clean()

        for section in make_sections(start, total, interval):
            # Set active column/sweep before processing the section
            if ori_inst.mode == "sweeps":
                ori_inst.set_resp(sweep_number)

            print(f"Processing sweep {sweep_number}, {section = }")

            # Perform correction and pass the isolated background instance
            rec = body(ori_inst, section, bg_inst)

            # Now we pass sweep_number so the changes are saved back to the list
            ori_inst.transfer(rec, sweep_number)

    show_plot(ori_inst, title="Corrected response.")

    # Final cleanup of the loop container
    if 'rec' in locals():
        del rec


if __name__ == "__main__":
    test_main(main)
    # # For testing purposes
    # from PyQt6.QtWidgets import (QFileDialog, QApplication)
    # import sys
    # import os
    # from lib_utility import get_previous_folder, save_previous_folder
    #
    # app = QApplication(sys.argv)
    # previous_folder = get_previous_folder()
    # if not previous_folder:
    #     previous_folder = os.path.expanduser("~")
    # file_path, _ = open_file_dialog(None, previous_folder, "ABF Files (*.abf);; CSV Files (*.csv *.CSV)")
    # if file_path:
    #     save_previous_folder(os.path.dirname(file_path))
    #     original = EvtPro(file_path, True)
    #     show_plot(original)
    #     bound = original.time[-1]
    #     main(original, 0, bound, bound)
