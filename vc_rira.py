#!/usr/bin/env python
import copy
from typing import Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from lib_event_detection import EvtPro
from lib_utility import auto_save, make_name, average_by, make_sections, remove_outlier, get_stats, smoothing, vtp, \
    remove_shift, replace, manage_settings
import csv
import lib_gui as gui
from copy import copy as cp_copy

# Constants for every recording
const = dict(
    # smoothed_width=0.002,  # seconds # smooths recordings # smaller values result in noisier results
    direction=-1,
    holding=-60,  # in mV
    I_vs_V=False,  # Perform I vs V testing. For this, select the time of the baseline pulse
    start_base=2244,  # Only works if I_vs_V is True
    end_base=2262,  # Only works if I_vs_V is True
    end_resp=2283,  # Only works if I_vs_V is True
    beg_ar=0.02,
    end_ar=0.005,
    beg_ir=0.25,
    end_ir=0.75,
    remove_outlier=False,  # False by default
    averaging=60
)


def body(ori_inst, section):
    start, end = section
    print()
    print(f"{start = } {end = }")
    print()

    rec = cp_copy(ori_inst)  # Instantiation of the recordings
    rec.section(start, end)
    rec.find_pulses()
    rec.get_rira(const["beg_ar"], const["end_ar"], const["beg_ir"], const["end_ir"])

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
    # gui.show_plot(ori_inst, title="Original recording.")
    # updated_const: Optional[dict[str, Any]] = gui.ConstDialog(const, "Resistance assessment")
    # if updated_const:
    #     const.update(updated_const)
    #     print("Constants updated.")
    # else:
    #     print("Constants dialog canceled.")

    # Names and routes of the files
    file_name: str = ori_inst.get_info('file', 'name')
    file_name = replace(".", "_", file_name)
    file_parent: str = ori_inst.get_info('file', 'parent')
    print(f"{(out_name_ri := file_parent + make_name([file_name, "Ri"]))}")
    print(f"{(out_name_ra := file_parent + make_name([file_name, "Ra"]))}")
    file_path: str = ori_inst.get_info('file', 'path')
    print(f"{file_path = }")
    # Averages
    print(f"{(out_name_ri_avg := file_parent + make_name([file_name, "Ri_avg"]))}")
    print(f"{(out_name_ra_avg := file_parent + make_name([file_name, "Ra_avg"]))}")
    print(f"{(out_name_rm_avg := file_parent + make_name([file_name, "Rm_avg"]))}")
    # Variables are saved in settings: keep this file if your settings are correct
    script_name = ori_inst.get_info('script', 'name')
    script_name = replace(".", "_", script_name)
    # settings_file_name = file_parent + file_name + script_name + "_settings.csv"

    const_file = file_path + script_name + "_const.json"
    # Loads the dictionary from the binary file if exists
    const.update(manage_settings(const_file, const))

    settings_file_name = file_parent + make_name([file_name, script_name, "settings"])
    with open(settings_file_name, 'w', newline='') as my_settings:
        wr = csv.writer(my_settings, quoting=csv.QUOTE_ALL)
        wr.writerow([keys for keys in const.keys()])
        wr.writerow([values for values in const.values()])

    total_input_res = np.array([[0, 0]])  # default values for axis match
    total_acc_res = np.array([[0, 0]])  # default values for axis match
    for section in make_sections(start, total, interval):
        # body function perform the analysis!!!!!!!
        rec = body(ori_inst, section)  # TODO use the new form with '+' operator instead of append
        input_res = rec.inp_res
        acc_res = rec.acc_res

        if const["remove_outlier"]:  # Identification and Removal of outliers
            input_res = remove_outlier(input_res)
            acc_res = remove_outlier(acc_res)

        ############################### Testing
        if const["I_vs_V"]:
            # TODO make a method called show_all_ivs
            accept_voltage = True
            common_voltage = np.ndarray[[]]
            # sample_currents = []
            basal_currents = []
            resp_currents = []
            plt.figure()
            plt.axhline(y=0, color='r', linestyle='dashed', label="0.0 [pA]")
            for pulse_pos in rec.pul_attrs:
                if const["start_base"] <= rec.pul_attrs[pulse_pos]["t_o_p"] <= const["end_resp"]:
                    current = rec.pul_attrs[pulse_pos]["input_curr"] - rec.pul_attrs[pulse_pos]["base_current"]
                    current = smoothing(current, vtp(0.002, rec.t_delta), 10, 2)
                    # sample_currents.append(current)
                    if accept_voltage:
                        common_voltage = rec.pul_attrs[pulse_pos]["input_volt"] + const["holding"]
                        accept_voltage = False
                    if const["start_base"] <= rec.pul_attrs[pulse_pos]["t_o_p"] <= const["end_base"]:
                        # Baseline
                        # plt.plot(common_voltage, current, "r")
                        basal_currents.append(current)
                    else:
                        # Response
                        # plt.plot(common_voltage, current, "b")
                        resp_currents.append(current)
            basal_currents = np.array(basal_currents)
            basal_current = np.average(basal_currents, axis=0)
            resp_currents = np.array(resp_currents) - basal_current
            plt.plot(common_voltage, basal_current, "b")
            for curr in resp_currents:
                plt.plot(common_voltage, curr, "k")
            plt.title("Testing...")
            plt.xlabel("Voltage [mV]")
            plt.ylabel("Current [pA]")
            plt.legend(loc='upper left')
            plt.show()
            out_name_resp = file_parent + make_name(
                [file_name, script_name, f"{const["start_base"]:>0.0f}", f"{const["end_resp"]:>0.0f}", "i_v_response"]
            )
            print(f"{out_name_resp = }")
            i_v_resp: np.ndarray = np.concatenate(([common_voltage], resp_currents), axis=0).T
            auto_save(i_v_resp, out_name_resp)

            out_name_passive = file_parent + make_name(
                [file_name, script_name, f"{const["start_base"]:>0.0f}", f"{const["end_resp"]:>0.0f}", "i_v_passive"]
            )
            print(f"{out_name_passive = }")
            i_v_passive: np.ndarray = np.concatenate(([common_voltage], [basal_current]), axis=0).T
            # i_v_passive: np.ndarray = np.array(common_voltage, basal_current).T
            auto_save(i_v_passive, out_name_passive)
        ############################### Testing

        total_input_res = np.append(total_input_res, input_res, axis=0)
        total_acc_res = np.append(total_acc_res, acc_res, axis=0)
        with open(settings_file_name, 'a', newline='') as my_settings:
            wr = csv.writer(my_settings, quoting=csv.QUOTE_ALL)
            res_names, res_values = get_stats(input_res.T[1])
            cap_names, cap_values = get_stats(acc_res.T[1])
            wr.writerow(["Input Resistance", f"{section}"])
            wr.writerow(res_names)
            wr.writerow(res_values)
            wr.writerow(["Access Resistance", f"{section}"])
            wr.writerow(cap_names)
            wr.writerow(cap_values)

    total_input_res = total_input_res[1:].T  # removing default values for axis match
    total_acc_res = total_acc_res[1:].T  # removing default values for axis match

    plt.figure()
    plt.plot(total_input_res[0], total_input_res[1], "b", label="Input Resistance")
    plt.plot(total_acc_res[0], total_acc_res[1], "r", label="Access Resistance")

    auto_save(total_input_res.T, out_name_ri)
    auto_save(total_acc_res.T, out_name_ra)

    avg_ires = average_by(total_input_res, const["averaging"])
    avg_ares = average_by(total_acc_res, const["averaging"])
    avg_mres = np.copy(avg_ires)
    avg_mres[1] = avg_ires[1] - avg_ares[1]

    plt.plot(avg_ires[0], avg_ires[1], 'bo', label="Averaged Input Resistance")
    plt.plot(avg_ares[0], avg_ares[1], 'ro', label="Averaged Access Resistance")
    plt.plot(avg_mres[0], avg_mres[1], 'ko', label="Averaged Membrane Resistance")
    plt.legend(loc='upper right')
    plt.show(block=False)

    auto_save(avg_ires.T, out_name_ri_avg)
    auto_save(avg_ares.T, out_name_ra_avg)
    auto_save(avg_mres.T, out_name_rm_avg)

    del rec


if __name__ == "__main__":
    # For testing purposes
    import cProfile
    from PyQt6.QtWidgets import (QFileDialog, QApplication)
    import sys
    import os
    from lib_event_detection import EvtPro
    from lib_utility import get_previous_folder, save_previous_folder

    app = QApplication(sys.argv)
    previous_folder = get_previous_folder()
    if not previous_folder:
        previous_folder = os.path.expanduser("~")
    file_path, _ = QFileDialog.getOpenFileName(None, "Open ABF File", previous_folder, "ABF Files (*.abf)")
    if file_path:
        save_previous_folder(os.path.dirname(file_path))
        original = EvtPro(file_path, True)
        gui.show_plot(original)
        bound: int = int(original.time[-1])
        # main(original, 0, bound, bound)
        cProfile.run('main(original, 0, bound, bound)', sort='tottime')
