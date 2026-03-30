#!/usr/bin/env python
import gc

import matplotlib.pyplot as plt
import numpy as np
from lib_event_detection import EvtPro
from lib_utility import auto_save, get_names, make_name, average_by, make_sections, remove_outlier
import lib_gui as gui
from copy import copy as cp_copy

# Constants for every recording
const = dict(
        # smoothed_width=0.002,  # seconds # smooths recordings # smaller values result in noisier results
        event_type="EPSC",  # AP, EPSP, EPSC, IPSP, IPSC or Calcium
        direction=-1,
        holding=-60,  # in mV
        beg_ar=0.02,
        end_ar=0.005,
        beg_ir=0.25,
        end_ir=0.75,
        remove_outlier=False,  # False by default
        plot_increment=60,
        )


def body(ori_inst, section):
    start, end = section
    print()
    print(f"{start = } {end = }")
    print()

    rec = cp_copy(ori_inst)  # Instantiation of the recordings
    rec.direction = const["direction"]
    rec.section(start, end)
    rec.find_pulses()
    match const["event_type"]:
        case "EPSC" | "IPSC":
            print(f"{20 * ' '}Voltage clamp recording...")
            rec.get_iv(const["beg_ar"], const["end_ar"], const["beg_ir"], const["end_ir"], const["holding"])
        case "EPSP" | "IPSP" | "AP":
            print(f"{20 * ' '}Current clamp recording... Not implemented...")

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

    # total_input_res = np.array([[0, 0]])  # default values for axis match # This is called zero_arr in other scripts
    # total_acc_res = np.array([[0, 0]])  # default values for axis match
    # sweep_number = 1
    for section in make_sections(start, total, interval):
        # body function perform the analysis!!!!!!!
        rec = body(ori_inst, section)

    #     total_input_res = np.append(total_input_res, rec.inp_res, axis=0)
    #
    #     match const["event_type"]:
    #         case "EPSC" | "IPSC":
    #             print(f"{20 * ' '}Voltage clamp recording...")
    #             total_acc_res = np.append(total_acc_res, rec.acc_res, axis=0)
    #         case "EPSP" | "IPSP" | "AP":
    #             print(f"{20 * ' '}Current clamp recording... Not implemented...")
    #
    # # Raw values names
    # out_name_ri = file_parent + make_name(common_name + [sweep_number] + ["Ri"])
    # out_name_ra = file_parent + make_name(common_name + [sweep_number] + ["Ra"])
    # print(f"{out_name_ri = }")
    # print(f"{out_name_ra = }")
    #
    # # Averages names
    # out_name_ri_avg = file_parent + make_name(common_name + [sweep_number] + ["Ri", average_by.__name__])
    # out_name_ra_avg = file_parent + make_name(common_name + [sweep_number] + ["Ra", average_by.__name__])
    # out_name_rm_avg = file_parent + make_name(common_name + [sweep_number] + ["Rm", average_by.__name__])
    # print(f"{out_name_ri_avg = }")
    # print(f"{out_name_ra_avg = }")
    # print(f"{out_name_rm_avg = }")
    #
    # plt.figure()
    # total_input_res = total_input_res[1:].T  # removing default values for axis match
    # plt.plot(total_input_res[0], total_input_res[1], "b", label="Input Resistance")
    #
    # auto_save(total_input_res.T, out_name_ri)
    # actual_plot_increment = {"start": start, "end": total, "increment": const["plot_increment"]}
    #
    # avg_ires = average_by(total_input_res, actual_plot_increment)
    # plt.plot(avg_ires[0], avg_ires[1], 'bo', label="Averaged Input Resistance")
    # auto_save(avg_ires.T, out_name_ri_avg)
    # match const["event_type"]:
    #     case "EPSC" | "IPSC":
    #         total_acc_res = total_acc_res[1:].T  # removing default values for axis match
    #         plt.plot(total_acc_res[0], total_acc_res[1], "r", label="Access Resistance")
    #         auto_save(total_acc_res.T, out_name_ra)
    #         avg_ares = average_by(total_acc_res, actual_plot_increment)
    #         avg_mres = np.copy(avg_ires)
    #         avg_mres[1] = avg_ires[1] - avg_ares[1]
    #         plt.plot(avg_ares[0], avg_ares[1], 'ro', label="Averaged Access Resistance")
    #         plt.plot(avg_mres[0], avg_mres[1], 'ko', label="Averaged Membrane Resistance")
    #         auto_save(avg_ares.T, out_name_ra_avg)
    #         auto_save(avg_mres.T, out_name_rm_avg)
    #     case "EPSP" | "IPSP" | "AP":
    #         print(f"{20 * ' '}Current clamp recording... Not implemented...")
    #
    # plt.legend(loc='upper right')
    # plt.show(block=False)

    del rec

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
        main(original, 0, bound, bound)
