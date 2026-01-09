#!/usr/bin/env python
from typing import Optional, Any
# import lib_gui as gui
import matplotlib.pyplot as plt
from lib_utility import replace, make_name, manage_settings, file_info
import csv
from copy import copy as cp_copy


# Constants for every recording
const = dict(
    direction=-1,
    baseline_start=371,
    baseline_end=377.224,
    response_end=441,
    linear=False,
)


def body(ori_inst, section):
    start, end = section
    print()
    print(f"{start = } {end = }")
    print()

    # ori, rec = na.make_instances(ori_inst, const["direction"], start, end, 2)
    rec = cp_copy(ori_inst)  # Instantiation of the recordings
    rec.section(start, end)
    rec.direction = const["direction"]
    # rec.get_fft()  # Fourier transform of the recording
    # rec.filter(const["mains_frequency_width"], const["mains_frequency"])  # Filtering of the recording: 1st stage
    # rec.filter(const["frequencies_width"], const["frequencies"])  # Filtering of the recording: 2nd stage

    rec.get_section_area(
        const["baseline_start"],
        const["baseline_end"],
        const["response_end"],
        const["linear"]
    )

    return rec


def main(ori_inst, start=0, total=1800, interval=600):
    file_name: str = ori_inst.get_info('file', 'name')
    file_name = replace(".", "_", file_name)
    print(f"{file_name = }")
    file_parent: str = ori_inst.get_info('file', 'parent')
    print(f"{file_parent = }")
    file_path: str = ori_inst.get_info('file', 'path')
    print(f"{file_path = }")
    script_name: str = file_info(__file__, 'name')
    script_name = replace(".", "_", script_name)  # Dot removal
    print(f"{script_name = }")

    const_file = file_path + script_name + "_const.json"
    # Loads the dictionary from the binary file if exists
    const.update(manage_settings(const_file, const))

    settings_file_name = file_parent + make_name([ file_name, script_name, "settings"])
    print(f"{settings_file_name = }")
    with open(settings_file_name, 'a', newline='') as my_settings:
        wr = csv.writer(my_settings, quoting=csv.QUOTE_ALL)
        wr.writerow([keys for keys in const.keys()])
        wr.writerow([values for values in const.values()])

    section = start, total
    rec = body(ori_inst, section)  # body function perform the analysis!!!!!!!
    print(f"{rec.area = }")  # Results of the analysis
    with open(settings_file_name, 'a', newline='') as my_settings:
        wr = csv.writer(my_settings, quoting=csv.QUOTE_ALL)
        for key in rec.area.keys():
            wr.writerow([key, rec.area[key]])

    # plt.figure()
    # plt.axhline(y=0.0, color="k", linestyle='--')
    # plt.plot(ori_inst.time, ori_inst.resp, "k")
    # plt.legend()
    # plt.title(f"Recording.")
    # plt.show(block=False)


if __name__ == "__main__":
    # For testing purposes
    import sys
    import os
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from lib_utility import save_previous_folder, get_previous_folder, make_name
    from lib_event_detection import EvtPro
    from lib_gui import show_plot

    app = QApplication(sys.argv)
    previous_folder: Optional[str] = get_previous_folder()
    if not previous_folder:
        previous_folder = os.path.expanduser("~")
    file_path, _ = QFileDialog.getOpenFileName(None, "Open ABF File", previous_folder, "ABF Files (*.abf)")
    if file_path:
        save_previous_folder(os.path.dirname(file_path))
        original: EvtPro = EvtPro(file_path, True)
        show_plot(original, title="Select the time of the sections: ")
        bound: int = int(original.time[-1])
        main(original, 0, bound, bound)
