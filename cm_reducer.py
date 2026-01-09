from typing import Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import lib_gui as gui
from lib_event_detection import EvtPro
from lib_utility import auto_save, make_name, make_sections, replace, file_info, manage_settings
from copy import copy as cp_copy

# Variables for every recording
const = dict(
    direction=-1,
    down_sample=10,
    smoothing=False,
    repetitions=10,
    sharpness=2,
)


# def body(ori_inst: EvtPro, section: tuple[int, int], down_sample: int = 10) -> EvtPro:
def body(ori_inst: EvtPro, section: tuple[int, int]) -> EvtPro:
    start, end = section
    print()
    print(f"{start = } {end = }")
    print()

    # ori, rec = make_instances(ori_inst, -1, start, end, 2)
    rec = cp_copy(ori_inst)  # Instantiation of the recordings
    rec.section(start, end)
    print(f"{len(rec.resp) = }")
    if const["smoothing"] and const["down_sample"] > 1:
        # Smoothing process
        print(f"{rec.t_delta = }")
        rec.get_smooth(rec.t_delta * const["down_sample"], const["repetitions"], const["sharpness"])  # TODO fix this

    rec.down_sample(const["down_sample"])
    print(f"{len(rec.resp) = }")

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

    out_name_ds = file_parent + make_name(
        [file_name, f"{start:>0.0f}", f"{total:>0.0f}", f"{const["down_sample"]:>0.0f}", "downsampled"]
    )
    print(f"{out_name_ds = }")

    rec = cp_copy(ori_inst)  # TODO find a better way to do this
    rec.clean()
    for section in make_sections(start, total, interval):
        rec + body(ori_inst, section)

    plt.figure()
    plt.axhline(y=0.0, color="k", linestyle='--')
    plt.plot(ori_inst.time, ori_inst.resp, "k", label="Original")
    plt.plot(rec.time, rec.resp, "r", label="Cleaned", alpha=0.9)
    plt.legend()
    plt.title(f"Original vs Reduced.")
    plt.show(block=False)

    auto_save(np.array([rec.time, rec.resp]).T, out_name_ds)
    ori_inst.transfer(rec)

    del rec


if __name__ == "__main__":
    # For testing purposes
    import sys
    import os
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from lib_utility import save_previous_folder, get_previous_folder
    from lib_event_detection import EvtPro
    from lib_gui import show_plot

    app = QApplication(sys.argv)
    previous_folder = get_previous_folder()
    if not previous_folder:
        previous_folder = os.path.expanduser("~")
    file_path, _ = QFileDialog.getOpenFileName(None, "Open ABF File", previous_folder, "ABF Files (*.abf)")
    if file_path:
        save_previous_folder(os.path.dirname(file_path))
        original = EvtPro(file_path, True)
        show_plot(original, title="Select the time of the sections: ")
        bound = original.time[-1]
        main(original, 0, bound, bound)
