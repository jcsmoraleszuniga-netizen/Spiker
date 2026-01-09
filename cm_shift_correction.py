#!/usr/bin/env python
import copy
from typing import Optional, Any
import numpy as np
from lib_event_detection import EvtPro
from lib_utility import remove_shift, make_sections
import lib_gui as gui
from copy import copy as cp_copy

# Constants for every recording
const = dict(
    linear=False,
    baseline_start=154,
    baseline_end=874,
)


def body(ori_inst: EvtPro, section: tuple[int, int]) -> EvtPro:
    start, end = section
    print()
    print(f"{start = } {end = }")
    print()
    rec = cp_copy(ori_inst)  # Instantiation of the recordings
    rec.section(start, end)

    b_s_p = np.where(rec.time == const["baseline_start"])[0][0]
    b_e_p = np.where(rec.time == const["baseline_end"])[0][0]
    base = copy.deepcopy(rec.resp[b_s_p:b_e_p])
    b_time = copy.deepcopy(rec.time[b_s_p:b_e_p])
    # Adjusts the shift in the response. Can be constant (linear=False) or linear (linear=True)
    if const["linear"]:
        rec.time, rec.resp = remove_shift(np.array([b_time, base]), np.array([rec.time, rec.resp]))
    else:
        rec.resp -= np.average(base)

    return rec


def main(ori_inst: EvtPro, start: float = 0, total: float = 1800, interval: float = 600) -> None:
    """
    Corrects the shift in of the recording. It can use a constant value or a linear correction.
    """
    range = ori_inst.time[-1] - ori_inst.time[0]
    updated_const: Optional[dict[str, Any]] = gui.ConstDialog(const, f"Resistance assessment {range = }")
    if updated_const:
        const.update(updated_const)
        print("Constants updated.")
    else:
        print("Constants dialog canceled.")
    rec = cp_copy(ori_inst)
    rec.clean()
    for section in make_sections(start, total, interval):
        rec + body(ori_inst, section)  # body function perform the analysis!!!!!!!
    ori_inst.transfer(rec)
    gui.show_plot(ori_inst, title="Linearly corrected response.")

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
        cProfile.run('main(original)', sort='tottime')
