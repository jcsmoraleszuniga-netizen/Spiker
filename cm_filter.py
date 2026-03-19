import gc

import matplotlib.pyplot as plt
import numpy as np
import lib_gui as gui
from lib_event_detection import EvtPro
from lib_utility import get_names, make_name, make_sections, apply_by
from typing import Any, Optional
from copy import copy as cp_copy

# Variables for every recording
const: dict[str, Any] = dict(
        component='resp',
        mains_frequency=[
                60
                ],  # Hertz
        mains_frequency_width=1.0,  # Hertz
        attenuation_mains=0.1,  # pA
        other_frequencies=[
                864, 2280
                ],  # Hertz
        other_frequencies_width=40,  # Hertz
        attenuation_other=0.1,  # pA
        show_spectrum=True
        )


def body(ori_inst: EvtPro, section: tuple[int, int]) -> EvtPro:
    start, end = section
    print()
    print(f"{start = } {end = }")
    print()

    rec = cp_copy(ori_inst)  # Instantiation of the recordings
    print(f"Inside of body {rec.resp = }  delete me after...")
    rec.section(start, end)
    # Fourier transform of the recording. Original of the filtered response without smoothing
    rec.get_fft(const['component'])
    # Filtering of the recording: 1st stage
    print("Filtering of the recording: 1st stage")
    rec.filter(const["mains_frequency_width"], const["mains_frequency"], const["attenuation_mains"])
    # Filtering of the recording: 2nd stage
    print("Filtering of the recording: 2nd stage")
    rec.filter(const["other_frequencies_width"], const["other_frequencies"], const["attenuation_other"])
    # Reconstruction of the original signal with filtered frequencies
    rec.get_ifft()

    return rec


def main(ori_inst: EvtPro, start: int = 0, total: int = 1800, interval: int = 600):
    # Names and routes of the files
    file_name, file_number, file_parent, script_name = get_names(ori_inst, __file__)
    common_name: list = [file_name, script_name]
    const_file = file_parent + make_name(common_name + ["const"], ".json")
    const.update(gui.manage_settings(const_file, const))

    if const["show_spectrum"]:
        ori_inst.get_fft(const['component'])
        ori_inst.fft_plot("Original")

    rec = cp_copy(ori_inst)
    rec.clean()
    for section in make_sections(start, total, interval):
        rec + body(ori_inst, section)

    last_pos = len(rec.time)
    increment = {"start": rec.time[0], "end": rec.time[-1], "increment": 0.1}
    std_resp = apply_by(
            np.std,
            np.array([rec.time, rec.resp - ori_inst.resp[:last_pos]]),
            increment,
            True
            )

    plt.figure()
    plt.axhline(y=0.0, color="k", linestyle='--')
    plt.plot(ori_inst.time, ori_inst.resp, "k", label="Original", linewidth=2)
    plt.plot(rec.time, rec.resp, "r", label="Filtered")
    factor = 100
    plt.plot(std_resp[0], -1 * std_resp[1] * factor, "b", label=f"STD * {factor}", alpha=0.75)
    plt.legend(loc='lower right')
    plt.title(f"Original vs filtered.")
    plt.show(block=False)

    ori_inst.transfer(rec)  # Changes are stored in the original object

    if const["show_spectrum"]:
        ori_inst.get_fft(const['component'])
        ori_inst.fft_plot("Filtered")

    del rec
    del std_resp  # Also delete std_resp if it's no longer needed

    # Manually trigger garbage collection
    collected = gc.collect()
    print(f"Garbage collector collected {collected} objects.")
    print("Memory should now be freed (though the OS might not immediately show it).")


if __name__ == "__main__":
    # For testing purposes
    # import sys
    # import os
    # from PyQt6.QtWidgets import QApplication, QFileDialog
    # from lib_utility import save_previous_folder, get_previous_folder
    # from lib_event_detection import EvtPro
    # from lib_gui import show_plot
    #
    # app = QApplication(sys.argv)
    # previous_folder: Optional[str] = get_previous_folder()
    # if not previous_folder:
    #     previous_folder = os.path.expanduser("~")
    # file_path_test, _ = QFileDialog.getOpenFileName(None, "Open ABF File", previous_folder, "ABF Files (*.abf)")
    # if file_path_test:
    #     save_previous_folder(os.path.dirname(file_path_test))
    #     original: EvtPro = EvtPro(file_path_test, True)
    #     show_plot(original, title="Select the time of the sections: ")
    #     bound: int = int(original.time[-1])
    #     main(original, 0, bound, bound)
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
