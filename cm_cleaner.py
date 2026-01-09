import matplotlib.pyplot as plt
import lib_gui as gui
from lib_event_detection import EvtPro
from lib_utility import make_name, make_sections, replace, file_info, manage_settings
from typing import Optional, Any
from copy import copy as cp_copy

const = dict(
    direction=-1,
    pulse_length=0.75,
)


# def body(ori_inst: EvtPro, section: tuple[int, int], del_length: float = 1.0) -> EvtPro:
def body(ori_inst: EvtPro, section: tuple[int, int]) -> EvtPro:
    start, end = section
    print()
    print(f"{start = } {end = }")
    print()

    # ori, rec = make_instances(ori_inst, const["direction"], start, end, 2)
    rec = cp_copy(ori_inst)  # Instantiation of the recordings
    rec.section(start, end)
    rec.find_pulses()
    rec.del_pulses(const["pulse_length"])

    return rec


def main(ori_inst: EvtPro, start: int = 0, total: int = 1800, interval: int = 600) -> None:
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
        [file_name, f"{start:>0.0f}", f"{total:>0.0f}", f"{const["pulse_length"] * 1000:>0.0f}ms_cleaned"]
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
    plt.title("Original vs Cleaned.")
    plt.show(block=False)

    ori_inst.transfer(rec)  # Changes are stored in the original object

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
