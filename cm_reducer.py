import gc
import matplotlib.pyplot as plt
import numpy as np
import lib_gui as gui
from lib_event_detection import EvtPro
from lib_utility import auto_save, make_name, make_sections, replace, file_info
from copy import copy as cp_copy

# Variables for every recording
const = dict(
        direction=-1,
        down_sample=10,
        with_threshold=False,
        noise_smooth_frame=0.1,  # seconds, width of the average
        n_deviations_peak=3.0,  # threshold deviations for peaks
        resp_increment=0.5,
        std_increment=60.0,
        noise_sharpness=4,  # Acuity of the gaussian kernel
        )


def body(ori_inst: EvtPro, section: tuple[int, int]) -> EvtPro:
    start, end = section
    print()
    print(f"{start = } {end = }")
    print()

    rec = cp_copy(ori_inst)  # Instantiation of the recordings
    rec.section(start, end)
    rec.direction = const["direction"]
    print(f"{len(rec.resp) = }")

    if not const["with_threshold"]:
        print(f"Using traditional reduction")
        rec.down_sample(const["down_sample"])
    elif const["with_threshold"]:
        print(f"Using smart reduction")
        rec.get_pk_noise(
                const["noise_smooth_frame"],
                const["n_deviations_peak"],
                const["resp_increment"],
                const["std_increment"],
                const["noise_sharpness"]
                )
        rec.get_o_thresh()
        rec.down_sample(const["down_sample"], const["with_threshold"], rec.o_thresh)
    else:
        raise ValueError("Threshold options must be 'True' or 'False'.")

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
    const.update(gui.manage_settings(const_file, const))

    out_name_ds = file_parent + make_name(
            [
                    file_name,
                    f"{start:>0.0f}",
                    f"{total:>0.0f}",
                    f"{const["down_sample"]:>0.0f}",
                    "downsampled",
                    f"{const["with_threshold"]}"
                    ]
            )
    print(f"{out_name_ds = }")

    rec = cp_copy(ori_inst)
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

    del rec
    # Manually trigger garbage collection
    collected = gc.collect()
    print(f"Garbage collector collected {collected} objects.")
    print("Memory should now be freed (though the OS might not immediately show it).")


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
