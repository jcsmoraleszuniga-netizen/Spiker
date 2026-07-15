import gc
import os
import matplotlib.pyplot as plt
import numpy as np
import lib_gui as gui
from lib_event_detection import EvtPro, setup_workspace, teardown_workspace
from lib_utility import auto_save, get_names, make_name, make_sections
from copy import copy as cp_copy

const_file = ""

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
        rec.down_sample(int(const["down_sample"]))
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
    # # ---------------------------------------------------------
    # # STANDARDIZED SETUP BLOCK (Copied from Events Analysis)
    # # ---------------------------------------------------------
    global const_file
    # Call the general setup. It returns the file_parent, common_name, and the const_file path
    file_parent, common_name, const_file = setup_workspace(ori_inst, __file__, const)

    # ---------------------------------------------------------
    # SCRIPT-SPECIFIC LOGIC
    # ---------------------------------------------------------
    # Use common_name to build the specific downsampling output name
    out_name_ds = file_parent + make_name(
            common_name + [
                    f"{start:>0.0f}",
                    f"{total:>0.0f}",
                    f"{const['down_sample']:>0.0f}",
                    "downsampled",
                    f"{const['with_threshold']}"
                    ]
            )
    print(f"{out_name_ds = }")

    rec = cp_copy(ori_inst)
    rec.clean()
    for section in make_sections(start, total, interval):
        rec = body(ori_inst, section)

    # Save immediately before the plot pauses the script
    auto_save(np.array([rec.time, rec.resp]).T, out_name_ds)

    # ---------------------------------------------------------
    # MEMORY-SAFE PLOTTING & SCORCHED EARTH TEARDOWN
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 7.5))  # Explicitly grab fig and ax

    ax.axhline(y=0.0, color="k", linestyle='--')
    ax.plot(ori_inst.time, ori_inst.resp, "k", label="Original")
    ax.plot(rec.time, rec.resp, "r", label="Cleaned", alpha=0.9)
    ax.legend()
    ax.set_title("Original vs Reduced.")

    # 1. Show the window without triggering the global block
    plt.show(block=False)
    fig.canvas.draw()

    # 2. Pass 'event' and use 'event.canvas' to avoid closure circular reference
    def on_close(event):
        event.canvas.stop_event_loop()

    cid = fig.canvas.mpl_connect('close_event', on_close)

    # 3. Start Matplotlib's internal loop (Pauses script until window is closed)
    fig.canvas.start_event_loop(timeout=0)

    # 4. Scorched Earth RAM Clearing
    fig.canvas.mpl_disconnect(cid)
    fig.clear()
    plt.close(fig)
    plt.close('all')

    # Explicitly delete the local variables holding the plot objects
    del fig, ax

    # 3. One line to handle all garbage collection
    teardown_workspace(rec)


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
