#!/usr/bin/env python
import gc
import numpy as np
from matplotlib import pyplot as plt
from lib_event_detection import EvtPro, setup_workspace, teardown_workspace, test_main
from lib_utility import auto_save, make_name, make_sections, smoothing, vtp
from copy import copy as cp_copy

const_file = ""

# Constants for every recording
const = dict(
        # smoothed_width=0.002,  # seconds # smooths recordings # smaller values result in noisier results
        event_type="EPSC",  # AP, EPSP, EPSC, IPSP, IPSC or Calcium
        direction=-1,
        holding=-60.0,  # in mV
        E_K=-102.0,
        E_Cl=-75.0,
        E_Na=74.0,
        E_Ca2=124.0,
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
        case "EPSC" | "IPSC" | "i_event":
            print(f"{20 * ' '}Voltage clamp recording...")
            rec.get_iv(const["beg_ar"], const["end_ar"], const["beg_ir"], const["end_ir"], const["holding"])
        case "EPSP" | "IPSP" | "AP" | "v_event":
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
    # ---------------------------------------------------------
    # STANDARDIZED SETUP BLOCK
    # ---------------------------------------------------------
    global const_file
    file_parent, common_name, const_file = setup_workspace(ori_inst, __file__, const)

    # ---------------------------------------------------------
    # SCRIPT-SPECIFIC LOGIC
    # ---------------------------------------------------------
    common_name += [const["event_type"]]

    for section in make_sections(start, total, interval):
        start_s, end_s = section
        print(f"{section = }")

        # Perform the analysis
        rec = body(ori_inst, section)

        if not getattr(rec, "iv_attrs", {}):
            print(f"No IV attributes found for section {section}. Skipping saving/plotting.")
            teardown_workspace(rec)
            continue

        # Reconstruct iv_iter from the stored iv_attrs
        iv_iter = tuple(
                zip(
                        [attrs["iv_time"] for attrs in rec.iv_attrs.values()],
                        [attrs["iv_res"] for attrs in rec.iv_attrs.values()],
                        [attrs["acc_res"] for attrs in rec.iv_attrs.values()]
                        )
                )
        base_resp_0 = iv_iter[0]
        base_resp_1 = iv_iter[1]
        voltage_ires = rec.voltage_ires
        n_p = vtp(0.002, rec.t_delta)
        voltage_axis = voltage_ires + const["holding"]

        # =========================================================
        # 1. PROCESS & SAVE DATA (Chronological Order)
        # =========================================================
        # Initialize save list with the Voltage axis as the first column
        iv_save_list = [voltage_axis]
        avg_base_resp = np.average(base_resp_0[1], base_resp_1[1])
        for pos, curr in enumerate(iv_iter):  # TODO make the average of the first to baseline traces
            if pos:
                raw_diff = curr[1] - base_resp_0[1]
            else:
                raw_diff = base_resp_0[1]
            smoothed_diff = smoothing(raw_diff, n_p, 2)
            iv_save_list.append(smoothed_diff)

            # Stack into a 2D array: [Voltage, Current_1, Current_2, ...] and Save
        iv_save_matrix = np.stack(iv_save_list, axis=0).T
        out_name_iv = file_parent + make_name(common_name + [f"{start_s:0>4}_{end_s:0>4}_IV_curves"])
        auto_save(iv_save_matrix, out_name_iv)

        if const.get("plot", True):
            # =========================================================
            # 2. PLOT I-V GRAPH (Reversed Order for Visual Layering)
            # =========================================================
            fig, ax = plt.subplots()
            i = (len(iv_save_list[1:]) - 1) * 10
            increment = 1 / (len(iv_save_list[1:]) + 1)
            j = increment

            # TODO implement apply_by sectioning to the data I-V
            # actual_plot_increment = {"start": start, "end": total, "increment": const["plot_increment"]}
            # avg_ires = average_by(total_input_res, actual_plot_increment)
            plt.axvline(const["E_K"], color="k", linestyle='--')
            plt.axvline(const["E_Cl"], color="g", linestyle='--')
            plt.axvline(const["E_Na"], color="b", linestyle='--')
            plt.axvline(const["E_Ca2"], color="r", linestyle='--')
            plt.axvline(const["holding"], color="k", linestyle=':')

            for curr in reversed(iv_save_list[1:]):
                plt.plot(voltage_axis, curr, label=f"{i}[s]", linewidth=3.0 * (0.0 + j), alpha=0.0 + j)
                i -= 10
                j += increment

            # Formatting and Showing the I-V Plot
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('Voltage [mV]', loc='right')
            ax.set_ylabel('Current [pA]', loc='bottom', rotation=0)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            plt.legend()
            plt.title(f"I-V Graph (Section: {start_s} - {end_s})")
            plt.show(block=False)

            # =========================================================
            # 3. PLOT PULSE DIFFERENCE (Time Course)
            # =========================================================
            plt.figure()
            plt.axhline(0.0, color="k", linestyle='--')
            plt.plot(rec.time, rec.resp, "r")
            for curr in iv_iter:
                plt.plot(curr[0], base_resp_0[1], "g")
                plt.plot(curr[0], curr[1] - base_resp_0[1], "b")
                plt.plot(curr[0], np.zeros_like(voltage_ires) + base_resp_0[2], "g", linewidth=5.0)
                plt.plot(curr[0], np.zeros_like(voltage_ires) + curr[2], "k", linewidth=3.0)
            plt.title(f"Pulse difference (Section: {start_s} - {end_s})")
            plt.show(block=False)

        # ---------------------------------------------------------
        # STANDARDIZED TEARDOWN BLOCK
        # ---------------------------------------------------------
        teardown_workspace(rec)

    # Manually trigger garbage collection
    collected = gc.collect()
    print(f"Garbage collector collected {collected} objects.")
    print("Memory should now be freed (though the OS might not immediately show it).")


if __name__ == "__main__":
    test_main(main)
