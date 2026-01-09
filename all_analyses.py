#!/usr/bin/env python
# import cProfile
# import tracemalloc
from PyQt6.QtWidgets import QApplication
import sys
from lib_gui import AnalysisSelector
from lib_utility import timing
import cm_ps_events as ce
import vc_rira as rr
import cm_area_peak as cap
import cm_filter as cf
import cm_cleaner as cl
import cm_reducer as cr
import cm_shift_correction as sc


@timing
def main() -> None:
    """
    Main function to run the application.

    Initializes the PyQt application, creates the AnalysisSelector window,
    displays it, and starts the application's event loop.
    """
    app = QApplication(sys.argv)
    programs_dict = {
            "Response filtering"                : cf.main,
            "Shift correction"                  : sc.main,
            "Resistance analysis"               : rr.main,
            "Event detection and analysis"      : ce.main,
            "Remove control pulses"             : cl.main,
            "Down sampling"                     : cr.main,
            "Manual AUC analysis"               : cap.main,
            }
    window: AnalysisSelector = AnalysisSelector("Ephys and Imaging", programs_dict)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # tracemalloc.start()
    # cProfile.run('main()', sort='tottime')

    main()

    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')
    #
    # top = 20
    # print(f"[{top = }]")
    # for stat in top_stats[:top]:
    #     print(stat)
    # tracemalloc.stop()
