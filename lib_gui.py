import gc
import os

from PyQt6.QtCore import QSettings, Qt, pyqtSignal

from lib_utility import get_previous_folder, load_dict, save_dict, save_previous_folder
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QApplication, QDoubleSpinBox, QHBoxLayout, QSlider, QSpinBox, QWidget, QDialog, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QMessageBox, QGridLayout, QListWidget, QListWidgetItem,
    QInputDialog, QCheckBox, QFileDialog, QScrollArea
    )


def open_file_dialog(
        parent,
        previous_folder: str,
        custom_filter: str = "All Files (*.*);; ABF Files (*.abf);; CSV Files (*.csv *.CSV);; JSON Files (*.json)"
        ) -> str:
    return QFileDialog.getOpenFileName(
            parent,
            "Open File",
            previous_folder,
            custom_filter
            )


def get_from_dialog(dialog_class):
    """Decorator to extract the result from dialog objects"""

    def wrapper(*args):
        dialog = dialog_class(*args)  # Instantiation
        if dialog.exec() == QDialog.DialogCode.Accepted:  # exec() is modal.

            return dialog.result
        else:
            return None

    return wrapper


_current_plot = None  # Global variable to store the current plot figure


def get_record_from_dialog(parent=None, location=1, custom_filter="ABF Files (*.abf);; CSV Files (*.csv *.CSV)"):
    """
    Handles the UI logic for file selection and object initialization.
    """
    import sys
    import os
    from PyQt6.QtWidgets import QApplication
    from lib_utility import get_previous_folder, save_previous_folder
    from lib_event_detection import EvtPro

    app = QApplication.instance() or QApplication(sys.argv)

    # Use the parent's title for folder history if available
    title = getattr(parent, 'title', None)
    previous_folder = get_previous_folder(title) or os.path.expanduser("~")

    file_path, _ = open_file_dialog(parent, previous_folder, custom_filter)

    if not file_path:
        return None, None

    save_previous_folder(os.path.dirname(file_path), title)

    # Return both the object and the path (so the UI can show the filename)
    return EvtPro(file_path, True, location), file_path


def show_plot(original, title="No Title.", values=(0.0, 0.0)):
    global _current_plot

    if _current_plot:
        _current_plot.clf()  # 1. Wipe the data arrays from the figure
        plt.close(_current_plot)  # 2. Close the Matplotlib window
        _current_plot = None  # 3. Kill the global reference so GC can run

    fig = plt.figure()
    match original.mode:
        case "continuous":
            plt.plot(original.time, original.resp, linewidth=0.5)
            plt.plot(original.time, original.cdac, "r")
        case "sweeps":
            for resp in reversed(original.sweeps):
                plt.plot(original.time, resp, linewidth=0.5)
    plt.axhline(y=0.0, color="k", linestyle='--')
    for x_value in values:
        plt.axvline(x=x_value, color="r", linestyle='--')
    plt.title(title)
    plt.show(block=False)

    _current_plot = fig  # Store the current figure


class AnalysisSelector(QWidget):

    def __init__(self, title, programs_dict):
        super().__init__()
        self.location = 0
        self.original = None
        self.title = title
        self.programs_dict = programs_dict
        self.checkboxes = []

        # --- NEW: Initialize the location selector ---
        self.location_spinbox = QSpinBox()
        self.location_spinbox.setMinimum(0)  # Starts from 0
        self.location_spinbox.setValue(0)  # Default value
        # Optional: self.location_spinbox.setMaximum(99) # Set a max if needed

        self.select_file_button = QPushButton("Select ABF File")
        self.select_file_button.clicked.connect(self.select_file)
        self.run_analysis_button = QPushButton("Run analysis")
        self.run_analysis_button.clicked.connect(self.run_analyses)
        self.status_label = QLabel("")
        self.init_ui()
        # self.select_file()

    def init_ui(self):
        self.setWindowTitle(self.title)

        layout = QVBoxLayout()

        for checkbox_text, program in self.programs_dict.items():
            print(f"{checkbox_text = }")
            self.checkboxes.append(QCheckBox(checkbox_text))

        for checkbox in self.checkboxes:
            layout.addWidget(checkbox)

        # --- NEW: Add the spinbox to the layout ---
        # Using a horizontal layout just for the label and spinbox makes it look cleaner
        location_layout = QHBoxLayout()
        location_layout.addWidget(QLabel("Location (Channel):"))
        location_layout.addWidget(self.location_spinbox)
        layout.addLayout(location_layout)

        # Add buttons and status label
        layout.addWidget(self.select_file_button)
        layout.addWidget(self.run_analysis_button)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def select_file(self):
        global _current_plot  # Bring in the global tracker

        # --- SCORCHED EARTH MEMORY CLEARING BLOCK ---
        # 1. Destroy the global Matplotlib figure keeping the data alive
        if _current_plot:
            _current_plot.clf()
            plt.close(_current_plot)
            _current_plot = None

        plt.close('all')

        # 2. Drop the explicit object reference
        self.original = None

        # 3. Force garbage collection NOW, before the new file loads
        import gc

        gc.collect()
        # ---------------------------------------------

        try:
            # Get the current location value from the UI
            current_loc = self.location_spinbox.value()

            # Call the helper function (now in the same file)
            new_record, file_path = get_record_from_dialog(parent=self, location=current_loc)

            if new_record:
                import os

                # Update class attributes
                self.original = new_record
                self.location = current_loc

                # Show the plot (show_plot is already in this file)
                show_plot(self.original, title="Total response.")

                # Update UI elements
                print(f"{self.original = }")
                self.status_label.setText(f"Loaded: {os.path.basename(file_path)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading file: {e}")
            self.status_label.setText(f"Error loading file: {e}")
    # def select_file(self):
    #     global _current_plot  # Bring in the global tracker
    #
    #     # --- MEMORY CLEARING BLOCK ---
    #     # 1. Destroy the global Matplotlib figure keeping the data alive
    #     if _current_plot:
    #         _current_plot.clf()
    #         plt.close(_current_plot)
    #         _current_plot = None
    #
    #     plt.close('all')
    #
    #     # 2. Drop the explicit object reference
    #     self.original = None
    #
    #     # 3. Force garbage collection NOW, before the new file loads
    #     import gc
    #
    #     gc.collect()
    #     # -----------------------------
    #
    #     previous_folder = get_previous_folder(self.title)
    #     if not previous_folder:
    #         previous_folder = os.path.expanduser("~")
    #
    #     try:
    #         file_path, _ = open_file_dialog(self, previous_folder, "ABF Files (*.abf);; CSV Files (*.csv *.CSV)")
    #         if file_path:
    #             save_previous_folder(os.path.dirname(file_path), self.title)
    #
    #             self.location = self.location_spinbox.value()
    #             from lib_event_detection import EvtPro
    #             # Load the NEW file
    #             self.original = EvtPro(file_path, True, self.location)
    #             show_plot(self.original, title="Total response.")
    #             print(f"{self.original = }")
    #             self.status_label.setText(f"Loaded: {os.path.basename(file_path)}")
    #
    #     except Exception as e:
    #         QMessageBox.critical(self, "Error", f"Error loading file: {e}")
    #         self.status_label.setText(f"Error loading file: {e}")

    def run_analyses(self):
        # --- MEMORY CLEARING BLOCK ---
        # Close all currently open analysis windows
        plt.close('all')
        # Force the garbage collector to destroy the old plot arrays
        gc.collect()
        # -----------------------------
        # if len(self.original):
        if self.original:
            self.status_label.setText("Running analyses...")
            try:
                analyses = dict(zip(self.checkboxes, self.programs_dict.values()))
                for checkbox, func in analyses.items():
                    if checkbox.isChecked():
                        with_sections(func, BoundariesDialog, self.original)
                # Run garbage collection one more time after all analyses finish
                # in case temporary arrays were left behind by the math functions
                gc.collect()
                self.status_label.setText("Analyses complete.")
            except Exception as e:
                self.status_label.setText(f"Error: {e}")
        else:
            self.status_label.setText("No file selected.")


def with_sections(main_func, section_callable, original):
    print(f"\n{main_func.__module__ = } {main_func.__name__ = }")
    print(f"{original.time[0] = } {original.time[-1] = }")
    sections = section_callable(original.time[0], original.time[-1], main_func.__module__)
    main_func(original, *sections)
    gc.collect()


@get_from_dialog
class BoundariesDialog(QDialog):

    def __init__(self, start=0.0, end=0.0, title="", parent=None):
        super().__init__(parent)

        # Persistence setup: Unique key per dataset title
        self.settings_key = f"Boundaries/{title}"
        self.settings = QSettings("MyLabApp", "BoundarySettings")

        self.start = start
        self.end = end
        self.title = title
        self.result = None

        # --- Define UI Elements ---
        self.start_input = QLineEdit()
        self.end_input = QLineEdit()
        self.interval_input = QLineEdit()

        # Checkboxes for min/max shortcuts
        self.min_start_cb = QCheckBox("Min")
        self.max_end_cb = QCheckBox("Max")
        self.max_interval_cb = QCheckBox("Max")

        self.default_checkbox = QCheckBox("Use default values (Reset all)")

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.get_values)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        # Build UI and connect interactivity
        self.init_ui()
        self.connect_logic()

        # Load values from the last time this specific title was opened
        self.load_remembered_values()

    def init_ui(self):
        self.setWindowTitle(f"{self.title} boundaries: {self.start:.3f}s to {self.end:.3f}s")
        self.setMinimumWidth(400)
        layout = QGridLayout()

        # Row 0: Start Time
        layout.addWidget(QLabel("Start Time:"), 0, 0)
        layout.addWidget(self.start_input, 0, 1)
        layout.addWidget(self.min_start_cb, 0, 2)

        # Row 1: End Time
        layout.addWidget(QLabel("End Time:"), 1, 0)
        layout.addWidget(self.end_input, 1, 1)
        layout.addWidget(self.max_end_cb, 1, 2)

        # Row 2: Interval Size
        layout.addWidget(QLabel("Interval Size:"), 2, 0)
        layout.addWidget(self.interval_input, 2, 1)
        layout.addWidget(self.max_interval_cb, 2, 2)

        # Row 3: Global Reset
        layout.addWidget(self.default_checkbox, 3, 0, 1, 3)

        # Row 4: Buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout, 4, 0, 1, 3)

        self.setLayout(layout)

    def connect_logic(self):
        """Connects UI signals to automate value filling and field locking."""
        # Start logic: When Min is checked, force value to self.start
        self.min_start_cb.toggled.connect(
                lambda checked: self._apply_auto_logic(self.start_input, self.start, checked)
                )

        # End logic: When Max is checked, force value to self.end
        self.max_end_cb.toggled.connect(
                lambda checked: self._apply_auto_logic(self.end_input, self.end, checked)
                )

        # Interval logic: Depends on the current text in Start and End boxes
        self.max_interval_cb.toggled.connect(self.update_max_interval)
        self.start_input.textChanged.connect(self.update_max_interval)
        self.end_input.textChanged.connect(self.update_max_interval)

    def _apply_auto_logic(self, line_edit, value, is_checked):
        """Helper to lock/unlock and set text values."""
        if is_checked:
            line_edit.setText(f"{value:.6f}")
            line_edit.setReadOnly(True)
            # Light gray background to indicate 'Locked'
            line_edit.setStyleSheet("background-color: #e9e9e9; color: #444;")
        else:
            line_edit.setReadOnly(False)
            line_edit.setStyleSheet("")

    def update_max_interval(self):
        """Calculates (End - Start) dynamically if the Max checkbox is active."""
        if self.max_interval_cb.isChecked():
            try:
                s = float(self.start_input.text())
                e = float(self.end_input.text())
                diff = max(0.0, e - s)
                self.interval_input.setText(f"{diff:.6f}")
                self.interval_input.setReadOnly(True)
                self.interval_input.setStyleSheet("background-color: #e9e9e9; color: #444;")
            except ValueError:
                # If fields are currently empty or invalid during typing, just wait
                self.interval_input.setText("0.000000")
        else:
            self.interval_input.setReadOnly(False)
            self.interval_input.setStyleSheet("")

    def load_remembered_values(self):
        """Retrieves last used values from disk or uses defaults if none exist."""
        last_start = self.settings.value(f"{self.settings_key}/start", str(self.start))
        last_end = self.settings.value(f"{self.settings_key}/end", str(self.end))
        last_interval = self.settings.value(f"{self.settings_key}/interval", str(self.end - self.start))

        self.start_input.setText(str(last_start))
        self.end_input.setText(str(last_end))
        self.interval_input.setText(str(last_interval))

    def save_current_values(self, start, end, interval):
        """Commits the current validated values to persistent storage."""
        self.settings.setValue(f"{self.settings_key}/start", start)
        self.settings.setValue(f"{self.settings_key}/end", end)
        self.settings.setValue(f"{self.settings_key}/interval", interval)

    def get_values(self):
        """Validates inputs, saves them to settings, and accepts the dialog."""
        if self.default_checkbox.isChecked():
            start, end, interval = self.start, self.end, (self.end - self.start)
            self.result = (start, end, interval)
            self.save_current_values(start, end, interval)
            self.accept()
            return

        try:
            # 1. Parse current inputs
            start = float(self.start_input.text())
            end = float(self.end_input.text())
            interval = float(self.interval_input.text())

            # 2. Hard Bounds Check
            if end > self.end:
                end = self.end
                print(f"Clamped 'end' to recording max: {self.end}")

            # 3. Logic Validation
            if not (0.0 <= self.start <= start < end):
                QMessageBox.critical(
                    self, "Invalid Range",
                    f"Start ({start}) must be >= 0 and less than End ({end})."
                    )
                return

            if interval <= 0.0:
                QMessageBox.critical(self, "Invalid Interval", "Interval must be greater than 0.")
                return

            # Ensure interval isn't physically larger than the selected window
            if interval > (end - start):
                interval = end - start
                print("Clamped 'interval' to current window size.")

            # 4. Finalize
            self.result = (start, end, interval)
            self.save_current_values(start, end, interval)
            self.accept()

        except ValueError:
            QMessageBox.critical(self, "Format Error", "Please enter valid numeric values.")

    def closeEvent(self, event):
        # Default behavior is fine, super handles clean-up
        super().closeEvent(event)


@get_from_dialog
class ConstDialog(QDialog):

    def __init__(self, param_dict, title, parent=None):
        super().__init__(parent)
        self.result = param_dict.copy()
        self.title = title
        self.line_edits = {}
        self.checkboxes = {}
        self.lists = {}
        self.spinboxes = {}  # NEW: Tracks the numeric inputs
        self.status_label = QLabel("")

        # Make the window a bit wider to accommodate the sliders nicely
        self.resize(550, 600)
        self.init_ui()

    def init_ui(self):
        # Clear any existing layout if this is called dynamically
        if self.layout() is not None:
            QWidget().setLayout(self.layout())

        self.setWindowTitle(self.title)
        print(f"Running init_ui ...")
        main_layout = QVBoxLayout()

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)

        for key, value in self.result.items():
            if isinstance(value, list):
                # --- List UI ---
                key_label = QLabel(key)
                layout.addWidget(key_label)
                self.lists[key] = QListWidget()
                for item in value:
                    QListWidgetItem(str(item), self.lists[key])
                layout.addWidget(self.lists[key])
                edit_button = QPushButton(f'Edit {key}')
                edit_button.clicked.connect(lambda checked, k=key: self.edit_list(k))
                layout.addWidget(edit_button)

            elif isinstance(value, bool):
                # --- Boolean UI ---
                checkbox = QCheckBox(key)  # Put label directly on checkbox for clean UI
                checkbox.setChecked(value)
                checkbox.setToolTip("Use several intervals if you activate this option: i.e. interval=600.")
                self.checkboxes[key] = checkbox
                layout.addWidget(checkbox)

            elif isinstance(value, (int, float)):
                # --- Slider & Spinbox UI for Numbers ---
                h_layout = QHBoxLayout()
                label = QLabel(key)
                label.setMinimumWidth(160)

                # 1. Apply your strict rules
                decimals = 4
                step = 0.0001

                # Set max_val to 1000x the value (with a default if value is 0)
                if value == 0:
                    max_val = 1000.0
                else:
                    max_val = abs(value) * 1000.0

                min_val = -max_val

                # 2. Configure Spinbox
                spinbox = QDoubleSpinBox()
                spinbox.setDecimals(decimals)
                spinbox.setRange(min_val, max_val)
                spinbox.setSingleStep(step)
                spinbox.setMinimumWidth(130)
                spinbox.setValue(float(value))

                # 3. Configure Slider (Normalized to 10,000 steps for smooth UI)
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setMaximumWidth(150)
                slider_steps = 10000
                slider.setRange(0, slider_steps)

                # Calculate initial slider position
                try:
                    initial_slider_pos = int(((value - min_val) / (max_val - min_val)) * slider_steps)
                except ZeroDivisionError:
                    initial_slider_pos = slider_steps // 2
                slider.setValue(initial_slider_pos)

                # 4. Link Slider and Spinbox with signals blocked to prevent recursion loops
                def update_spinbox(v, sb=spinbox, mn=min_val, mx=max_val, steps=slider_steps):
                    real_val = mn + (v / steps) * (mx - mn)
                    sb.blockSignals(True)
                    sb.setValue(real_val)
                    sb.blockSignals(False)

                def update_slider(v, sl=slider, mn=min_val, mx=max_val, steps=slider_steps):
                    try:
                        s_val = int(((v - mn) / (mx - mn)) * steps)
                    except ZeroDivisionError:
                        s_val = steps // 2
                    sl.blockSignals(True)
                    sl.setValue(s_val)
                    sl.blockSignals(False)

                slider.valueChanged.connect(update_spinbox)
                spinbox.valueChanged.connect(update_slider)

                self.spinboxes[key] = spinbox

                h_layout.addWidget(label)
                h_layout.addWidget(slider)
                h_layout.addWidget(spinbox)
                layout.addLayout(h_layout)
            else:
                # --- Standard Text UI ---
                h_layout = QHBoxLayout()
                key_label = QLabel(key)
                h_layout.addWidget(key_label)
                line_edit = QLineEdit(str(value))
                self.line_edits[key] = line_edit
                h_layout.addWidget(line_edit)
                layout.addLayout(h_layout)

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        # --- Action Buttons ---
        button_layout = QHBoxLayout()
        load_button = QPushButton("Load configuration")
        load_button.clicked.connect(self.select_file)
        button_layout.addWidget(load_button)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept_and_save)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def accept_and_save(self):
        # 1. Save Text Inputs
        for key, line_edit in self.line_edits.items():
            text = line_edit.text()
            try:
                # Handle string values like 'z'
                if text.isalpha() or not text:
                    self.result[key] = text
                else:
                    self.result[key] = float(text) if '.' in text else int(text)
            except ValueError:
                self.result[key] = text

        # 2. Save Checkboxes
        for key, checkbox in self.checkboxes.items():
            self.result[key] = checkbox.isChecked()

        # 3. Save Numeric Spinboxes
        for key, spinbox in self.spinboxes.items():
            if spinbox.decimals() == 0:
                self.result[key] = int(spinbox.value())
            else:
                self.result[key] = spinbox.value()

        self.accept()

    def edit_list(self, key):
        list_dialog = ListEditDialog(self.result[key])
        if list_dialog.exec() == QDialog.DialogCode.Accepted:
            self.result[key] = list_dialog.result
            self.lists[key].clear()
            for item in self.result[key]:
                QListWidgetItem(str(item), self.lists[key])

    def select_file(self):
        """
        Loads a JSON configuration file and updates the dialog's settings.
        """
        import os
        from PyQt6.QtWidgets import QMessageBox
        from lib_utility import get_previous_folder, save_previous_folder, load_dict

        # Local import to prevent circularity if open_file_dialog is not globally available

        previous_folder = get_previous_folder(self.title)
        if not previous_folder:
            previous_folder = os.path.expanduser("~")

        try:
            # Note the custom filter is set strictly to JSON
            file_path, _ = open_file_dialog(self, previous_folder, "JSON Files (*.json)")

            if file_path:
                save_previous_folder(os.path.dirname(file_path), self.title)

                # 1. Load the new configuration directly into the result dictionary
                loaded_dict = load_dict(file_path, self.result)
                self.result = loaded_dict.copy()

                self.status_label.setText(f"Loaded config: {os.path.basename(file_path)}")

                # 2. Accept the dialog so the main application can process the new settings
                self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading configuration: {e}")
            self.status_label.setText(f"Error loading file: {e}")
    # def select_file(self):
    #     print(f"Delete after {self.title=}")
    #     previous_folder = get_previous_folder(self.title)
    #     print(f"Delete after {previous_folder=}")
    #     if not previous_folder:
    #         previous_folder = os.path.expanduser("~")
    #         print(f"Delete after ~{previous_folder=}")
    #     try:
    #         file_path, _ = open_file_dialog(self, previous_folder, "JSON Files (*.json)")
    #         print(f"Delete after {file_path=}")
    #         if file_path:
    #             save_previous_folder(os.path.dirname(file_path), self.title)
    #
    #             # 1. Load the new data directly into the result dictionary
    #             loaded_dict = load_dict(file_path, self.result)
    #             print(f"Delete after {loaded_dict=}")
    #             self.result = loaded_dict.copy()
    #
    #             # 2. Skip the UI rebuild entirely and just close the dialog!
    #             self.accept()
    #
    #     except Exception as e:
    #         QMessageBox.critical(self, "Error", f"Error loading file: {e}")
    #         print(f"Delete after Error loading file: {e}")
    #         self.status_label.setText(f"Error loading file: {e}")

    def closeEvent(self, event):
        # Remove closeEvent, accept() handles it.
        super().closeEvent(event)


class ListEditDialog(QDialog):

    def __init__(self, items):
        super().__init__()
        self.result = items.copy()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Edit List")
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        for item in self.result:
            QListWidgetItem(str(item), self.list_widget)
        layout.addWidget(self.list_widget)
        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_item)
        layout.addWidget(add_button)
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self.remove_item)
        layout.addWidget(remove_button)
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        layout.addWidget(ok_button)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(cancel_button)
        self.setLayout(layout)

    def add_item(self):
        item, ok = QInputDialog.getText(self, "Add Item", "Enter item:")
        if ok and item:
            try:
                self.result.append(float(item) if '.' in item else int(item))
                QListWidgetItem(str(item), self.list_widget)
            except ValueError:
                QMessageBox.critical(self, "Error", "Invalid input.")

    def remove_item(self):
        selected_items = self.list_widget.selectedItems()
        for item in selected_items:
            index = self.list_widget.row(item)
            del self.result[index]
            self.list_widget.takeItem(index)

    def accept(self):
        super().accept()  # super accept is what closes the modal dialog.

    def closeEvent(self, event):
        # Remove closeEvent, accept() handles it.
        super().closeEvent(event)


@get_from_dialog
class NoiseDialog(QDialog):

    def __init__(self, start=0.0, end=0.0, parent=None):
        super().__init__(parent)
        self.start = start
        self.end = end
        self.result = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f"Noise region must be between {self.start} and {self.end} s")

        self.start_label = QLabel("Start Time:")
        self.start_input = QLineEdit()

        self.end_label = QLabel("End Time:")
        self.end_input = QLineEdit()

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.get_values)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        layout = QGridLayout()
        layout.addWidget(self.start_label, 0, 0)
        layout.addWidget(self.start_input, 0, 1)
        layout.addWidget(self.end_label, 1, 0)
        layout.addWidget(self.end_input, 1, 1)

        layout.addWidget(self.ok_button, 2, 0)
        layout.addWidget(self.cancel_button, 2, 1)

        self.setLayout(layout)

    def get_values(self):
        try:
            noise_start = float(self.start_input.text())
            noise_end = float(self.end_input.text())

            if not (self.start <= noise_start < self.end and self.start < noise_end <= self.end):
                QMessageBox.critical(self, "Error", "Noise values are out of range.")
                return

            self.result = (noise_start, noise_end)
            self.accept()
        except ValueError:
            QMessageBox.critical(self, f"Error", f"Values are out of range: {self.end - self.start} max.")

    def closeEvent(self, event):
        # Remove closeEvent, accept() handles it.
        super().closeEvent(event)


@get_from_dialog
class InputDialog(QDialog):  # Inherit from QDialog

    def __init__(self, title="", message="", warning="", num_type=""):
        super().__init__()
        self.title = title
        self.message = message
        self.warning = warning
        self.num_type = num_type
        self.result = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)

        layout = QVBoxLayout()

        # label = QLabel("How much time do you want to delete per control pulse (example: 0.1 sec):")
        label = QLabel(self.message)
        self.input_field = QLineEdit()
        ok_button = QPushButton("OK")

        ok_button.clicked.connect(self.get_value)

        layout.addWidget(label)
        layout.addWidget(self.input_field)
        layout.addWidget(ok_button)

        self.setLayout(layout)

    def get_value(self):
        try:
            match self.num_type:
                case "float":
                    self.result = float(self.input_field.text())
                case "integer":
                    self.result = int(self.input_field.text())
            self.accept()  # Use accept() to close the modal dialog
        except ValueError:
            # QMessageBox.critical(self, "Error", "Incorrect Value, try 0.1")
            QMessageBox.critical(self, "Error", self.warning)


def manage_settings(const_file: str, const: dict):
    """Opens a stored dictionary 'const_file' to be modified. If there is no 'const_file' stored,
    uses 'const' as a default to begin the modification"""
    loaded_dict = load_dict(const_file, const)
    updated_const: dict = ConstDialog(loaded_dict, "Save to JSON?")
    if updated_const:
        loaded_dict.update(updated_const)
        print("Constants updated.")
    else:
        print("Constants-dialog canceled.")
    save_dict(const_file, loaded_dict)
    return loaded_dict


class ParameterTunerDialog(QDialog):

    def __init__(self, initial_consts, aliases=None):
        super().__init__()
        self.const = initial_consts.copy()
        self.aliases = aliases or {}
        self.controls = {}

        self.setWindowTitle("Tune Parameters")
        self.resize(550, 600)
        self.window_layout = QVBoxLayout(self)
        self.setup_ui()

    def setup_ui(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.controls_layout = QVBoxLayout(scroll_content)

        for key, value in self.const.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue

            display_name = self.aliases.get(key, key.replace("_", " ").title())

            if isinstance(value, int):
                decimals = 0
                step = 1
                max_val = abs(value) * 100 if value != 0 else 100
                min_val = -max_val
            else:
                val_str = f"{value:.6f}".rstrip('0')
                if val_str.endswith('.'):
                    decimals = 1
                else:
                    decimals = len(val_str.split('.')[1])
                decimals = max(2, min(decimals, 6))
                step = 10 ** -decimals
                max_val = abs(value) * 1000 if value != 0 else 1.0
                min_val = -max_val

            self.add_control(key, display_name, min_val, max_val, step, decimals)

        self.controls_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        self.window_layout.addWidget(scroll_area)

        # --- The Two Buttons ---
        button_layout = QHBoxLayout()

        self.test_btn = QPushButton("Test Settings")
        self.test_btn.setStyleSheet("font-weight: bold; background-color: #e0e0e0; padding: 10px;")
        self.test_btn.clicked.connect(self.save_and_test)

        self.continue_btn = QPushButton("Accept & Continue")
        self.continue_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white; padding: 10px;")
        self.continue_btn.clicked.connect(self.save_and_accept)

        button_layout.addWidget(self.test_btn)
        button_layout.addWidget(self.continue_btn)
        self.window_layout.addLayout(button_layout)

    def add_control(self, key, label_text, min_val, max_val, step, decimals):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setMinimumWidth(160)

        spinbox = QDoubleSpinBox()
        spinbox.setDecimals(decimals)
        spinbox.setRange(min_val, max_val)
        spinbox.setSingleStep(step)
        spinbox.setMinimumWidth(130)

        initial_value = self.const.get(key, min_val)
        spinbox.setValue(initial_value)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMaximumWidth(150)

        scale_factor = 10 ** decimals
        slider.setRange(int(min_val * scale_factor), int(max_val * scale_factor))
        slider.setValue(int(initial_value * scale_factor))

        slider.valueChanged.connect(lambda v, sb=spinbox, sf=scale_factor: sb.setValue(v / sf))
        spinbox.valueChanged.connect(lambda v, sl=slider, sf=scale_factor: sl.setValue(int(v * sf)))

        self.controls[key] = spinbox
        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(spinbox)
        self.controls_layout.addLayout(layout)

    def save_gui_to_const(self):
        """Helper to read all GUI values back into the dictionary."""
        for key, spinbox in self.controls.items():
            if spinbox.decimals() == 0:
                self.const[key] = int(spinbox.value())
            else:
                self.const[key] = spinbox.value()

    def save_and_test(self):
        self.save_gui_to_const()
        self.done(2)  # Closes dialog and returns custom code 2 for "Test"

    def save_and_accept(self):
        self.save_gui_to_const()
        self.accept()  # Closes dialog and returns code 1 for "Accept"


if __name__ == "__main__":
    app = QApplication([])
    selector = AnalysisSelector()
    selector.show()
    app.exec()
