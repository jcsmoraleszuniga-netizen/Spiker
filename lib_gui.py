import gc
import os
from lib_event_detection import EvtPro
from lib_utility import get_previous_folder, load_dict, save_previous_folder
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QApplication, QWidget, QDialog, QLabel, QLineEdit, QPushButton,
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


def show_plot(original, title="No Title.", values=(0.0, 0.0)):
    """
    Displays a plot of the original data, optionally with vertical lines.

    Args:
        original: The data to plot (an object with coti and resp attributes).
        title: The title of the plot.
        values: A tuple or list of x-values for vertical lines.
    """
    global _current_plot
    if _current_plot:  # Close previous plot if it exists
        plt.close(_current_plot)
    fig = plt.figure()
    match original.mode:
        case "continuous":
            plt.plot(original.time, original.resp, linewidth=0.5)
            plt.plot(original.time, original.cdac, "r")  # TODO fix this in case of difference in length
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
        self.original = None
        self.title = title
        self.programs_dict = programs_dict
        self.checkboxes = []
        self.select_file_button = QPushButton("Select ABF File")
        self.select_file_button.clicked.connect(self.select_file)
        self.run_analysis_button = QPushButton("Run analysis")
        self.run_analysis_button.clicked.connect(self.run_analyses)
        self.status_label = QLabel("")
        self.init_ui()
        self.select_file()

    def init_ui(self):
        self.setWindowTitle(self.title)
        for checkbox_text, program in self.programs_dict.items():
            print(f"{checkbox_text = }")
            self.checkboxes.append(QCheckBox(checkbox_text))
        layout = QVBoxLayout()
        for checkbox in self.checkboxes:
            layout.addWidget(checkbox)
        layout.addWidget(self.select_file_button)
        layout.addWidget(self.run_analysis_button)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

    def select_file(self):
        previous_folder = get_previous_folder(self.title)
        if not previous_folder:
            previous_folder = os.path.expanduser("~")

        try:
            file_path, _ = open_file_dialog(self, previous_folder, "ABF Files (*.abf);; CSV Files (*.csv *.CSV)")
            if file_path:
                save_previous_folder(os.path.dirname(file_path), self.title)
                self.original = EvtPro(file_path, True)
                show_plot(self.original, title="Total response.")
                print(f"{self.original = }")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading file: {e}")
            self.status_label.setText(f"Error loading file: {e}")

    def run_analyses(self):
        plt.close('all')
        if len(self.original):
            self.status_label.setText("Running analyses...")
            try:
                analyses = dict(zip(self.checkboxes, self.programs_dict.values()))
                for checkbox, func in analyses.items():
                    if checkbox.isChecked():
                        with_sections(func, BoundariesDialog, self.original)
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
        self.start = start
        self.end = end
        self.title = title
        self.result = None
        self.start_label = QLabel("Start Time:")
        self.start_input = QLineEdit()
        self.end_label = QLabel("End Time:")
        self.end_input = QLineEdit()
        self.interval_label = QLabel(f"Interval Size:")
        self.interval_input = QLineEdit()
        self.default_checkbox = QCheckBox("Use default values")
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.get_values)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f"{self.title} boundaries: {self.start:5.3f}s to {self.end:5.3f}s")
        layout = QGridLayout()
        layout.addWidget(self.start_label, 0, 0)
        layout.addWidget(self.start_input, 0, 1)
        layout.addWidget(self.end_label, 1, 0)
        layout.addWidget(self.end_input, 1, 1)
        layout.addWidget(self.interval_label, 2, 0)
        layout.addWidget(self.interval_input, 2, 1)
        layout.addWidget(self.default_checkbox, 3, 0)
        layout.addWidget(self.ok_button, 4, 0)
        layout.addWidget(self.cancel_button, 4, 1)
        self.setLayout(layout)

    def get_values(self):
        if self.default_checkbox.isChecked():
            start = self.start
            end = self.end
            interval = self.end
            self.result = (start, end, interval)
            self.accept()
        else:
            try:
                start = float(self.start_input.text())
                end = float(self.end_input.text())
                interval = float(self.interval_input.text())

                if not (end < self.end):
                    end = self.end
                    print("Using max value for 'end'.")

                if not (0.0 <= self.start <= start < end and start < self.end):
                    QMessageBox.critical(self, f"Error", f"Range values are wrong: {self.end - self.start} max.")
                    return

                if not (interval > 0.0):
                    QMessageBox.critical(self, f"Error", f"Interval value must not be 0.0.")
                    return

                if not (interval <= (end - start)):
                    interval = end - start
                    print("Using full range for 'interval'.")

                self.result = (start, end, interval)
                self.accept()
            except ValueError:
                QMessageBox.critical(self, "Error", "Incorrect input/value, try again.")

    def closeEvent(self, event):
        # Remove closeEvent, accept() handles it.
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
        self.status_label = QLabel("")
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        print(f"Running init_ui ...")
        main_layout = QVBoxLayout()

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)

        for key, value in self.result.items():
            key_label = QLabel(key)
            layout.addWidget(key_label)
            if isinstance(value, list):
                self.lists[key] = QListWidget()
                for item in value:
                    QListWidgetItem(str(item), self.lists[key])
                layout.addWidget(self.lists[key])
                edit_button = QPushButton(f'Edit {key}')
                edit_button.clicked.connect(lambda checked, k=key: self.edit_list(k))
                layout.addWidget(edit_button)
            elif isinstance(value, bool):
                checkbox = QCheckBox()
                checkbox.setChecked(value)
                checkbox.setToolTip("Use several intervals if you activate this option: i.e. interval=600.")
                self.checkboxes[key] = checkbox
                layout.addWidget(checkbox)
            else:
                line_edit = QLineEdit(str(value))
                self.line_edits[key] = line_edit
                layout.addWidget(line_edit)

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept_and_save)
        main_layout.addWidget(ok_button)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        main_layout.addWidget(cancel_button)
        load_button = QPushButton("Load configuration")
        load_button.clicked.connect(self.select_file)
        main_layout.addWidget(load_button)

        self.setLayout(main_layout)

    def accept_and_save(self):
        for key, line_edit in self.line_edits.items():
            try:
                # Handle string values like 'z'
                if line_edit.text().isalpha():
                    self.result[key] = line_edit.text()
                else:
                    self.result[key] = float(line_edit.text()) if '.' in line_edit.text() else int(line_edit.text())
            except ValueError:
                QMessageBox.critical(self, "Error", f"Invalid value for {key}")
                return
        for key, checkbox in self.checkboxes.items():
            self.result[key] = checkbox.isChecked()

        self.accept()

    def edit_list(self, key):
        list_dialog = ListEditDialog(self.result[key])
        if list_dialog.exec() == QDialog.DialogCode.Accepted:
            self.result[key] = list_dialog.result
            self.lists[key].clear()
            for item in self.result[key]:
                QListWidgetItem(str(item), self.lists[key])

    def select_file(self):
        previous_folder = get_previous_folder(self.title)
        if not previous_folder:
            previous_folder = os.path.expanduser("~")
        try:
            file_path, _ = open_file_dialog(self, previous_folder, "JSON Files (*.json)")
            if file_path:
                save_previous_folder(os.path.dirname(file_path), self.title)
                loaded_dict = load_dict(file_path, self.result)
                self.result = loaded_dict.copy()
                self.line_edits = {}
                self.checkboxes = {}
                self.status_label = QLabel("")
                self.init_ui()
                self.accept_and_save()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading file: {e}")
            self.status_label.setText(f"Error loading file: {e}")

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
        super().accept()  #super accept is what closes the modal dialog.

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


if __name__ == "__main__":
    app = QApplication([])
    selector = AnalysisSelector()
    selector.show()
    app.exec()
