from PyQt5.QtWidgets import QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QGraphicsPathItem, QGraphicsRectItem, QGraphicsPolygonItem
from PyQt5.QtGui import QImage, QPixmap, QPen, QBrush, QColor, QPainterPath
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, qInstallMessageHandler, QtMsgType
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtWidgets
from PyQt5.QtSvg import QSvgGenerator
from PyQt5.QtGui import QPainter
import numpy as npy
import os
import sys
import joblib
import pickle
from sklearn.neural_network import MLPClassifier
import onnxruntime as ort
from PyQt5.QtWidgets import QFileDialog, QListWidgetItem
from datetime import datetime
import csv

from modules.OCTopus_ui import Ui_OCTopus
from modules.ui_timeline import Ui_timeline
from modules.fundus_view import FundusView
from modules.retina_data import RetinaData
from modules.persistence import PersistenceManager

from PyQt5.QtCore import QThread, pyqtSignal

from preprocessing.retina_preprocessing import extract_frames_from_avi, process_folder

import warnings
from cryptography.utils import CryptographyDeprecationWarning

warnings.simplefilter("ignore", category=CryptographyDeprecationWarning)

# enable high-DPI scaling (Windows)
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

class PreprocessWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, selected_items, force_redo=False):
        super().__init__()
        self.selected_items = selected_items
        self.force_redo = force_redo

    def run(self):
        total = len(self.selected_items)
        for i, item in enumerate(self.selected_items):
            if item.endswith('.avi'):
                output_folder = os.path.splitext(item)[0]
                if self.force_redo or not os.path.exists(output_folder):
                    extract_frames_from_avi(item, output_folder)
                process_folder(output_folder, self.force_redo)
            elif os.path.isdir(item):
                process_folder(item, self.force_redo)
            else:
                print(f"Skipping {item}: not an .avi file or folder.")
            self.progress.emit(int((i + 1) / total * 100))
        self.finished.emit()

class PredictionWorker(QThread):
    # signals to communicate with the main thread
    prediction_complete = pyqtSignal()
    finished = pyqtSignal()
    progress_update = pyqtSignal(int)

    def __init__(self, parent, stack_index):
        super().__init__(parent)
        self.stack_index = stack_index

    def run(self):
        # get the total number of slices
        total_slices = self.parent().data.stacks[self.stack_index].shape[0]
        self.parent().pushButton_5.setEnabled(True)  # enable stop button 
        # process each slice
        for slice_index in range(total_slices):
            if self.parent().stop_flag:
                print("Prediction stopped by user!")
                self.parent().log_action(f"Prediction stopped at slice {slice_index+1}/{total_slices}")
                # reset button states
                self.parent().run_model_on_stack.setEnabled(True)  # re-enable run button
                self.parent().pushButton_5.setEnabled(False)   # disable stop button

                break
                
            self.parent()._predict_single_slice(self.stack_index, slice_index)
            # emit progress as a percentage
            self.progress_update.emit(int((slice_index + 1) / total_slices * 100))
        # reset button states
        self.parent().run_model_on_stack.setEnabled(True)  # re-enable run button
        self.parent().pushButton_5.setEnabled(False)   # disable stop button
        # signal that we’re done
        self.prediction_complete.emit()

class BatchPredictionWorker(QThread):
    progress_update = pyqtSignal(int)
    stack_processed = pyqtSignal(int)  # signal for fundus export
    finished = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

    def run(self):
        total_stacks = len(self.parent.data.stacks)
        for stack_index in range(total_stacks):
            if self.parent.stop_flag:
                break
            try:
                total_slices = self.parent.data.stacks[stack_index].shape[0]
                self.parent.log_action(f"Batch prediction started on stack {self.parent.data.stack_names[stack_index]} {stack_index+1}/{total_stacks} : {total_slices} slices")
                for slice_index in range(total_slices):
                    if self.parent.stop_flag:
                        print("Prediction stopped by user!")
                        self.parent.log_action(f"Batch prediction stopped at slice {slice_index+1}/{total_slices}")
                        break
                    self.parent._predict_single_slice(stack_index, slice_index)
                stack_path = self.parent.data.stack_paths[stack_index]
                self.parent.data.save_stack_data(stack_index, stack_path)
                self.stack_processed.emit(stack_index)  # trigger fundus export
                self.progress_update.emit(int((stack_index + 1) / total_stacks * 100))
            except Exception as e:
                print(f"Error processing stack {stack_index}: {e}")
                continue
        self.finished.emit()

class TimelineGUI(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent  # set main_window before setupUi
        self.ui = Ui_timeline()
        self.ui.setupUi(self)
        
        # populate combo box with formatted animal ID and eye
        self.timeline_keys = list(self.main_window.data.timelines.keys())
        display_names = []
        for key in self.timeline_keys:
            animal_id, eye = key.rsplit('_', 1)
            eye_str = "Right" if eye == "R" else "Left"
            display_names.append(f"{animal_id} ({eye_str})")
        self.ui.animal_combo.addItems(display_names)
        self.ui.animal_combo.currentIndexChanged.connect(self.on_animal_combo_changed)
        self.ui.generate_timeline_button.clicked.connect(self.generate_timeline)
        self.ui.export_timeline_button.clicked.connect(self.export_timeline)
        
        # ensure the window is large enough to show content
        self.setMinimumSize(1200, 600)
        self.ui.control_group.setMinimumHeight(50)

    def generate_timeline(self):
        current_index = self.ui.animal_combo.currentIndex()
        if current_index < 0 or not self.timeline_keys:
            print("No timeline selected or available!")
            return
        
        timeline_key = self.timeline_keys[current_index]
        raw_stats = self.main_window.data.get_timeline_stats_raw(timeline_key)
        if not raw_stats:
            print("No raw stats data available for this timeline!")
            self.ui.plot_widget.set_data([])
            for i in reversed(range(self.ui.grid_layout.count())):
                widget = self.ui.grid_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            return

        stats_list = []
        for raw_stat in raw_stats:  # iterate over each stack in the timeline
            stack_index = raw_stat['stack_index']
            stack_index = raw_stat['stack_index']
            
            # jump to the stack to update all relevant views and state
            self.main_window.jump_to_stack(stack_index)

            # calculate area using FundusView
            total_area_px, region_areas_px = self.main_window.graphicsView_2.calculate_annotated_area()
            _, um_per_px = self.main_window.data.get_fundus_scalebar_info(stack_index)

            folder = raw_stat['folder']
            fundus_path = os.path.join(folder, 'averaged_frame.png')
            
            # set up FundusView for this specific stack
            if os.path.exists(fundus_path):
                pixmap = QPixmap(fundus_path)
                if not pixmap.isNull():
                    self.main_window.graphicsView_2.set_fundus_image(pixmap)  # set new image
                else:
                    print(f"Invalid pixmap for {fundus_path}, skipping stack {stack_index}")
                    continue
            else:
                print(f"Fundus image not found at {fundus_path}, skipping stack {stack_index}")
                continue

            # reset annotations and update context for this stack
            self.main_window.current_stack_index = stack_index
            self.main_window.graphicsView_2.current_stack_index = stack_index # set stack index on FundusView
            # clear existing annotation items to prevent carryover
            for item in self.main_window.graphicsView_2.annotation_items:
                if item.scene() == self.main_window.graphicsView_2.scene:  # remove from scene if it's there
                    self.main_window.graphicsView_2.scene.removeItem(item)
            self.main_window.graphicsView_2.annotation_items = []  # reset the list
            self.main_window.graphicsView_2.update_annotations()  # update annotations for this stack

            # calculate area using FundusView
            total_area_um2 = total_area_px * (um_per_px ** 2) if um_per_px else total_area_px
            num_regions = len(region_areas_px)

            # stats for this stack
            stats = {
                'date': raw_stat['date'],
                'scanned_area': f"{raw_stat['stack'].shape[1]}x{raw_stat['stack'].shape[2]}",
                'num_slices': raw_stat['stack'].shape[0],
                'num_regions': num_regions,
                'total_area_um2': total_area_um2,
                'analysis_date': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                'stack_index': stack_index,
                'eye': timeline_key.split('_')[-1]
            }
            stats_list.append(stats)
            # print(f"Processed stack {stack_index}: Area = {total_area_um2} um², Regions = {num_regions}")  # debug

        self.ui.plot_widget.set_data(stats_list)
        for i in reversed(range(self.ui.grid_layout.count())):
            widget = self.ui.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        for i, stats in enumerate(stats_list):
            pixmap = self.main_window.generate_fresh_fundus_pixmap(stats['stack_index'])
            thumbnail_label = QtWidgets.QLabel()
            thumbnail_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
            stats_label = QtWidgets.QLabel()
            stats_label.setText(f"Date: {stats['date']}\nArea: {stats['total_area_um2']:.2f} μm²\nRegions: {stats['num_regions']}")
            self.ui.grid_layout.addWidget(thumbnail_label, 0, i)
            self.ui.grid_layout.addWidget(stats_label, 1, i)

    def on_animal_combo_changed(self):
        # auto-generate timeline when animal is selected
        self.generate_timeline()

    
    def export_timeline(self):
        if not self.ui.grid_layout.count():
            QMessageBox.information(self, "No timeline!", "No timeline to export")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save Timeline", "", "PNG Files (*.png)")
        if fname:
            pixmap = QtWidgets.QWidget.grab(self.ui.scroll_area.viewport())
            pixmap.save(fname, "PNG")
            self.main_window.log_action(f"Exported timeline to {fname}")



class oct_main(QMainWindow, Ui_OCTopus):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.data = RetinaData()

        self.setWindowTitle("OCTOPUS")

        # conditional scaling for high-DPI Windows screens
        screen = QApplication.primaryScreen()
        dpi = screen.logicalDotsPerInch()
        scale_ratio = screen.devicePixelRatio()
        is_windows = sys.platform == "win32"
        if is_windows and (dpi > 96 or scale_ratio > 1):
            scale_factor = 0.9
            for widget in self.findChildren(QtWidgets.QWidget):
                if widget.geometry().isValid():
                    geo = widget.geometry()
                    widget.setGeometry(
                        int(geo.x() * scale_factor),
                        int(geo.y() * scale_factor),
                        int(geo.width() * scale_factor),
                        int(geo.height() * scale_factor)
                    )

        self.graphicsView.main_window = self
        self.graphicsView_2.main_window = self

        self.current_stack_index = 0
        self.current_slice_index = 0
        self.selected_range = None
        self.threshold = 0.5
        self.column_probability_method = 'max'
        self.highlight_mode = False
        self.model_annotation_items = []  # store model annotations separately

        self.stop_flag = False # stop prediction thread

        self.log_dir = None

        self.setup_slice_view()
        self.connect_signals()
        self.update_ui_states()

    def setup_slice_view(self):
        self.slice_scene = QGraphicsScene()
        self.graphicsView.setScene(self.slice_scene)
        self.slice_pixmap_item = None
        self.probability_plot_item = None
        self.threshold_line_item = None
        self.annotation_items = []

    def connect_signals(self):
        self.next_slice.clicked.connect(self.on_next_slice)
        self.previous_slice.clicked.connect(self.on_previous_slice)
        self.horizontalSlider.valueChanged.connect(self.slider_changed)
        self.threshold_SpinBox.valueChanged.connect(self.update_threshold)
        self.add_annotation.clicked.connect(self.on_add_annotation)
        self.remove_annotation.clicked.connect(self.on_remove_annotation)
        self.clear_allannotations.clicked.connect(self.clear_all_annotations)
        self.enable_highlighter.stateChanged.connect(self.toggle_highlighter)
        self.doubleSpinBox_2.valueChanged.connect(self.update_scalebar)
        
        self.open_folder_dialog.clicked.connect(self.open_folder_dlg)
        self.preprocess_script_run.clicked.connect(self.preprocess_folders)
        self.load_stacks_from_selected_folders.clicked.connect(self.load_stacks)

        self.open_unet_dialog.clicked.connect(self.load_unet_model)

        self.next_stack.clicked.connect(self.on_next_stack)
        self.previous_stack.clicked.connect(self.on_previous_stack)

        self.run_model_on_stack.clicked.connect(self.predict_stack)
        self.pushButton_5.clicked.connect(self.stop_prediction_process)
        self.run_model_single_slice.clicked.connect(self.predict_slice)
        self.show_model_probability.stateChanged.connect(self.update_display)
        self.save_stack_images.clicked.connect(self.export_fundus_svg)
        self.save_slice_image.clicked.connect(self.export_slice_svg)
        self.open_log_dialog.clicked.connect(self.set_log_location)
        self.append_log_current_stack.clicked.connect(self.save_note)
        self.batch_process_all.clicked.connect(self.on_batch_process_all)

        self.open_mlp_model_dialog.clicked.connect(self.open_timeline)

    @pyqtSlot()
    def stop_prediction_process(self):
        """Slot to set the stop flag when the stop button is clicked."""
        if hasattr(self, 'active_worker') and self.active_worker.isRunning():
            self.stop_flag = True
            print("Stopping current process...")
    def update_slice_display(self):
        if not self.data.stacks:
            return
        slice_data = self.data.get_slice(self.current_stack_index, self.current_slice_index)
        if slice_data is None:
            return
        
        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
        slice_data = slice_data.astype(npy.uint8)
        height, width = slice_data.shape
        qimage = QImage(slice_data.data, width, height, slice_data.strides[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        if self.slice_pixmap_item:
            self.slice_scene.removeItem(self.slice_pixmap_item)
        self.slice_pixmap_item = QGraphicsPixmapItem(pixmap)
        self.slice_scene.addItem(self.slice_pixmap_item)
        self.update_overlays()
        
        total_slices = self.data.stacks[self.current_stack_index].shape[0]
        self.current_slice_label.setText(str(self.current_slice_index + 1))
        self.total_slices_label.setText(f"/{total_slices}")

    def update_overlays(self):
        annotations = self.data.get_annotations(self.current_stack_index, self.current_slice_index)
        probabilities = self.data.get_probabilities(self.current_stack_index, self.current_slice_index)
        self.update_probability_plot(probabilities)
        self.update_annotations(annotations)

    def update_probability_plot(self, probabilities):
        if probabilities is None or not self.show_model_probability.isChecked():
            if self.probability_plot_item:
                self.slice_scene.removeItem(self.probability_plot_item)
                self.probability_plot_item = None
            if self.threshold_line_item:
                self.slice_scene.removeItem(self.threshold_line_item)
                self.threshold_line_item = None
            return
        slice_data = self.data.get_slice(self.current_stack_index, self.current_slice_index)
        if slice_data is None:
            return
        height, width = slice_data.shape
        plot_path = QPainterPath()
        for x, p in enumerate(probabilities):
            y = height - p * height
            if x == 0:
                plot_path.moveTo(x, y)
            else:
                plot_path.lineTo(x, y)
        if self.probability_plot_item:
            self.slice_scene.removeItem(self.probability_plot_item)
        self.probability_plot_item = QGraphicsPathItem(plot_path)
        self.probability_plot_item.setPen(QPen(QColor("red"), 2))
        self.slice_scene.addItem(self.probability_plot_item)
        threshold = self.threshold_SpinBox.value()
        threshold_y = height - threshold * height
        if self.threshold_line_item:
            self.slice_scene.removeItem(self.threshold_line_item)
        self.threshold_line_item = self.slice_scene.addLine(0, threshold_y, width, threshold_y, 
                                                            QPen(QColor("blue"), 1, Qt.DotLine))

    def update_annotations(self, annotations):
        if annotations is None:
            return
        slice_data = self.data.get_slice(self.current_stack_index, self.current_slice_index)
        if slice_data is None:
            return
        height, width = slice_data.shape
        for item in self.annotation_items:
            self.slice_scene.removeItem(item)
        self.annotation_items = []
        start = None
        for i, val in enumerate(annotations):
            if val != 0 and start is None:
                start = i
            elif val == 0 and start is not None:
                color = QColor(0, 0, 255, 40) if annotations[start] == 2 else QColor(145, 0, 110, 40)  # blue for manual, pink for model
                rect = QGraphicsRectItem(start, 0, i - start, height)
                rect.setBrush(QBrush(color))
                self.slice_scene.addItem(rect)
                self.annotation_items.append(rect)
                start = None
        if start is not None:
            color = QColor(0, 0, 255, 40) if annotations[start] == 2 else QColor(145, 0, 110, 40)
            rect = QGraphicsRectItem(start, 0, len(annotations) - start, height)
            rect.setBrush(QBrush(color))
            self.slice_scene.addItem(rect)
            self.annotation_items.append(rect)

    def update_model_annotations(self, annotations):
        if annotations is None:
            return
        slice_data = self.data.get_slice(self.current_stack_index, self.current_slice_index)
        if slice_data is None:
            return
        height, width = slice_data.shape
        for item in self.model_annotation_items:
            self.slice_scene.removeItem(item)
        self.model_annotation_items = []
        start = None
        for i, val in enumerate(annotations):
            if val != 0 and annotations[i] != 2 and start is None:  # only model annotations (not manual)
                start = i
            elif val == 0 and start is not None:
                color = QColor(145, 0, 110, 40)  # pink for model annotations
                rect = QGraphicsRectItem(start, 0, i - start, height)
                rect.setBrush(QBrush(color))
                self.slice_scene.addItem(rect)
                self.model_annotation_items.append(rect)
                start = None
        if start is not None:
            color = QColor(145, 0, 110, 40)
            rect = QGraphicsRectItem(start, 0, len(annotations) - start, height)
            rect.setBrush(QBrush(color))
            self.slice_scene.addItem(rect)
            self.model_annotation_items.append(rect)

    def load_fundus_image(self):
        if not self.data.stacks:
            return
        fundus_image = self.data.get_fundus_image(self.current_stack_index)
        bounding_box, frame_edge = self.data.get_fundus_overlays(self.current_stack_index)
        self.graphicsView_2.set_fundus_image(fundus_image)
        if not fundus_image.isNull():
            self.graphicsView_2.set_fundus_overlays(bounding_box, frame_edge)
        self.update_fundus_view()

    def update_fundus_view(self):
        if not self.data.stacks:
            return
        slice_pos_y = self.data.slice_pos_y.get(self.current_stack_index, [])
        if slice_pos_y.size > 0:
            y_pos = slice_pos_y[self.current_slice_index]
            self.graphicsView_2.update_slice_line(y_pos)
            self.graphicsView_2.update_annotations()  # fundus keeps blue annotations
            self.graphicsView_2.update_annotated_areas()
            self.update_statistics()

    def open_folder_dlg(self):
        base_dir = QFileDialog.getExistingDirectory(self, "Select Base Directory")
        if base_dir:
            self.base_dir = base_dir
            
            log_path_file = os.path.join(self.base_dir, 'log_path.txt')

            # check if log_path.txt exists and load the log directory
            if os.path.exists(log_path_file):
                with open(log_path_file, 'r') as f:
                    potential_log_dir = f.read().strip()
                    # make sure the stored path still exists
                    if os.path.exists(potential_log_dir):
                        self.log_dir = potential_log_dir
                    else:
                        # fall back to default if the stored path is invalid
                        self.log_dir = os.path.join(self.base_dir, 'logs')
                        if not os.path.exists(self.log_dir):
                            os.makedirs(self.log_dir)
            else:
                # no log_path.txt, so use the default logs folder
                self.log_dir = os.path.join(self.base_dir, 'logs')
                with open(log_path_file, 'w') as f:
                    f.write(self.log_dir)
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)

            
            # update the UI to show the log location
            log_path = os.path.join(self.log_dir, 'session.log')
            with open(log_path, "a") as f:
                f.write(f"\n--- Session Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            self.label_3.setText(f"Log location: {self.log_dir}")
            self.label_3.setToolTip(f"Log location: {self.log_dir}")

            self.folders_list.clear()
            for item in os.listdir(base_dir):
                full_path = os.path.join(base_dir, item)
                if os.path.isdir(full_path) or full_path.endswith('.avi'):
                    item_name = os.path.basename(full_path)
                    list_item = QListWidgetItem(item_name)
                    list_item.setData(Qt.UserRole, full_path)
                    self.folders_list.addItem(list_item)

    def preprocess_folders(self):
        selected_items = self.folders_list.selectedItems()
        avi_files = [item.data(Qt.UserRole) for item in selected_items if item.data(Qt.UserRole).endswith('.avi')]
        if not avi_files:
            return
        self.processing_progressbar.setValue(0)
        self.worker = PreprocessWorker(avi_files, self.force_reprocessing.isChecked())
        self.worker.progress.connect(self.processing_progressbar.setValue)
        self.worker.finished.connect(self.on_preprocess_finished)
        self.worker.start()

    def on_preprocess_finished(self):
        self.processing_progressbar.setValue(100)
        self.refresh_folders_list()

    def refresh_folders_list(self):
        if hasattr(self, 'base_dir') and self.base_dir:
            self.folders_list.clear()
            for item in os.listdir(self.base_dir):
                full_path = os.path.join(self.base_dir, item)
                if os.path.isdir(full_path) or full_path.endswith('.avi'):
                    item_name = os.path.basename(full_path)
                    list_item = QListWidgetItem(item_name)
                    list_item.setData(Qt.UserRole, full_path)
                    self.folders_list.addItem(list_item)

    def load_stacks(self):
        selected_items = self.folders_list.selectedItems()
        self.selected_folders = [item.data(Qt.UserRole) for item in selected_items if os.path.isdir(item.data(Qt.UserRole))]
        if not self.selected_folders:
            return
        self.folders_list.clear()
        complete_folders = []

        for folder in self.selected_folders:
            item_name = os.path.basename(folder)
            list_item = QListWidgetItem(item_name)
            
            # required contents:
            required = ['averaged_frame.png',
                'green_frame.npz',
                'grey_oct.npz',
                'OCT_stack.npz']
            print(f'Loading folder: {folder}')
            if all(os.path.exists(os.path.join(folder, required_file)) for required_file in required):
                complete_folders.append(folder)
            else:
                print(f'Missing required files in {folder} folder not loaded. Please run the preprocessing step.')
                continue


            list_item.setData(Qt.UserRole, folder)
            self.folders_list.addItem(list_item)
        
        if self.folders_list.count() > 0:
            self.data.load_stacks(complete_folders)
            
            if self.data.stacks:
                total_slices = self.data.stacks[self.current_stack_index].shape[0]
                self.horizontalSlider.setMaximum(total_slices - 1)
                self.horizontalSlider.setValue(self.current_slice_index)
            self.current_stack_index = 0
            self.current_slice_index = 0
            self.load_settings()
            self.load_fundus_image()
            self.update_slice_display()
            self.log_action(f'loaded stacks: {self.selected_folders}')
            self.log_action(f'active stack: {self.data.stack_names[self.current_stack_index]}')
            self.setWindowTitle(f"OCTOPUS : {self.data.stack_names[self.current_stack_index]}")
            self.update_ui_states()

    def load_unet_model(self):
        """Load multiple unet models from a selected folder."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select UNet Models Folder")
        if folder_path:
            onnx_files = [f for f in os.listdir(folder_path) if f.endswith('.onnx')]
            if not onnx_files:
                print("No ONNX files found in the selected folder.")
                self.unet_location.setText("No models loaded")
                self.unet_location.setToolTip("No models loaded")
                return
            self.data.unet_models = []
            for onnx_file in onnx_files:
                model_path = os.path.join(folder_path, onnx_file)
                try:
                    self.data.model_names.append(onnx_file)
                    session = ort.InferenceSession(model_path)
                    self.data.unet_models.append(session)
                except Exception as e:
                    print(f"Failed to load {model_path}: {e}")
            if self.data.unet_models:
                self.unet_location.setText(f"Loaded {len(self.data.unet_models)} models from {folder_path}")
                self.unet_location.setToolTip(f"Loaded {len(self.data.unet_models)} models from {folder_path}:\n{self.data.model_names}")
                self.log_action(f'loaded {self.data.model_names} from {folder_path}')
            else:
                self.unet_location.setText("No models loaded")
                self.unet_location.setToolTip("No models loaded")
                self.log_action(f'failed to load models from {folder_path}')

    def default_model_locations(self):
        try:
            folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'UNet')
            onnx_files = [f for f in os.listdir(folder_path) if f.endswith('.onnx')]
            if not onnx_files:
                print("No ONNX files found in the selected folder.")
                self.unet_location.setText("No models loaded")
                self.unet_location.setToolTip("No models found")
                return
            self.data.unet_models = []
            for onnx_file in onnx_files:
                model_path = os.path.join(folder_path, onnx_file)
                try:
                    self.data.model_names.append(onnx_file)
                    session = ort.InferenceSession(model_path)
                    self.data.unet_models.append(session)
                except Exception as e:
                    print(f"Failed to load {model_path}: {e}")
            if self.data.unet_models:
                self.unet_location.setText(f"Loaded {len(self.data.unet_models)} models from {folder_path}")
                self.unet_location.setToolTip(f"Loaded {len(self.data.unet_models)} models from {folder_path}:\n{self.data.model_names}")
                self.log_action(f'loaded (default) {self.data.model_names} from {folder_path}')
            else:
                self.unet_location.setText("No models loaded")
                self.unet_location.setToolTip("No models loaded")
                self.log_action(f'failed to load models from {folder_path}')
        except:
            print("Failed to load default UNet models.")
            self.unet_location.setText("No default UNet models found")
            self.unet_location.setToolTip("No default UNet models found")
            self.log_action(f'failed to load default UNet models')



    def on_next_stack(self):
        if self.current_stack_index < len(self.data.stacks) - 1:
            self.current_stack_index += 1
            self.current_slice_index = 0
            total_slices = self.data.stacks[self.current_stack_index].shape[0]
            self.horizontalSlider.setMaximum(total_slices - 1)
            self.horizontalSlider.setValue(0)
            self.log_action(f'active stack: {self.data.stack_names[self.current_stack_index]}')
            self.load_settings()
            self.load_fundus_image()
            self.update_slice_display()
            self.setWindowTitle(f"OCTOPUS : {self.data.stack_names[self.current_stack_index]}")
            self.update_ui_states()

    def on_previous_stack(self):
        if self.current_stack_index > 0:
            self.current_stack_index -= 1
            self.current_slice_index = 0
            total_slices = self.data.stacks[self.current_stack_index].shape[0]
            self.horizontalSlider.setMaximum(total_slices - 1)
            self.horizontalSlider.setValue(0)
            self.log_action(f'active stack: {self.data.stack_names[self.current_stack_index]}')
            self.load_settings()
            self.update_slice_display()
            self.load_fundus_image()
            self.setWindowTitle(f"OCTOPUS : {self.data.stack_names[self.current_stack_index]}")
            self.update_ui_states()

    def slider_changed(self, value):
        self.current_slice_index = value
        self.update_slice_display()
        self.update_fundus_view()

    def update_threshold(self, value):
        if self.data.stacks and self.current_stack_index < len(self.data.stacks):
            self.threshold = value
            probabilities = self.data.get_probabilities(self.current_stack_index, self.current_slice_index)
            self.update_probability_plot(probabilities)
            stack_path = self.data.stack_paths[self.current_stack_index]
            PersistenceManager.save_settings(
                stack_path,
                self.current_stack_index,
                self.threshold,
                self.doubleSpinBox_2.value()
            )

    def on_next_slice(self):
        if self.data.stacks and self.current_stack_index < len(self.data.stacks):
            total_slices = self.data.stacks[self.current_stack_index].shape[0]
            if self.current_slice_index < total_slices - 1:
                self.current_slice_index += 1
                self.update_slice_display()
                self.update_ui_states()
                self.horizontalSlider.setValue(self.current_slice_index)

    def on_previous_slice(self):
        if self.data.stacks and self.current_stack_index < len(self.data.stacks):
            if self.current_slice_index > 0:
                self.current_slice_index -= 1
                self.update_slice_display()
                self.update_ui_states()
                self.horizontalSlider.setValue(self.current_slice_index)

    def on_add_annotation(self):
        if self.selected_range and self.data.stacks:
            start, end = self.selected_range
            # ensure the stack index exists in annotations
            if self.current_stack_index not in self.data.annotations:
                self.data.annotations[self.current_stack_index] = {}
            
            # if the slice index doesn't exist, set it as a view of annotations_array
            if self.current_slice_index not in self.data.annotations[self.current_stack_index]:
                self.data.annotations[self.current_stack_index][self.current_slice_index] = self.data.annotations_array[self.current_stack_index][self.current_slice_index, :]
            self.data.annotations_array[self.current_stack_index][self.current_slice_index, start:end+1] = 2
            self.data.annotations[self.current_stack_index][self.current_slice_index][start:end+1] = 2
            stack_path = self.data.stack_paths[self.current_stack_index]
            self.data.save_stack_data(self.current_stack_index, stack_path)
            total_slices = self.data.stacks[self.current_stack_index].shape[0]
            self.log_action(f'add annotation (manual); slice: {self.current_slice_index+1}/{total_slices} from {start} to {end}')
            self.update_overlays()
            self.update_fundus_view()
            self.graphicsView.clear_selection_rect()  # clear after adding

    def on_remove_annotation(self):
        if self.selected_range and self.data.stacks:
            start, end = self.selected_range
            # ensure the stack index exists in annotations
            if self.current_stack_index not in self.data.annotations:
                self.data.annotations[self.current_stack_index] = {}
            
            # if the slice index doesn't exist, set it as a view of annotations_array
            if self.current_slice_index not in self.data.annotations[self.current_stack_index]:
                self.data.annotations[self.current_stack_index][self.current_slice_index] = self.data.annotations_array[self.current_stack_index][self.current_slice_index, :]

            self.data.annotations_array[self.current_stack_index][self.current_slice_index, start:end+1] = 0
            self.data.annotations[self.current_stack_index][self.current_slice_index][start:end+1] = 0
            stack_path = self.data.stack_paths[self.current_stack_index]
            self.data.save_stack_data(self.current_stack_index, stack_path)
            total_slices = self.data.stacks[self.current_stack_index].shape[0]
            self.log_action(f'remove annotation (manual); slice: {self.current_slice_index+1}/{total_slices} from {start} to {end}')
            self.update_overlays()
            self.update_fundus_view()
            self.graphicsView.clear_selection_rect()  # clear after removing

    def update_ui_states(self):
        if self.data.stacks:
            total_slices = self.data.stacks[self.current_stack_index].shape[0]
            self.next_slice.setEnabled(self.current_slice_index < total_slices - 1)
            self.previous_slice.setEnabled(self.current_slice_index > 0)
        else:
            self.next_slice.setEnabled(False)
            self.previous_slice.setEnabled(False)

    def clear_all_annotations(self):
        if self.data.stacks:
            stack_index = self.current_stack_index
            slice_index = self.current_slice_index
            if slice_index not in self.data.annotations[stack_index]:
                # if not, create a new array of zeros based on slice width
                slice_width = self.data.stacks[stack_index].shape[2]  #  shape is (slices, height, width)
                self.data.annotations[stack_index][slice_index] = npy.zeros(slice_width, dtype='uint8')
            else:
                # if it exists, set all values to zero
                self.data.annotations[stack_index][slice_index][:] = 0
        
            self.data.annotations_array[self.current_stack_index][self.current_slice_index, :] = 0
            # self.data.annotations[self.current_stack_index][self.current_slice_index][:] = 0
            stack_path = self.data.stack_paths[self.current_stack_index]
            self.data.save_stack_data(self.current_stack_index, stack_path)
            total_slices = self.data.stacks[self.current_stack_index].shape[0]
            self.log_action(f'clear all annotations (manual); slice: {self.current_slice_index+1}/{total_slices}')
            self.update_overlays()
            self.update_slice_display()
            self.update_fundus_view()
            self.graphicsView.clear_selection_rect()  # clear after clearing all
        else:
            return

    def toggle_highlighter(self, state):
        self.highlight_mode = state == Qt.Checked
        self.add_annotation.setEnabled(self.highlight_mode)
        self.remove_annotation.setEnabled(self.highlight_mode)

    def load_settings(self):
        """Load threshold and scalebar settings."""
        stack_path = self.data.stack_paths[self.current_stack_index]
        threshold, scalebar_um = PersistenceManager.load_settings(stack_path, self.current_stack_index)
        if threshold is not None:
            self.threshold_SpinBox.setValue(threshold)
        if scalebar_um is not None:
            self.doubleSpinBox_2.setValue(scalebar_um)

    

    def generate_annotations(self, probs, threshold):
        return (probs > threshold).astype(int)

    def prepare_unet_input(self, slice_data):
        """Prepare inputs for unet with padding if necessary."""
        original_width = slice_data.shape[1]  #  slice_data is [height, width]
        if original_width < 1536:
            pad_left = (1536 - original_width) // 2
            pad_right = 1536 - original_width - pad_left
            # pad slice_data (2D: height, width)
            slice_data_padded = npy.pad(slice_data, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
            
        else:
            slice_data_padded = slice_data
        
        # prepare inputs as NumPy arrays
        slice_input = npy.expand_dims(npy.expand_dims(slice_data_padded, 0), 0).astype(npy.float32)[:, :, :250, :]  # [1, 1, height, 1536]
        # stat_input = npy.expand_dims(stat_probs_padded, 0).astype(npy.float32)  # [1, 1536] ## not used with the unet- keep for potential future extension
        return slice_input
    
    def _predict_single_slice(self, stack_index, slice_index):
        # check if required models and data are loaded
        if not self.data.stacks or not self.data.unet_models:
            print("Required models or data not loaded.")
            return
        
        # get the slice data
        slice_data = self.data.get_slice(stack_index, slice_index)
        if slice_data is None:
            return
        
        # get the reduced slice data
        reduced_slice_data = self.data.get_reduced_slice(stack_index, slice_index)
        if reduced_slice_data is None:
            print(f"Reduced slice not available for slice {slice_index}.")
            return
        
        if not self.data.unet_models:
            print("No UNet models loaded.")
            return

        original_width = slice_data.shape[1]  # get original width
        pad_left = (1536 - original_width) // 2 if original_width < 1536 else 0

        print(f'Loading stats for slice {slice_index}')
        
        
        
        # unet prediction with multiple models
        print('Predicting from UNet')
        slice_input = self.prepare_unet_input(reduced_slice_data)
        all_probs = []
        for model in self.data.unet_models:
            probs = model.run(None, {'image': slice_input})[0]
            all_probs.append(probs.squeeze())  # shape: [height, width]
        if all_probs:
            averaged_probs = npy.mean(all_probs, axis=0)  # average across models
            unet_probs = averaged_probs  # shape: [height, width]
            
        else:
            print("No predictions from models.")
            return
        
        # reduce to 1D probabilities and annotations
        self.column_probability_option()
        if self.column_probability_method == 'max':
            probabilities_1d = npy.max(unet_probs, axis=0)  # shape: [width]
            print(self.column_probability_method)
        if self.column_probability_method == 'p99':
            probabilities_1d = npy.percentile(unet_probs, 99, axis=0)  # shape: [width]
            print(self.column_probability_method)
        if original_width < 1536:
            probabilities_1d = probabilities_1d[pad_left:pad_left + original_width]
        threshold = self.threshold_SpinBox.value()
        annotations_1d = (probabilities_1d > threshold).astype(int)  # shape: [width]
        total_slices = self.data.stacks[self.current_stack_index].shape[0]
        self.log_action(f'model prediction; slice: {slice_index+1}/{total_slices}, threshold: {threshold}')
        print(f'probabilities_1d average: {npy.average(probabilities_1d)}')
        
        
        # store the results
        if stack_index not in self.data.probabilities:
            self.data.probabilities[stack_index] = {}
        self.data.probabilities[stack_index][slice_index] = probabilities_1d
        
        if stack_index not in self.data.annotations:
            self.data.annotations[stack_index] = {}
        self.data.annotations[stack_index][slice_index] = annotations_1d

        # update arrays
        self.data.probabilities_array[stack_index][slice_index, :] = probabilities_1d
        self.data.annotations_array[stack_index][slice_index, :] = annotations_1d
        
        # update dictionaries
        self.data.probabilities[stack_index][slice_index] = probabilities_1d
        self.data.annotations[stack_index][slice_index] = annotations_1d
        
        # save after prediction
        stack_path = self.data.stack_paths[stack_index]
        self.data.save_stack_data(stack_index, stack_path)


    def predict_slice(self):
        if self.data.unet_models:
            pass
        else:
            # go fer default locations
            self.default_model_locations()
        
        self._predict_single_slice(self.current_stack_index, self.current_slice_index)
        self.update_overlays()
        self.update_fundus_view()

    def predict_stack(self):
        if self.data.unet_models:
            pass
        else:
            # go fer default locations
            self.default_model_locations()
        
        # check if everything’s loaded
        if not self.data.stacks or not self.data.unet_models:
            print("Required models or data not loaded.")
            return
        
        # disable the button to prevent multiple runs
        self.run_model_on_stack.setEnabled(False)  # disable run button
        self.pushButton_5.setEnabled(True)    # enable stop button
        self.stop_flag = False          # reset stop flag

        # create and start the worker
        self.active_worker = PredictionWorker(self, self.current_stack_index)
        self.active_worker.prediction_complete.connect(self.on_prediction_complete)
        self.active_worker.progress_update.connect(self.update_progress)
        self.active_worker.finished.connect(self.on_worker_finished)
        self.active_worker.start()

    def on_stack_processed(self, stack_index):
        # save the current stack index
        original_stack_index = self.current_stack_index
        # set the current stack to the processed one
        self.current_stack_index = stack_index
        self.load_fundus_image()
        self.update_fundus_view()
        self.export_fundus_svg()
        # restore the original stack index
        self.current_stack_index = original_stack_index
        self.load_fundus_image()
        self.update_fundus_view()

    def on_prediction_complete(self):
        # reset progress bar and update UI
        self.processing_progressbar.setValue(0)
        self.update_overlays()
        self.update_fundus_view()
        self.run_model_on_stack.setEnabled(True)  # re-enable the button
        print("Stack prediction complete.")

    def on_batch_process_all(self):
        if not hasattr(self, 'base_dir') or not self.base_dir:
            print("No base directory selected.")
            return

        if self.data.unet_models:
            pass
        else:
            # go fer default locations
            self.default_model_locations()


        # get all subfolders in the base directory
        subfolders = [os.path.join(self.base_dir, d) for d in os.listdir(self.base_dir) 
                    if os.path.isdir(os.path.join(self.base_dir, d))]
        
        # define required files for a valid stack folder
        required_files = ['grey_oct.npz', 'green_frame.npz']
        
        # filter valid folders
        valid_folders = [folder for folder in subfolders 
                        if all(os.path.exists(os.path.join(folder, f)) for f in required_files)]
        
        if not valid_folders:
            print("No valid folders found for batch processing.")
            return
        
        # clear and update the listbox with all valid folders
        self.folders_list.clear()
        for folder in valid_folders:
            item_name = os.path.basename(folder)
            list_item = QListWidgetItem(item_name)
            list_item.setData(Qt.UserRole, folder)
            self.folders_list.addItem(list_item)
        
        # load all stacks at once
        self.data.load_stacks(valid_folders)
        
        # start the batch processing worker
        self.active_worker = BatchPredictionWorker(self)
        self.active_worker.progress_update.connect(self.update_batch_progress)
        self.active_worker.stack_processed.connect(self.on_stack_processed)
        self.active_worker.finished.connect(self.on_batch_process_finished)
        self.active_worker.start()

    def update_progress(self, value):
        # update the existing progress bar
        self.processing_progressbar.setValue(value)

    def update_batch_progress(self, value):
        self.processing_progressbar.setValue(value)

    def on_batch_process_finished(self):
        self.processing_progressbar.setValue(100)
        self.batch_process_all.setEnabled(True)
        print("Batch processing complete.")

    def on_worker_finished(self):
        self.stop_flag = False
        self.active_worker = None
        self.processing_progressbar.setValue(0)
        self.run_model_on_stack.setEnabled(True)   # re-enable single stack start
        self.batch_process_all.setEnabled(True)    # re-enable batch start
        self.pushButton_5.setEnabled(False)        # disable stop button

    def update_scalebar(self):
        if self.data.stacks and self.current_stack_index < len(self.data.stacks):
            stack_path = self.data.stack_paths[self.current_stack_index]
            PersistenceManager.save_settings(
                stack_path,
                self.current_stack_index,
                self.threshold,
                self.doubleSpinBox_2.value()
            )
            # self.data.fundus_scalebar_um[self.current_stack_index] = self.doubleSpinBox_2.value()

    def calculate_scalebar(self):
        px_length, um_per_px = self.data.get_fundus_scalebar_info(self.current_stack_index)
        if px_length is None or um_per_px is None:
            um_length = self.doubleSpinBox_2.value()
            self.data.measure_scalebar_length(self.current_stack_index, um_length)
            px_length, um_per_px = self.data.get_fundus_scalebar_info(self.current_stack_index)
    
    def update_statistics(self):
        """Update the statistics display with the annotated area in μm²."""
        if not self.data.stacks:
            self.plainTextEdit.setPlainText("No stack loaded.")
            return
        
        # calculate area in square pixels
        area_px, region_areas = self.graphicsView_2.calculate_annotated_area()
        
        # get scale factor (μm per pixel)
        _, um_per_px = self.data.get_fundus_scalebar_info(self.current_stack_index)

        if um_per_px is None:
            if self.doubleSpinBox_2.value() > 0:
                self.calculate_scalebar()
        
        if um_per_px is not None:
            # convert to square micrometers
            area_um2 = area_px * (um_per_px ** 2)
            # format the text (keeping slice stats as placeholders for now)
            num_regions = len(region_areas)
            region_areas_um = [f"{area* (um_per_px ** 2):.2f} μm²" for area in region_areas]
            breakdown = "\n".join([f"  {area}" for area in region_areas_um])
            text = (
                f"{num_regions} regions\n"
                f"area (μm²): {area_um2:.2f}\n"
                f"breakdown:\n"
                f"{breakdown}\n"
            )
            self.plainTextEdit.setPlainText(text)
        else:
            self.plainTextEdit.setPlainText("Scale bar information not set.\nPlease set fundus scale (μm) in 'scale bars'.")


    def update_display(self, state):
        self.update_overlays()

    def export_fundus_svg(self):
        if self.data.stacks:
            # generate the filename for the SVG
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            stack_name = self.data.stack_names[self.current_stack_index]
            base_name = f"fundus_{stack_name}"
            if self.overwrite_saved_files.isChecked():
                fname = f"{base_name}.svg"
            else:
                fname = f"{base_name}_{timestamp}.svg"
            full_path = os.path.join(self.log_dir, fname)
            
            # export the SVG
            generator = QSvgGenerator()
            generator.setFileName(full_path)
            generator.setSize(self.graphicsView_2.scene.sceneRect().size().toSize())
            generator.setViewBox(self.graphicsView_2.scene.sceneRect())
            painter = QPainter(generator)
            self.graphicsView_2.scene.render(painter)
            painter.end()
            
            # log the export action
            self.log_action(f"Exported fundus SVG to '{full_path}'")
            
            # log statistics with the SVG filename
            self.log_statistics(fname)

    def export_slice_svg(self):
        if self.data.stacks:
            generator = QSvgGenerator()
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            stack_name = self.data.stack_names[self.current_stack_index]
            slice_index = self.current_slice_index
            base_name = f"slice_{stack_name}_frame_{slice_index}"
            if self.overwrite_saved_files.isChecked():
                fname = f"{base_name}.svg"
            else:
                fname = f"{base_name}_{timestamp}.svg"
            full_path = os.path.join(self.log_dir, fname)
            
            generator.setFileName(full_path)
            generator.setSize(self.slice_scene.sceneRect().size().toSize())
            generator.setViewBox(self.slice_scene.sceneRect())
            painter = QPainter(generator)
            self.slice_scene.render(painter)
            painter.end()
    

    def set_log_location(self):
        self.log_dir = QFileDialog.getExistingDirectory(self, "Select Log Directory")
        if self.log_dir:
            log_path_file = os.path.join(self.base_dir, 'log_path.txt')
            with open(log_path_file, 'w') as f:
                f.write(self.log_dir)
            log_path = os.path.join(self.log_dir, "session.log")
            if not os.path.exists(log_path):
                with open(log_path, "w", encoding='utf-8') as f:
                    f.write(f"--- Session Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            else:
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"\n--- Session Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            self.log_action(f"Log location set to '{self.log_dir}'")

    def log_action(self, message):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} - {message}"
            with open(os.path.join(self.log_dir, "session.log"), "a", encoding='utf-8') as f:
                f.write(log_entry + "\n")
            # self.log_text.appendPlainText(log_entry)
        except:
            print('Failed to log action')

    def append_stack_note(self):
        note = self.log_text.toPlainText().strip()
        if note:
            self.log_action(f"User Note for stack '{self.data.stack_names[self.current_stack_index]}': {note}")
            self.log_text.clear()

    def save_note(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # current time
        if self.data.stacks:
            stack_name = self.data.stack_names[self.current_stack_index]
        else:
            stack_name = "no_folder_loaded"
        note = self.log_text.toPlainText().strip()
        entry = f"{timestamp} - {stack_name}: {note}"
        
        # add to main log
        if self.log_dir:
            with open(os.path.join(self.log_dir, "session.log"), "a", encoding='utf-8') as log:
                log.write(f"[NOTE] {entry}\n")
            
            # add to notes file
            with open(os.path.join(self.log_dir,"notes.txt"), "a", encoding='utf-8') as notes:
                notes.write(f"{entry}\n")
            self.log_text.clear()
        else:
            print("No log location set, cannot save notes.")

    def log_statistics(self, svg_filename):
        # calculate area statistics
        area_px, region_areas = self.graphicsView_2.calculate_annotated_area()
        _, um_per_px = self.data.get_fundus_scalebar_info(self.current_stack_index)
        csv_path = os.path.join(self.log_dir, "statistics.csv")
        
        # prepare the row data
        if um_per_px:
            area_um2 = area_px * (um_per_px ** 2)
            region_areas_um2 = [area * (um_per_px ** 2) for area in region_areas]
            model_names = str(self.data.model_names) if self.data.unet_models else "manual"
            row = [
                self.data.stack_names[self.current_stack_index],
                model_names,
                f"{self.threshold_SpinBox.value():.4f}",
                f"{area_um2:.2f}",
                len(region_areas),
                ",".join(f"{area:.2f}" for area in region_areas_um2),
                svg_filename  # include the SVG filename
            ]
            
            # write to CSV
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if os.path.getsize(csv_path) == 0:  # write header if file is new
                    writer.writerow(["Stack Name", "Model Name(s)", "Threshold", "Area (μm²)", "Number of Regions", "Region Areas", "SVG Filename"])
                writer.writerow(row)
            
            self.log_action(f"Logged statistics for stack '{row[0]}' to CSV with SVG '{svg_filename}'")

    def open_timeline(self):
        self.timeline_window = TimelineGUI(self)
        self.timeline_window.show()
    
    def generate_fresh_fundus_pixmap(self, stack_index):
        """Generate a fresh fundus pixmap with annotations for a given stack index."""
        # get fundus image path
        folder = self.data.stack_paths[stack_index]
        fundus_path = os.path.join(folder, 'averaged_frame.png')

        if not os.path.exists(fundus_path):
            return QPixmap()

        # create a temporary FundusView to render the thumbnail
        temp_fundus_view = FundusView(self)
        temp_fundus_view.main_window = self  # link back to the main window

        # set the fundus image
        pixmap = QPixmap(fundus_path)
        if pixmap.isNull():
            return QPixmap()
        temp_fundus_view.set_fundus_image(pixmap)

        # set overlays to enable proper annotation scaling (x_min/x_max)
        bounding_box, frame_edge = self.data.get_fundus_overlays(stack_index)
        temp_fundus_view.set_fundus_overlays(bounding_box, frame_edge)

        # update annotations for the specific stack
        temp_fundus_view.update_annotations(stack_index=stack_index)

        temp_fundus_view.update_annotated_areas(stack_index=stack_index)

        # render the scene to a new pixmap
        thumbnail = QPixmap(temp_fundus_view.scene.sceneRect().size().toSize())
        thumbnail.fill(Qt.transparent)
        painter = QPainter(thumbnail)
        temp_fundus_view.scene.render(painter)
        painter.end()

        return thumbnail
    
    def jump_to_stack(self, stack_index):
        print(f"Timeline jump to stack {self.data.stack_names[stack_index]}")
        if self.current_stack_index >= 0:
            self.current_stack_index = stack_index
            self.current_slice_index = 0
            total_slices = self.data.stacks[self.current_stack_index].shape[0]
            self.horizontalSlider.setMaximum(total_slices - 1)
            self.horizontalSlider.setValue(0)
            self.log_action(f'active stack: {self.data.stack_names[self.current_stack_index]}')
            self.load_settings()
            self.update_slice_display()
            self.load_fundus_image()
            self.update_ui_states()
            self.update_display(None)
            self.log_action(f"Timeline jump to stack {self.data.stack_names[stack_index]}.")
            self.setWindowTitle(f"OCTOPUS : {self.data.stack_names[self.current_stack_index]}")

    def column_probability_option(self):
        if self.max_p.isChecked():
            self.column_probability_method = "max"
            self.log_action("Column probability method set to max.")
            self.update_display(None)
        if self.p99.isChecked():
            self.column_probability_method = "p99"
            self.log_action("Column probability method set to 99th percentile.")
            self.update_display(None)
        

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    import qdarkstyle

    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = oct_main()
    base_width, base_height = 1600 * 1, 960 * 1
    screen = app.primaryScreen()
    screen_geometry = screen.availableGeometry()
    window.resize(min(int(base_width), screen_geometry.width()), min(int(base_height), screen_geometry.height()))
    window.show()
    sys.exit(app.exec_())