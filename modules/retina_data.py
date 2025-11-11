import numpy as npy
import os
from PyQt5.QtGui import QPixmap
import cv2
from modules.persistence import PersistenceManager
from datetime import datetime
import re
from PyQt5.QtWidgets import QInputDialog

class RetinaData:
    def __init__(self):
        # data storage
        self.stack_names = []  # list of stack names from .avi or folder names
        self.stack_paths = []  # full folder paths
        self.stacks = []  # list of 3D NumPy arrays (the image stacks)
        self.reduced_stacks = []  # list of 3D NumPy arrays (reduced stacks for unet)
        self.stack_index_to_names = {}  # {stack_index: stack_name}
        self.stack_names_to_index = {}  # {stack_name: stack_index}
        self.annotations = {}  # {stack_index: {slice_index: 1D annotation array}}
        self.probabilities = {}  # {stack_index: {slice_index: 1D probability array}}
        self.mlp_model = None  # placeholder for the MLP model
        self.unet_model = None  # placeholder for the unet model
        self.unet_models = []  # list to hold multiple unet models
        self.model_names = []
        self.slice_pos_y = {}
        self.frame_edge = {}
        self.fundus_scalebar_px = {}
        self.fundus_scalebar_um = {}
        self.probabilities_array = {}  # {stack_index: array}
        self.annotations_array = {}    # {stack_index: array}
        self.timelines = {}  # {animal_id_eye: [(date, stack_index), ...]}

    def load_stacks(self, folders):
        """Load stacks and green frame data from folders."""
        self.stacks = []
        self.reduced_stacks = []
        self.stack_names = []
        self.stack_paths = []
        for folder in folders:
            stack_path = os.path.join(folder, 'grey_oct.npz')
            green_path = os.path.join(folder, 'green_frame.npz')
            
            if os.path.exists(stack_path) and os.path.exists(green_path):
                stack_data = npy.load(stack_path)
                self.stacks.append(stack_data['grey'])
                self.stack_names.append(os.path.basename(folder))
                self.stack_paths.append(folder)
                # load reduced stack
                if os.path.exists(stack_path):
                    reduced_data = npy.load(stack_path)
                    self.reduced_stacks.append(reduced_data['grey'])
                else:
                    self.reduced_stacks.append(None)
                    print(f"Warning: grey_oct.npz not found in {folder}.")
                green_data = npy.load(green_path)
                greens = green_data['green_positions']
                greens[:, :20, :60] = 0
                average_green_ys = []
                for s in range(greens.shape[0]):
                    current_green = greens[s] - npy.median(greens, axis=0)
                    green_pixels = npy.where(current_green > 0)
                    average_y = npy.mean(green_pixels[0]) if len(green_pixels[0]) > 0 else 0
                    average_green_ys.append(average_y)
                slice_pos_y = npy.round(average_green_ys).astype(int)
                median_green = npy.median(greens, axis=0)
                binary_frame = median_green[slice_pos_y, :]
                frame_edge = []
                for s, row in enumerate(binary_frame):
                    nonzero = npy.nonzero(row)[0]
                    left_edge = nonzero[0] if len(nonzero) > 0 else 0
                    right_edge = nonzero[-1] if len(nonzero) > 0 else 0
                    frame_edge.append([slice_pos_y[s], left_edge, right_edge])
                frame_edge = npy.array(frame_edge)
                stack_index = len(self.stacks) - 1
                self.slice_pos_y[stack_index] = slice_pos_y
                self.frame_edge[stack_index] = frame_edge

            # parse folder name for animal ID, eye, and date
                match = re.search(r'(.+)_([RL])_(\d{4}-\d{2}-\d{2})', stack_path)
                if match:
                    animal_id, eye, date_str = match.groups()
                    try:
                        date = datetime.strptime(date_str, '%Y-%m-%d')
                    except ValueError:
                        date = datetime.fromtimestamp(os.path.getmtime(folder))
                    timeline_key = f"{animal_id}_{eye}"  # e.g., 83S-Z7245_R
                else:
                    print(f"Could not parse folder name {stack_path}, asking user...")
                    animal_id, ok1 = QInputDialog.getText(None, "Hewwo!", f"What's the animal ID for {stack_path}? ")
                    eye, ok2 = QInputDialog.getItem(None, "Hewwo!", f"Which eye for {stack_path}? ", ["Right", "Left"], 0, False)
                    date_str, ok3 = QInputDialog.getText(None, "Hewwo!", f"What's the date (YYYY-MM-DD) for {stack_path}? ")
                    if ok1 and ok2 and ok3:
                        try:
                            date = datetime.strptime(date_str, '%Y-%m-%d')
                        except ValueError:
                            date = datetime.fromtimestamp(os.path.getmtime(folder))
                        eye = "R" if eye == "Right" else "L"
                        timeline_key = f"{animal_id}_{eye}"
                    else:
                        timeline_key = f"{stack_path}_R"  # fallback
                        date = datetime.fromtimestamp(os.path.getmtime(folder))
                
                if timeline_key not in self.timelines:
                    self.timelines[timeline_key] = []
                self.timelines[timeline_key].append((date, stack_index))
                self.timelines[timeline_key].sort(key=lambda x: x[0])  # sort by date

                num_slices, _, slice_width = stack_data['grey'].shape
                
                # load or initialize arrays
                if PersistenceManager.is_data_valid(folder, stack_index, os.path.join(folder, 'grey_oct.npz')):
                    self.probabilities_array[stack_index], self.annotations_array[stack_index] = \
                        PersistenceManager.load_arrays(folder, stack_index, num_slices, slice_width)
                else:
                    self.probabilities_array[stack_index] = npy.zeros((num_slices, slice_width), dtype='float32')
                    self.annotations_array[stack_index] = npy.zeros((num_slices, slice_width), dtype='uint8')
                
                # sync probabilities dictionary
                self.probabilities[stack_index] = {}
                for slice_index in range(num_slices):
                    if npy.any(self.probabilities_array[stack_index][slice_index, :] != 0):
                        self.probabilities[stack_index][slice_index] = self.probabilities_array[stack_index][slice_index, :]
                
                # sync annotations dictionary
                self.annotations[stack_index] = {}
                for slice_index in range(num_slices):
                    if npy.any(self.annotations_array[stack_index][slice_index, :] != 0):
                        self.annotations[stack_index][slice_index] = self.annotations_array[stack_index][slice_index, :]

        self.stack_index_to_names = {i: name for i, name in enumerate(self.stack_names)}
        self.stack_names_to_index = {name: i for i, name in enumerate(self.stack_names)}
    
    def get_timeline_stats_raw(self, timeline_key):
        """Get raw data for all timepoints of an animal_eye for stats calculation."""
        if timeline_key not in self.timelines:
            return []
        timeline = sorted(self.timelines[timeline_key], key=lambda x: x[0])
        stats_list = []
        for date, stack_index in timeline:
            stack = self.stacks[stack_index]
            stats = {
                'date': date.strftime("%Y-%m-%d"),
                'stack': stack,
                'stack_index': stack_index,
                'annotations': self.annotations.get(stack_index, {}),
                'annotations_array': self.annotations_array.get(stack_index),
                'slice_pos_y': self.slice_pos_y.get(stack_index, []),
                'folder': self.stack_paths[stack_index]
            }
            stats_list.append(stats)
        return stats_list

    def save_stack_data(self, stack_index, stack_path):
        """Save arrays for a stack."""
        PersistenceManager.save_arrays(
            stack_path,
            stack_index,
            self.probabilities_array.get(stack_index),
            self.annotations_array.get(stack_index)
        )

    def get_slice(self, stack_index, slice_index):
        """Get a specific slice from a stack."""
        if stack_index < len(self.stacks) and slice_index < self.stacks[stack_index].shape[0]:
            return self.stacks[stack_index][slice_index]
        return None

    def get_reduced_slice(self, stack_index, slice_index):
        """Get a specific reduced slice from a stack."""
        if stack_index < len(self.reduced_stacks) and self.reduced_stacks[stack_index] is not None:
            if slice_index < self.reduced_stacks[stack_index].shape[0]:
                return self.reduced_stacks[stack_index][slice_index]
        return None

    def get_fundus_image(self, stack_index):
        """Load the averaged_frame.png for the stack using full path."""
        if stack_index < len(self.stack_paths):
            folder = self.stack_paths[stack_index]  # use full path
            fundus_path = os.path.join(folder, 'averaged_frame.png')
            if os.path.exists(fundus_path):
                return QPixmap(fundus_path)
        return QPixmap()  # return empty QPixmap if not found

    def get_fundus_overlays(self, stack_index):
        """Get bounding box and frame edge for fundus overlays."""
        if stack_index in self.frame_edge:
            frame_edge = self.frame_edge[stack_index]
            # filter valid slices where left_edge < right_edge
            valid_mask = frame_edge[:, 1] < frame_edge[:, 2]
            valid_frame_edge = frame_edge[valid_mask]
            
            if valid_frame_edge.size > 0:
                min_y = npy.min(valid_frame_edge[:, 0])
                max_y = npy.max(valid_frame_edge[:, 0])
                min_x = npy.min(valid_frame_edge[:, 1])
                max_x = npy.max(valid_frame_edge[:, 2])
                bounding_box = (min_x, min_y, max_x, max_y)
            else:
                bounding_box = (0, 0, 0, 0)
            return bounding_box, frame_edge
        return (0, 0, 0, 0), npy.array([])

    def add_annotation(self, stack_index, slice_index, start, end, value):
        """Add an annotation to a slice over a range."""
        if stack_index not in self.annotations:
            self.annotations[stack_index] = {}
        if slice_index not in self.annotations[stack_index]:
            self.annotations[stack_index][slice_index] = npy.zeros(self.stacks[stack_index].shape[2])
        self.annotations[stack_index][slice_index][start:end + 1] = value
        # self.annotations_array[stack_index][slice_index, start:end+1] = 2

    def remove_annotation(self, stack_index, slice_index, start, end):
        """Remove an annotation from a slice over a range."""
        if stack_index in self.annotations and slice_index in self.annotations[stack_index]:
            self.annotations[stack_index][slice_index][start:end + 1] = 0
            # self.data.annotations_array[stack_index][slice_index, start:end+1] = 0

    def clear_annotations(self, stack_index, slice_index):
        """Wipe all annotations for a slice."""
        if stack_index in self.annotations and slice_index in self.annotations[stack_index]:
            del self.annotations[stack_index][slice_index]
            # self.data.annotations_array[stack_index][slice_index, :] = 0

    def get_annotations(self, stack_index, slice_index):
        """Get annotations for a slice."""
        return self.annotations.get(stack_index, {}).get(slice_index, None)

    def get_probabilities(self, stack_index, slice_index):
        """Get probabilities for a slice."""
        return self.probabilities.get(stack_index, {}).get(slice_index, None)

    def update_fundus_scalebar(self, stack_index, scalebar_px, scalebar_um):
        """Update the fundus scalebar information."""
        self.fundus_scalebar_px[stack_index] = scalebar_px
        self.fundus_scalebar_um[stack_index] = scalebar_um

    def get_fundus_scalebar_info(self, stack_index):
        px = self.fundus_scalebar_px.get(stack_index, None)
        um = self.fundus_scalebar_um.get(stack_index, None)
        if px is not None and um is not None:
            return px, um
        else:
            return None, None

    def measure_scalebar_length(self, stack_index, user_input_um):
        # load the fundus image
        folder = self.stack_paths[stack_index]
        fundus_path = os.path.join(folder, 'averaged_frame.png')
        if not os.path.exists(fundus_path):
            print(f"Fundus image not found: {fundus_path}")
            return

        # read the image in grayscale
        fundus_clip = cv2.imread(fundus_path, cv2.IMREAD_GRAYSCALE)
        height, width = fundus_clip.shape

        # crop the bottom-left 10% of the image
        crop_height = int(height * 0.1)
        crop_width = int(width * 0.1)
        cropped_clip = fundus_clip[-crop_height:, :crop_width]

        # apply Otsu's thresholding to make it binary
        _, binary_clip = cv2.threshold(cropped_clip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_clip, connectivity=8)

        # exclude background (label 0) and find the largest component
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            print("No components found in the cropped image.")
            return
        max_label = areas.argmax() + 1  # +1 because we skipped label 0

        # get the bounding box dimensions of the largest component
        w = stats[max_label, cv2.CC_STAT_WIDTH]
        h = stats[max_label, cv2.CC_STAT_HEIGHT]

        # calculate the scale bar length as the average of width and height
        scalebar_px = (w + h) / 2.0

        # optional check: warn if width and height differ significantly
        if abs(w - h) / max(w, h) > 0.2:
            print(f"Warning: Detected scale bar dimensions differ significantly (w={w}, h={h}). Detection might be inaccurate.")

        # store the results
        self.fundus_scalebar_px[stack_index] = scalebar_px
        self.fundus_scalebar_um[stack_index] = user_input_um / scalebar_px
        print(f"Scale bar: {scalebar_px} pixels = {user_input_um} μm")
