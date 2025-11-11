from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem, QGraphicsRectItem, QGraphicsPolygonItem, QGraphicsItemGroup
from PyQt5.QtGui import QImage, QPixmap, QPen, QColor, QBrush, QPolygonF, QPainterPath, QPainter
from PyQt5.QtCore import Qt, QPointF
import numpy as npy
from skimage.measure import label, regionprops
from skimage.draw import polygon

class FundusView(QGraphicsView):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window  # reference to RetinaThingy for syncing
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.fundus_pixmap_item = None  # Background image
        self.green_box_item = None  # static green box
        self.slice_arrows = []  # list of slice position arrows/lines
        self.current_slice_line = None  # dynamic line for current slice
        self.annotation_items = []  # overlay for projected annotations
        self.x_min = 0  # store green frame bounds for annotation scaling
        self.x_max = 0
        self.annotated_area_items = []  # list to store polygon items
        self.overlays_group = QGraphicsItemGroup()
        self.scene.addItem(self.overlays_group)

    def set_fundus_image(self, pixmap):
        """Set the fundus image as the background pixmap."""
        if self.fundus_pixmap_item:
            if self.fundus_pixmap_item.scene() == self.scene:
                self.scene.removeItem(self.fundus_pixmap_item)
        if pixmap and not pixmap.isNull():  # check if pixmap is valid
            self.fundus_pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.fundus_pixmap_item)
            self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        else:
            self.fundus_pixmap_item = None

    def set_fundus_overlays(self, bounding_box, frame_edge):
        """Draw the faint green bounding box and fainter slice position lines."""
        self.overlays_group.setZValue(1)
        # clear existing items
        if self.green_box_item:
            if self.fundus_pixmap_item.scene() == self.scene:
                self.scene.removeItem(self.green_box_item)
        for arrow in self.slice_arrows:
            if arrow.scene() == self.scene:
                self.scene.removeItem(arrow)
        self.slice_arrows = []

        # draw faint green rectangular frame (bounding box)
        self.x_min, y_min, self.x_max, y_max = bounding_box
        self.green_box_item = QGraphicsRectItem(self.x_min, y_min, self.x_max - self.x_min, y_max - y_min)
        self.green_box_item.setPen(QPen(QColor(0, 255, 0, 100), 2))
        self.overlays_group.addToGroup(self.green_box_item)
        # self.scene.addItem(self.green_box_item)

        # draw even fainter green horizontal lines for each slice
        for y_pos, x_start, x_end in frame_edge:
            arrow = QGraphicsLineItem(x_start, y_pos, x_end, y_pos)
            arrow.setPen(QPen(QColor(0, 255, 0, 48), 1))  # very faint green
            # self.scene.addItem(arrow)
            self.overlays_group.addToGroup(arrow)
            self.slice_arrows.append(arrow)
        self.overlays_group.setVisible(True)

    def set_green_box_and_arrows(self, green_box_coords, arrow_y_positions):
        self.overlays_group.setVisible(True)
        """Draw the static green box and slice position arrows."""
        # clear existing items
        if self.green_box_item:
            # self.scene.removeItem(self.green_box_item)
            if self.green_box_item.scene() == self.scene:
                self.scene.removeItem(self.green_box_item)
        for arrow in self.slice_arrows:
            if arrow.scene() == self.scene:
                self.scene.removeItem(arrow)
            # self.scene.removeItem(arrow)
        self.slice_arrows = []

        # green box
        x_min, y_min, x_max, y_max = green_box_coords
        self.green_box_item = QGraphicsRectItem(x_min, y_min, x_max - x_min, y_max - y_min)
        self.green_box_item.setPen(QPen(QColor("green"), 2))
        # self.scene.addItem(self.green_box_item)
        self.overlays_group.addToGroup(self.green_box_item)

        # horizontal arrows/lines for each slice
        for y in arrow_y_positions:
            arrow = QGraphicsLineItem(x_min, y, x_max, y)
            arrow.setPen(QPen(QColor("green"), 1))
            # self.scene.addItem(arrow)
            self.overlays_group.addToGroup(arrow)
            self.slice_arrows.append(arrow)

    def update_slice_line(self, y_position):
        self.overlays_group.setVisible(True)
        """Update the yellow line showing the current slice position."""
        if self.current_slice_line:
            if self.current_slice_line.scene() == self.scene:
                self.scene.removeItem(self.current_slice_line)
        width = self.scene.width()
        self.current_slice_line = QGraphicsLineItem(0, y_position, width, y_position)
        self.current_slice_line.setPen(QPen(QColor(192, 192, 0, 100), 2))
        # self.scene.addItem(self.current_slice_line)
        self.overlays_group.addToGroup(self.current_slice_line)

    def update_annotations(self, stack_index=None):
        self.overlays_group.setVisible(True)
        """Project 1D slice annotations for all slices in the stack onto the fundus image."""
        for item in self.annotation_items:
            if item.scene() == self.scene:
                self.scene.removeItem(item)
            # self.overlays_group.removeItem(item)
        self.annotation_items = []

        if not self.main_window or not self.main_window.data.stacks:
            return

        if stack_index is None:
            stack_index = self.main_window.current_stack_index

        slice_pos_y = self.main_window.data.slice_pos_y.get(stack_index, [])
        if not self.main_window.data.stacks or stack_index >= len(self.main_window.data.stacks):
            return
        total_slices = self.main_window.data.stacks[stack_index].shape[0]

        for slice_index in range(total_slices):
            annotations = self.main_window.data.get_annotations(stack_index, slice_index)
            if annotations is None:
                continue
            y_pos = slice_pos_y[slice_index] if slice_index < len(slice_pos_y) else 0
            slice_width = len(annotations)
            if slice_width == 0 or self.x_max == self.x_min:
                continue  # avoid division by zero or invalid scaling
            start = None
            for x, val in enumerate(annotations):
                if val != 0 and start is None:
                    start = x
                elif val == 0 and start is not None:
                    scaled_start = int(npy.round(self.x_min + (start / slice_width) * (self.x_max - self.x_min)))
                    scaled_end = int(npy.round(self.x_min + (x / slice_width) * (self.x_max - self.x_min)))
                    rect = QGraphicsRectItem(scaled_start, y_pos - 1, scaled_end - scaled_start, 2)
                    rect.setBrush(QBrush(QColor(0, 64, 192, 64)))
                    # self.scene.addItem(rect)
                    self.overlays_group.addToGroup(rect)
                    self.annotation_items.append(rect)
                    start = None
            if start is not None:
                scaled_start = int(npy.round(self.x_min + (start / slice_width) * (self.x_max - self.x_min)))
                scaled_end = int(npy.round(self.x_min + (slice_width / slice_width) * (self.x_max - self.x_min)))
                rect = QGraphicsRectItem(scaled_start, y_pos - 1, scaled_end - scaled_start, 2)
                rect.setBrush(QBrush(QColor(0, 64, 192, 64)))
                # self.scene.addItem(rect)
                self.overlays_group.addToGroup(rect)
                self.annotation_items.append(rect)

    def update_annotated_areas(self, stack_index=None):
        self.overlays_group.setVisible(True)
        # clear existing items
        for item in self.annotated_area_items:
            if item.scene() == self.scene:
                self.scene.removeItem(item)
        self.annotated_area_items.clear()

        for item in self.overlays_group.childItems():
            if item != self.green_box_item and item not in self.slice_arrows and item != self.current_slice_line:
                if item.scene() == self.scene:
                    self.scene.removeItem(item)

        if not self.main_window or not self.main_window.data.stacks:
            return

        if stack_index is None:
            stack_index = self.main_window.current_stack_index
            
        slice_pos_y = self.main_window.data.slice_pos_y.get(stack_index, [])
        total_slices = self.main_window.data.stacks[stack_index].shape[0]

        # collect annotation segments per slice
        segments_per_slice = [[] for _ in range(total_slices)]
        for slice_index in range(total_slices):
            annotations = self.main_window.data.get_annotations(stack_index, slice_index)
            if annotations is None or not npy.any(annotations != 0):
                continue
            y_pos = slice_pos_y[slice_index]
            slice_width = len(annotations)
            start = None
            for x, val in enumerate(annotations):
                if val != 0 and start is None:
                    start = x
                elif val == 0 and start is not None:
                    scaled_start = int(self.x_min + (start / slice_width) * (self.x_max - self.x_min))
                    scaled_end = int(self.x_min + ((x - 1) / slice_width) * (self.x_max - self.x_min))
                    # clamp coordinates to valid range
                    scaled_start = max(self.x_min, min(self.x_max, scaled_start))
                    scaled_end = max(self.x_min, min(self.x_max, scaled_end))
                    segments_per_slice[slice_index].append((scaled_start, scaled_end, y_pos))
                    start = None
            if start is not None:
                scaled_start = int(self.x_min + (start / slice_width) * (self.x_max - self.x_min))
                scaled_end = int(self.x_min + ((slice_width - 1) / slice_width) * (self.x_max - self.x_min))
                scaled_start = max(self.x_min, min(self.x_max, scaled_start))
                scaled_end = max(self.x_min, min(self.x_max, scaled_end))
                segments_per_slice[slice_index].append((scaled_start, scaled_end, y_pos))

        # sort slices by y-position (assuming y increases downward)
        sorted_slice_indices = sorted(range(total_slices), key=lambda i: slice_pos_y[i])

        # connect overlapping annotations between consecutive slices
        for i in range(len(sorted_slice_indices) - 1):
            slice_index1 = sorted_slice_indices[i]
            slice_index2 = sorted_slice_indices[i + 1]
            y1 = slice_pos_y[slice_index1]
            y2 = slice_pos_y[slice_index2]
            segments1 = segments_per_slice[slice_index1]
            segments2 = segments_per_slice[slice_index2]
            for seg1 in segments1:
                start1, end1, _ = seg1
                for seg2 in segments2:
                    start2, end2, _ = seg2
                    overlap_start = max(start1, start2)
                    overlap_end = min(end1, end2)
                    if overlap_start <= overlap_end:
                        polygon_points = [
                            (start1, y1),
                            (end1, y1),
                            (end2, y2),
                            (start2, y2),
                            (start1, y1)
                        ]
                        # ensure polygon points are within bounds
                        polygon = QPolygonF([QPointF(max(self.x_min, min(self.x_max, x)), y) 
                                            for x, y in polygon_points])
                        polygon_item = QGraphicsPolygonItem(polygon)
                        polygon_item.setBrush(QBrush(QColor(0, 0, 255, 40)))
                        polygon_item.setPen(QPen(Qt.NoPen))
                        self.overlays_group.addToGroup(polygon_item)
                        self.annotated_area_items.append(polygon_item)

        # display single-slice annotations
        for slice_index in range(total_slices):
            segments = segments_per_slice[slice_index]
            if segments:
                y_pos = slice_pos_y[slice_index]
                for seg in segments:
                    x_start, x_end, _ = seg
                    rect = QGraphicsRectItem(x_start, y_pos - 1, x_end - x_start, 2)
                    rect.setBrush(QBrush(QColor(0, 0, 128, 80)))
                    rect.setPen(QPen(Qt.NoPen))
                    self.overlays_group.addToGroup(rect)
                    self.annotated_area_items.append(rect)

    def calculate_annotated_area(self):
        """Calculate the total area and individual areas of annotated regions."""
        self.overlays_group.setVisible(True)
        if not self.annotated_area_items:
            # print("No annotated area items found.")
            return 0.0, []

        # get scene dimensions
        scene_rect = self.scene.sceneRect()
        width = int(scene_rect.width())
        height = int(scene_rect.height())

        # create a binary mask
        mask = npy.zeros((height, width), dtype=npy.uint8)

        # fill the mask with annotated areas
        for item in self.annotated_area_items:
            if isinstance(item, QGraphicsRectItem):
                rect = item.rect()
                x1 = max(0, int(rect.left()))    # ensure within bounds
                y1 = max(0, int(rect.top()))
                x2 = min(width, int(rect.right()))
                y2 = min(height, int(rect.bottom()))
                mask[y1:y2, x1:x2] = 1
            elif isinstance(item, QGraphicsPolygonItem):
                poly = item.polygon()
                # convert QPointF to (row, col) for skimage
                points = [(p.y(), p.x()) for p in poly]
                rr, cc = polygon([p[0] for p in points], [p[1] for p in points], shape=mask.shape)
                mask[rr, cc] = 1


        # check if the mask has any annotated pixels
        if npy.sum(mask) == 0:
            print("No annotated areas found in the mask.")
            return 0.0, []

        # label connected components
        labeled_mask, num_regions = label(mask, return_num=True, connectivity=2)  # 8-connectivity
        
        # if no regions were labeled, return early
        if num_regions == 0:
            print("No regions detected after labeling.")
            return 0.0, []

        regions = regionprops(labeled_mask)

        # calculate areas
        total_area = 0.0
        areas = []
        for region in regions:
            area = region.area  # number of pixels in the region
            total_area += area
            areas.append(area)

        # print(f"Total area: {total_area}, Number of regions: {num_regions}")
        return total_area, areas

    def calculate_polygon_area(self, polygon):
        """Calculate the area of a QPolygonF using the shoelace formula."""
        area = 0.0
        n = len(polygon)
        if n < 3:
            return 0.0  # not a valid polygon
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i].x() * polygon[j].y()
            area -= polygon[j].x() * polygon[i].y()
        return abs(area) / 2.0

    def mouseDoubleClickEvent(self, event):
        """Toggle overlay visibility on double-click."""
        self.overlays_group.setVisible(not self.overlays_group.isVisible())
        super().mouseDoubleClickEvent(event)