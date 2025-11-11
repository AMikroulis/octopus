from PyQt5.QtWidgets import QGraphicsView, QGraphicsRectItem
from PyQt5.QtGui import QBrush, QColor

class SliceView(QGraphicsView):
    """Custom QGraphicsView for interactive highlighting."""
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.selection_rect = None # the green rectangle overlay
        self.start_column = 0
        self.main_window = main_window
        self.is_selecting = False  # are we currently selecting?
        
    @property
    def enable_highlighter(self):
        return self.main_window.enable_highlighter if self.main_window else None
    
    def mousePressEvent(self, event):
        if self.enable_highlighter and self.enable_highlighter.isChecked():
            # clear existing selection if any
            if self.selection_rect:
                self.scene().removeItem(self.selection_rect)
                self.selection_rect = None
                self.main_window.selected_range = None
            # start new selection
            pos = self.mapToScene(event.pos())
            self.start_column = int(pos.x())
            height = self.scene().height()
            self.selection_rect = QGraphicsRectItem(self.start_column, 0, 0, height)
            self.selection_rect.setBrush(QBrush(QColor(0, 255, 0, 40)))  # green with transparency
            self.scene().addItem(self.selection_rect)
            self.is_selecting = True

    def mouseMoveEvent(self, event):
        # update the rectangle while dragging, but only if selecting
        if self.is_selecting and self.selection_rect:
            pos = self.mapToScene(event.pos())
            current_column = int(pos.x())
            width = current_column - self.start_column
            self.selection_rect.setRect(min(self.start_column, current_column), 0, abs(width), self.scene().height())

    def mouseReleaseEvent(self, event):
        # finalize the selection but keep the rectangle visible
        if self.is_selecting and self.selection_rect:
            pos = self.mapToScene(event.pos())
            current_column = int(pos.x())
            start = min(self.start_column, current_column)
            end = max(self.start_column, current_column)
            if start < end:  # only set selected range if there's a valid selection
                self.main_window.selected_range = (start, end)
            # even if it's a simple click (start == end), keep the rectangle until the next selection
            self.is_selecting = False  # done selecting, rectangle stays

    def clear_selection_rect(self):
        # clear the selection when called (e.g., after an action)
        if self.selection_rect:
            self.scene().removeItem(self.selection_rect)
            self.selection_rect = None
            self.main_window.selected_range = None