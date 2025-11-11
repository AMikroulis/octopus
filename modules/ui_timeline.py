from PyQt5 import QtCore, QtGui, QtWidgets

class TimelinePlot(QtWidgets.QWidget):
    """Custom widget to draw a plot for total annotated area over time."""
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.stats_list = []
        self.setMinimumHeight(200)
        self.clicked_index = None
        self.main_window = main_window

    def set_data(self, stats_list):
        self.stats_list = stats_list
        self.update()  # trigger repaint

    def paintEvent(self, event):
        if not self.stats_list:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # dimensions
        width = self.width()
        height = self.height()
        margin = 50  # extra space for axes and labels
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin
        
        # data
        areas = [stats['total_area_um2'] for stats in self.stats_list]
        dates = [stats['date'] for stats in self.stats_list]
        if not areas:
            return
        
        # scale
        max_area = max(areas, default=1)
        min_area = min(areas, default=0)
        if max_area == min_area:
            max_area += 1  # just in case (0)
        num_points = len(areas)
        
        # axes
        painter.setPen(QtGui.QPen(QtGui.QColor("#404040"), 2))
        painter.drawLine(margin, height - margin, margin, margin)  # y-axis
        painter.drawLine(margin, height - margin, width - margin, height - margin)  # x-axis
        
        # labels
        painter.setPen(QtGui.QPen(QtGui.QColor("white"), 2))
        painter.setFont(QtGui.QFont("Arial", 8))
        painter.drawText(60, margin + 10, "Area (μm²)")
        painter.drawText(width - margin - 50, height - margin + 25, "Timepoints")
        
        # y-axis ticks (simple: min, max)
        painter.drawText(5, height - margin + 10, f"{min_area:.0f}")
        painter.drawText(5, margin + 10, f"{max_area:.0f}")
        
        # x-axis ticks (dates, + rotate maybe?)
        for i, date in enumerate(dates):
            x = margin + i * plot_width // (num_points - 1 if num_points > 1 else 1)
            painter.save()
            painter.translate(x-10, height - margin + 20)
            painter.rotate(0)
            painter.drawText(0, 0, date)
            painter.restore()
        
        # draw line and points
        painter.setPen(QtGui.QPen(QtGui.QColor("pink"), 2, QtCore.Qt.DashLine))
        painter.setBrush(QtGui.QBrush(QtGui.QColor("pink")))
        points = []
        for i, area in enumerate(areas):
            x = margin + i * plot_width // (num_points - 1 if num_points > 1 else 1)
            y = height - margin - ((area - min_area) / (max_area - min_area)) * plot_height if max_area != min_area else height - margin
            points.append(QtCore.QPointF(x, y))
            painter.drawEllipse(QtCore.QPointF(x, y), 5, 5)
        
        # connecting lines
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])
        
        painter.end()

    def mousePressEvent(self, event):
        if not self.stats_list:
            return
        margin = 50
        plot_width = self.width() - 2 * margin
        num_points = len(self.stats_list)
        x = event.pos().x()
        # closest point
        for i in range(num_points):
            point_x = margin + i * plot_width // (num_points - 1 if num_points > 1 else 1)
            if abs(x - point_x) < 80:  # click within 80px of a point
                self.clicked_index = i
                print(f"Clicked on point {i}")
                print(f"Jumping to stack {self.stats_list[i]['stack_index']}")
                self.main_window.jump_to_stack(self.stats_list[i]['stack_index'])
                break

class Ui_timeline(object):
    def setupUi(self, timeline_ui):
        timeline_ui.setObjectName("timeline_ui")
        timeline_ui.resize(1200, 600)
        self.central_widget = QtWidgets.QWidget(timeline_ui)
        self.central_widget.setObjectName("central_widget")
        
        # vertical layout
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.main_layout.setObjectName("main_layout")
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(10, 10, 10, 10)  

        # top: selection and generate button
        self.control_group = QtWidgets.QGroupBox(self.central_widget)
        self.control_group.setTitle("Timeline")
        self.control_group.setObjectName("control_group")
        self.control_layout = QtWidgets.QHBoxLayout(self.control_group)  # horizontal alignment
        self.control_layout.setContentsMargins(5, 5, 5, 5)

        self.animal_combo = QtWidgets.QComboBox(self.control_group)
        self.animal_combo.setObjectName("animal_combo")
        self.control_layout.addWidget(self.animal_combo)

        self.generate_timeline_button = QtWidgets.QPushButton(self.control_group)
        self.generate_timeline_button.setText("Generate")
        self.generate_timeline_button.setObjectName("generate_timeline_button")
        self.control_layout.addWidget(self.generate_timeline_button)

        self.export_timeline_button = QtWidgets.QPushButton(self.control_group)
        self.export_timeline_button.setText("Export PNG")
        self.export_timeline_button.setObjectName("export_timeline_button")
        self.control_layout.addWidget(self.export_timeline_button)

        self.main_layout.addWidget(self.control_group)
        
        # middle: grid for thumbnails and stats
        self.scroll_area = QtWidgets.QScrollArea(self.central_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setObjectName("scroll_area")
        self.scroll_content = QtWidgets.QWidget()
        self.scroll_content.setObjectName("scroll_content")
        self.grid_layout = QtWidgets.QGridLayout(self.scroll_content)
        self.grid_layout.setObjectName("grid_layout")
        self.grid_layout.setSpacing(10)
        self.scroll_area.setWidget(self.scroll_content)
        self.main_layout.addWidget(self.scroll_area)
        
        # bottom: plot widget
        self.plot_widget = TimelinePlot(timeline_ui, timeline_ui.main_window)
        self.plot_widget.setObjectName("plot_widget")
        self.main_layout.addWidget(self.plot_widget)
        
        timeline_ui.setCentralWidget(self.central_widget)
        
        self.retranslateUi(timeline_ui)
        QtCore.QMetaObject.connectSlotsByName(timeline_ui)
    
    def retranslateUi(self, timeline_ui):
        _translate = QtCore.QCoreApplication.translate
        timeline_ui.setWindowTitle(_translate("timeline_ui", "Timeline view"))
        self.control_group.setTitle(_translate("timeline_ui", "Timeline"))
        self.generate_timeline_button.setText(_translate("timeline_ui", "Generate"))
        self.export_timeline_button.setText(_translate("timeline_ui", "Export PNG"))