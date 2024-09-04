import sys
from PyQt5.QtCore import QPoint, QTimer, Qt
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QFormLayout,
                             QLabel, QSlider, QFileDialog, QHBoxLayout)
from PyQt5.QtGui import QImage, QMouseEvent, QPixmap

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player")
        
        self.widget = QWidget(self)
        
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)
        
        self.l_layout = QVBoxLayout()
        self.r_layout = QVBoxLayout()
        self.main_layout.addLayout(self.l_layout)
        self.main_layout.addLayout(self.r_layout)
        
        self.video_label = QLabel(self)
        self.video_label.setMouseTracking(True)
        self.l_layout.addWidget(self.video_label)
        self.l_layout.addStretch(1)

        self.progress_slider = QSlider(self)
        self.progress_slider.setOrientation(1)  # Horizontal
        self.progress_slider.valueChanged.connect(self.seek_video)
        self.l_layout.addWidget(self.progress_slider)
        
        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.load_video)
        self.r_layout.addWidget(self.load_button)
        
        self.play_button = QPushButton("Auto Play", self)
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_playback)
        self.r_layout.addWidget(self.play_button)

        self.print_button = QPushButton("Print Frame Position", self)
        self.print_button.clicked.connect(self.print_frame_position)
        self.r_layout.addWidget(self.print_button)
        
        self.clear_all_button = QPushButton("Clear Annotations", self)
        self.clear_all_button.clicked.connect(self.clear_annotations)
        self.r_layout.addWidget(self.clear_all_button)
        
        self.remove_last_button = QPushButton("Remove Last Annotation", self)
        self.remove_last_button.clicked.connect(self.remove_last_annotation)
        self.r_layout.addWidget(self.remove_last_button)
        
        self.r_layout.addStretch(1)

        self.cap = None
        self.frame_count = 0
        self.last_frame = None
        
        self.pos_click_position = []
        self.neg_click_position = []
        self.click_action = []
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_video)

    def clear_annotations(self):
        self.pos_click_position = []
        self.neg_click_position = []
        self.click_action = []
        self.draw_point()
    
    def remove_last_annotation(self):
        if len(self.click_action) > 0 and self.click_action[-1] == 1 and len(self.pos_click_position) > 0:
            self.pos_click_position.pop()
            self.click_action.pop()
        elif len(self.click_action) > 0 and self.click_action[-1] == -1 and len(self.neg_click_position) > 0:
            self.neg_click_position.pop()
            self.click_action.pop()
        self.draw_point()
    
    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Select a Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.progress_slider.setMaximum(self.frame_count - 1)
            self.update_frame(0)

    def update_frame(self, frame_number):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_img))
                self.progress_slider.setValue(frame_number)
                self.last_frame = frame

    def seek_video(self):
        frame_number = self.progress_slider.value()
        self.update_frame(frame_number)
    
    def toggle_playback(self):
        if self.play_button.isChecked():
            self.play_button.setText("Stop Auto Play")
            self.current_frame = self.progress_slider.value()
            self.timer.start(30)  # Set timer to update frame every 30 ms
        else:
            self.play_button.setText("Auto Play")
            self.timer.stop()

    def print_frame_position(self):
        current_position = self.progress_slider.value()
        print(f"Current Frame Position: {current_position}")
    
    def play_video(self):
        if self.cap is not None:
            if self.current_frame < self.frame_count - 1:
                self.current_frame += 1
                self.update_frame(self.current_frame)
            else:
                self.timer.stop()
                self.play_button.setChecked(False)
                self.play_button.setText("Auto Play")
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.last_frame is not None:
            pos = self.video_label.mapFromGlobal(event.globalPos())
            click_position = QPoint(pos.x(), pos.y())
            print(f"Clicked Position: {click_position.x(), click_position.y()}")
            self.pos_click_position.append(click_position)
            self.click_action.append(1)
            # Draw a point on the frame
        elif event.button() == Qt.RightButton and self.last_frame is not None:
            pos = self.video_label.mapFromGlobal(event.globalPos())
            click_position = QPoint(pos.x(), pos.y())
            print(f"Clicked Position: {click_position.x(), click_position.y()}")
            self.neg_click_position.append(click_position)
            self.click_action.append(-1)
        
        self.draw_point()
    
    def draw_point(self):
        frame = self.last_frame.copy()
        for point in self.pos_click_position:
            x, y = point.x(), point.y()
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        for point in self.neg_click_position:
            x, y = point.x(), point.y()
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.resize(800, 600)
    player.show()
    sys.exit(app.exec_())
