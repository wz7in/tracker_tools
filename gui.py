import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                             QLabel, QSlider, QFileDialog)
from PyQt5.QtGui import QImage, QPixmap

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player")
        
        self.layout = QVBoxLayout()
        
        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        self.progress_slider = QSlider(self)
        self.progress_slider.setOrientation(1)  # Horizontal
        self.progress_slider.valueChanged.connect(self.seek_video)
        self.layout.addWidget(self.progress_slider)

        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.load_video)
        self.layout.addWidget(self.load_button)

        self.print_button = QPushButton("Print Frame Position", self)
        self.print_button.clicked.connect(self.print_frame_position)
        self.layout.addWidget(self.print_button)
        
        self.setLayout(self.layout)

        self.cap = None
        self.frame_count = 0

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
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def seek_video(self):
        frame_number = self.progress_slider.value()
        self.update_frame(frame_number)

    def print_frame_position(self):
        current_position = self.progress_slider.value()
        print(f"Current Frame Position: {current_position}")

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
