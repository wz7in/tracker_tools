import sys
from PyQt5.QtCore import QPoint, QTimer, Qt
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                             QLabel, QSlider, QFileDialog, QLineEdit, QHBoxLayout, QFrame, QButtonGroup, QRadioButton, QToolTip)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QMouseEvent
from PyQt5.QtCore import Qt, QRect, QEvent, QPoint


class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player")
        
        # Main layout to contain both video display and the toolbar
        main_layout = QHBoxLayout()
        
        # Video area layout
        video_layout = QVBoxLayout()
        
        # Video display label
        self.video_label = QLabel(self)
        self.video_label.setMouseTracking(True)
        self.video_label.setAlignment(Qt.AlignCenter)  # Center the QLabel
        video_layout.addWidget(self.video_label)
        
        # Progress slider
        self.progress_slider = QSlider(self)
        self.progress_slider.setOrientation(Qt.Horizontal)  # Horizontal
        self.progress_slider.valueChanged.connect(self.seek_video)
        video_layout.addWidget(self.progress_slider)

        # Keyframe indicator bar
        self.keyframe_bar = QLabel(self)
        self.keyframe_bar.setFixedHeight(20)  # Set the height of the keyframe bar
        self.keyframe_bar.setMouseTracking(True)
        self.keyframe_bar.installEventFilter(self)  # Install event filter to handle mouse events
        video_layout.addWidget(self.keyframe_bar)

        # Initialize a list to keep track of keyframes
        self.keyframes = {}
        self.selected_keyframe = None  # Track the selected keyframe for removal
        
        # Dynamic frame position label that floats above the slider
        self.frame_position_label = QLabel(self)
        self.frame_position_label.setStyleSheet("background-color: white;")
        self.frame_position_label.setAlignment(Qt.AlignCenter)
        self.frame_position_label.setFixedSize(100, 20)
        self.frame_position_label.hide()  # Hide initially
        
        # Load video button
        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.load_video)
        video_layout.addWidget(self.load_button)

        # Print frame position button
        self.print_button = QPushButton("Print Frame Position", self)
        self.print_button.clicked.connect(self.print_frame_position)
        video_layout.addWidget(self.print_button)
        
        # Add video layout to the main layout
        main_layout.addLayout(video_layout)
        
        # Separator line between video area and toolbar
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Toolbar layout
        toolbar_layout = QVBoxLayout()

        # Create a horizontal layout for the title and line
        annotation_title_layout = QHBoxLayout()

        # Add a label for the per-frame annotation title
        annotation_title = QLabel("Control Tool Box", self)
        annotation_title.setAlignment(Qt.AlignLeft)  # Left align the title
        annotation_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        annotation_title_layout.addWidget(annotation_title)

        # Add a horizontal line to fill the remaining space
        line = QFrame(self)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: grey;")  # Set the same color as the title
        annotation_title_layout.addWidget(line)

        # Add the horizontal layout to the toolbar layout
        toolbar_layout.addLayout(annotation_title_layout)

        # self.load_button = QPushButton("Load Video", self)
        # self.load_button.clicked.connect(self.load_video)
        # toolbar_layout.addWidget(self.load_button)
        
        # play_button
        self.play_button = QPushButton("Auto Play", self)
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_playback)
        toolbar_layout.addWidget(self.play_button)
        
        # clear_all_button
        self.clear_all_button = QPushButton("Clear Annotations", self)
        self.clear_all_button.clicked.connect(self.clear_annotations)
        toolbar_layout.addWidget(self.clear_all_button)
        
        # remove_last_button
        self.remove_last_button = QPushButton("Remove Last Annotation", self)
        self.remove_last_button.clicked.connect(self.remove_last_annotation)
        toolbar_layout.addWidget(self.remove_last_button)

        # Create a horizontal layout for the title and line
        annotation_title_layout = QHBoxLayout()

        # Add a label for the per-frame annotation title
        annotation_title = QLabel("Per Video Annotation", self)
        annotation_title.setAlignment(Qt.AlignLeft)  # Left align the title
        annotation_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        annotation_title_layout.addWidget(annotation_title)

        # Add a horizontal line to fill the remaining space
        line = QFrame(self)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: grey;")  # Set the same color as the title
        annotation_title_layout.addWidget(line)

        # Add the horizontal layout to the toolbar layout
        toolbar_layout.addLayout(annotation_title_layout)
        
        # Language description input
        self.description_input = QLineEdit(self)
        self.description_input.setPlaceholderText("Enter description...")
        toolbar_layout.addWidget(self.description_input)
        
        # Submit button for language description
        self.submit_description_button = QPushButton("Submit Description", self)
        self.submit_description_button.clicked.connect(self.submit_description)
        toolbar_layout.addWidget(self.submit_description_button)

        # Create a horizontal layout for the title and line
        annotation_title_layout = QHBoxLayout()

        # Add a label for the per-frame annotation title
        annotation_title = QLabel("Per Frame Annotation", self)
        annotation_title.setAlignment(Qt.AlignLeft)  # Left align the title
        annotation_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        annotation_title_layout.addWidget(annotation_title)

        # Add a horizontal line to fill the remaining space
        line = QFrame(self)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: grey;")  # Set the same color as the title
        annotation_title_layout.addWidget(line)

        # Add the horizontal layout to the toolbar layout
        toolbar_layout.addLayout(annotation_title_layout)

        # Keyframe controls
        keyframe_option_layout = QHBoxLayout()
        self.keyframe_button_group = QButtonGroup(self)
        self.start_button = QRadioButton("Start", self)
        self.end_button = QRadioButton("End", self)
        self.keyframe_button_group.addButton(self.start_button)
        self.keyframe_button_group.addButton(self.end_button)
        keyframe_option_layout.addWidget(self.start_button)
        keyframe_option_layout.addWidget(self.end_button)
        toolbar_layout.addLayout(keyframe_option_layout)

        keyframe_button_layout = QHBoxLayout()
        # Mark keyframe button
        self.mark_keyframe_button = QPushButton("Mark Keyframe", self)
        self.mark_keyframe_button.clicked.connect(self.mark_keyframe)
        keyframe_button_layout.addWidget(self.mark_keyframe_button)
        
        # Remove keyframe button
        self.remove_keyframe_button = QPushButton("Remove Keyframe", self)
        self.remove_keyframe_button.clicked.connect(self.remove_keyframe)
        keyframe_button_layout.addWidget(self.remove_keyframe_button)

        # Add the horizontal layout to the toolbar layout
        toolbar_layout.addLayout(keyframe_button_layout)
        
        # Add spacer to push the items to the top
        toolbar_layout.addStretch()
        
        # Add toolbar layout to the main layout
        main_layout.addLayout(toolbar_layout)

        self.setLayout(main_layout)

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
            self.update_keyframe_bar()  # Initialize keyframe bar

    def update_frame(self, frame_number):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                # Resize frame to fit the QLabel while keeping aspect ratio
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.height, self.width, channel = frame.shape

                # Scale the image to fit QLabel
                label_width = self.video_label.width()
                label_height = self.video_label.height()
                self.scale_width = label_width / self.width
                self.scale_height = label_height / self.height
                scale = min(self.scale_width, self.scale_height)
                new_width = int(self.width * scale)
                new_height = int(self.height * scale)

                resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

                bytes_per_line = 3 * new_width
                q_img = QImage(resized_frame.data, new_width, new_height, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_img))
                
                # Update and reposition frame position label
                self.update_frame_position_label()

                self.last_frame = resized_frame

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

    def update_frame_position_label(self):
        # Update the text of the label to show the current frame position
        frame_number = self.progress_slider.value()
        self.frame_position_label.setText(f"Frame: {frame_number}")

        # Calculate the position for the label above the slider handle
        slider_x = self.progress_slider.x()
        slider_width = self.progress_slider.width()
        value_ratio = frame_number / (self.progress_slider.maximum() - self.progress_slider.minimum())
        label_x = slider_x + int(value_ratio * slider_width) - self.frame_position_label.width() // 2
        
        # Set the position of the label
        label_y = self.progress_slider.y() - 30  # Adjust the Y position above the slider
        self.frame_position_label.move(label_x, label_y)
        self.frame_position_label.show()  # Show the label

    def print_frame_position(self):
        current_position = self.progress_slider.value()
        print(f"Current Frame Position: {current_position}")
    
    def play_video(self):
        if self.cap is not None:
            if self.current_frame < self.frame_count - 1:
                self.current_frame += 1
                self.update_frame(self.current_frame)
                self.progress_slider.setValue(self.current_frame)
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
        label_height, label_width = self.video_label.height(), self.video_label.width()

        for point in self.pos_click_position:
            x, y = point.x(), point.y()

            resized_width = int(self.width * min(self.scale_width, self.scale_height))
            resized_height = int(self.height * min(self.scale_width, self.scale_height))

            # Calculate the offsets for centering
            offset_x = (label_width - resized_width) // 2
            offset_y = (label_height - resized_height) // 2

            x -= offset_x
            y -= offset_y
    
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        for point in self.neg_click_position:
            x, y = point.x(), point.y()

            resized_width = int(self.width * min(self.scale_width, self.scale_height))
            resized_height = int(self.height * min(self.scale_width, self.scale_height))

            # Calculate the offsets for centering
            offset_x = (label_width - resized_width) // 2
            offset_y = (label_height - resized_height) // 2

            x -= offset_x
            y -= offset_y
            
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def submit_description(self):
        # Handle the submitted description
        description = self.description_input.text()
        print(f"Submitted Description: {description}")
        # You can add more processing logic for the description here

    def mark_keyframe(self):
        current_frame = self.progress_slider.value()
        if self.start_button.isChecked():
            self.keyframes[current_frame] = 'start'
        elif self.end_button.isChecked():
            self.keyframes[current_frame] = 'end'
        self.update_keyframe_bar()

    def remove_keyframe(self):
        # Remove the selected keyframe if any
        if self.selected_keyframe is not None:
            frame_to_remove = self.selected_keyframe
            if frame_to_remove in self.keyframes:
                del self.keyframes[frame_to_remove]
                self.update_keyframe_bar()
                self.selected_keyframe = None

    def update_keyframe_bar(self):
        # Clear the keyframe bar
        keyframe_image = QImage(self.keyframe_bar.width(), self.keyframe_bar.height(), QImage.Format_RGB32)
        keyframe_image.fill(Qt.white)

        painter = QPainter(keyframe_image)
        for frame, key_type in self.keyframes.items():
            x_position = int((frame / self.frame_count) * self.keyframe_bar.width())
            color = QColor('red') if key_type == 'start' else QColor('blue')
            painter.fillRect(QRect(x_position, 0, 5, self.keyframe_bar.height()), color)
        painter.end()

        # Set the updated image to the QLabel
        self.keyframe_bar.setPixmap(QPixmap.fromImage(keyframe_image))
    
    def eventFilter(self, source, event):
        if source == self.keyframe_bar:
            # Handle mouse move event to show tooltip for keyframes
            if event.type() == QEvent.MouseMove:
                self.show_tooltip(event)
                return True  # Event handled
            
            # Handle mouse button press event to select a keyframe
            elif event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.select_keyframe(event)
                return True  # Event handled
                
        # Call the base class eventFilter for other events
        return super().eventFilter(source, event)

    def show_tooltip(self, event):
        # Show tooltip when hovering over a keyframe
        mouse_pos = event.pos()
        for frame, key_type in self.keyframes.items():
            x_position = int((frame / self.frame_count) * self.keyframe_bar.width())
            if abs(mouse_pos.x() - x_position) <= 5:  # Small range to detect hover
                QToolTip.showText(self.keyframe_bar.mapToGlobal(mouse_pos), f"Frame: {frame}")
                return
        QToolTip.hideText()

    def select_keyframe(self, event):
        # Select a keyframe on click
        mouse_pos = event.pos()
        self.selected_keyframe = None
        for frame, key_type in self.keyframes.items():
            x_position = int((frame / self.frame_count) * self.keyframe_bar.width())
            if abs(mouse_pos.x() - x_position) <= 5:  # Small range to detect click
                self.selected_keyframe = frame
                print(f"Selected Keyframe: Frame {frame}")
                return

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.resize(1000, 600)  # Adjusted size to accommodate the toolbar
    player.show()
    sys.exit(app.exec_())