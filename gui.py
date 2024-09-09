import sys
import os
import json
from PyQt5.QtCore import QPoint, QTimer, Qt
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                             QLabel, QSlider, QFileDialog, QLineEdit, QHBoxLayout, QFrame, QButtonGroup, QRadioButton, QToolTip, QComboBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QMouseEvent
from PyQt5.QtCore import Qt, QRect, QEvent, QPoint

import yaml
from client_utils import request_sam, request_cotracker
from cotracker.utils.visualizer import Visualizer


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
        self.frame_position_label.setStyleSheet("background-color: gray;")
        self.frame_position_label.setAlignment(Qt.AlignCenter)
        self.frame_position_label.setFixedSize(100, 20)
        self.frame_position_label.hide()  # Hide initially
        
        # Load video button
        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.load_video)
        video_layout.addWidget(self.load_button)

        # Print frame position button
        self.print_button = QPushButton("Print Frame Position", self)
        self.print_button.clicked.connect(self.get_frame_position)
        video_layout.addWidget(self.print_button)
        
        # Add video layout to the main layout
        main_layout.addLayout(video_layout)
        
        # Separator line between video area and toolbar
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Toolbar layout
        self.toolbar_layout = QVBoxLayout()

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
        self.toolbar_layout.addLayout(annotation_title_layout)

        # self.load_button = QPushButton("Load Video", self)
        # self.load_button.clicked.connect(self.load_video)
        # toolbar_layout.addWidget(self.load_button)
        
        # play_button
        self.play_button = QPushButton("Auto Play", self)
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_playback)
        self.toolbar_layout.addWidget(self.play_button)
        
        # clear_all_button
        self.clear_all_button = QPushButton("Clear Annotations", self)
        self.clear_all_button.clicked.connect(self.clear_annotations)
        self.toolbar_layout.addWidget(self.clear_all_button)
        
        # remove_last_button
        self.remove_last_button = QPushButton("Remove Last Annotation", self)
        self.remove_last_button.clicked.connect(self.remove_last_annotation)
        self.toolbar_layout.addWidget(self.remove_last_button)
        
        # remove_video_button
        self.remove_video_button = QPushButton("Remove Video", self)
        self.remove_video_button.clicked.connect(self.clear_video)
        self.toolbar_layout.addWidget(self.remove_video_button)

        # remove_video_button
        self.load_clip_data_button = QPushButton("Load Clip Preprocess Data", self)
        self.load_clip_data_button.clicked.connect(self.load_clip_data)
        self.toolbar_layout.addWidget(self.load_clip_data_button)

        # visualize layout
        vis_button_layout = QHBoxLayout()

        self.vis_button = QPushButton("Visualize Video", self)
        self.vis_button.clicked.connect(self.load_res)
        vis_button_layout.addWidget(self.vis_button)

        self.vis_ori = QRadioButton("original video", self)
        vis_button_layout.addWidget(self.vis_ori)

        self.vis_sam = QRadioButton("sam result", self)
        vis_button_layout.addWidget(self.vis_sam)

        self.vis_tracker = QRadioButton("track result", self)
        vis_button_layout.addWidget(self.vis_tracker)

        self.toolbar_layout.addLayout(vis_button_layout)


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
        self.toolbar_layout.addLayout(annotation_title_layout)
        
        # Language description input
        self.description_input = QLineEdit(self)
        self.description_input.setPlaceholderText("Enter description...")
        self.toolbar_layout.addWidget(self.description_input)
        
        # Submit button for language description
        self.submit_description_button = QPushButton("Submit Description", self)
        self.submit_description_button.clicked.connect(self.submit_description)
        self.toolbar_layout.addWidget(self.submit_description_button)

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
        self.toolbar_layout.addLayout(annotation_title_layout)

        # Keyframe controls
        keyframe_option_layout = QHBoxLayout()
        self.keyframe_button_group = QButtonGroup(self)
        self.start_button = QRadioButton("Start", self)
        self.end_button = QRadioButton("End", self)
        self.keyframe_button_group.addButton(self.start_button)
        self.keyframe_button_group.addButton(self.end_button)
        keyframe_option_layout.addWidget(self.start_button)
        keyframe_option_layout.addWidget(self.end_button)
        self.toolbar_layout.addLayout(keyframe_option_layout)

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
        self.toolbar_layout.addLayout(keyframe_button_layout)

        function_title_layout = QHBoxLayout()
        function_title = QLabel("Auto Label Tools", self)
        function_title.setAlignment(Qt.AlignLeft)  # Left align the title
        function_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        function_title_layout.addWidget(function_title)
        
        line = QFrame(self)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: grey;")  # Set the same color as the title
        function_title_layout.addWidget(line)
        self.toolbar_layout.addLayout(function_title_layout)
        
        # Anno buttons
        anno_button_layout = QHBoxLayout()
        self.anno_function_select = QComboBox()
        self.anno_function_select.addItem('sam')
        self.anno_function_select.addItem('tracker')
        
        # button params for different functions
        self.button_param_select = QComboBox()
        self.button_param_select.addItem('Frame Mode')
        self.button_param_select.addItem('Video Mode')
        
        self.anno_function_select.currentIndexChanged.connect(self.update_function_select)
    
        click_action_button = QPushButton("Run", self)
        # select color
        click_action_button.clicked.connect(self.get_anno_result)
        anno_button_layout.addWidget(self.anno_function_select)
        anno_button_layout.addWidget(self.button_param_select)
        anno_button_layout.addWidget(click_action_button)
        self.toolbar_layout.addLayout(anno_button_layout)
        

        # Add spacer to push the items to the top
        self.toolbar_layout.addStretch()
        
        # Add toolbar layout to the main layout
        main_layout.addLayout(self.toolbar_layout)

        self.setLayout(main_layout)

        self.cap = None
        self.frame_count = 0
        self.last_frame = None
        self.tracking_points = dict()
        self.tracking_masks = dict()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_video)

        # add config
        config_path = "./config/config.yaml"
        with open(config_path, "r") as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)
        self.sam_config = self.model_config["sam"]
        self.co_tracker_config = self.model_config["cotracker"]

        # initialize self.vis_track_res
        self.vis_track_res = False

    def get_anno_result(self):
        if self.anno_function_select.currentText() == 'sam':
            self.get_sam_result()
        elif self.anno_function_select.currentText() == 'tracker':
            self.get_tap_result()
    
    def update_function_select(self):
        if self.anno_function_select.currentText() == 'sam':
            self.button_param_select.clear()
            self.button_param_select.addItem('Frame Mode')
            self.button_param_select.addItem('Video Mode')
        elif self.anno_function_select.currentText() == 'tracker':
            self.button_param_select.clear()
            self.button_param_select.addItem('Point Mode')
            self.button_param_select.addItem('Mask Mode')
            self.button_param_select.addItem('Grid Mode')
    
    def load_res(self):
        if self.vis_sam.isChecked():
            self.vis_track_res = True
            self.track_res = self.sam_res
        elif self.vis_tracker.isChecked():
            self.vis_track_res = True
            self.track_res = self.tracker_res
        else:
            self.vis_track_res = False

        frame_number = self.progress_slider.value()
        self.update_frame(frame_number)

    def clear_annotations(self):
        self.tracking_points[self.progress_slider.value()]['pos'] = []
        self.tracking_points[self.progress_slider.value()]['raw_pos'] = []
        self.tracking_points[self.progress_slider.value()]['neg'] = []
        self.tracking_points[self.progress_slider.value()]['raw_neg'] = []
        self.tracking_points[self.progress_slider.value()]['labels'] = []
        if self.last_frame is not None:
            self.draw_image()

    def load_clip_data(self):
        video_name = self.video_path.split('/')[-1].split('.')[0]
        clip_data_pth = f'/Users/dingzihan/Documents/projects/tracker_tools/{video_name}.json'
        with open(clip_data_pth, 'r') as f:
            clip_data = json.load(f)

        caption_ls = []
        for clip_id, clip_info in clip_data.items():
            self.keyframes[clip_info['s_e'][0]] = 'start'
            self.keyframes[clip_info['s_e'][1]] = 'end'
            caption_ls.append(clip_info['des'])

        # update keyframe bar
        self.update_keyframe_bar()

        # update caption options
        self.clip_caption_ls = caption_ls

        # create description choice buttons dynamically
        for clip_des_ls in self.clip_caption_ls:
            self.label = QLabel('Please choose the best description:')
            self.toolbar_layout.addWidget(self.label)

            self.des_choice_buttons = []
            
            for des in clip_des_ls:
                choice_button = QRadioButton(des)
                self.toolbar_layout.addWidget(choice_button)
                self.des_choice_buttons.append(choice_button)
           
    def clear_video(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.video_label.clear()
        self.progress_slider.setValue(0)
        self.frame_position_label.hide()
        self.keyframes = {}
        self.selected_keyframe = None
        self.update_keyframe_bar()
        self.tracking_points = dict()
        self.tracking_masks = dict()
        self.last_frame = None
    
    def remove_last_annotation(self):
        click_action = self.tracking_points[self.progress_slider.value()]['labels']
        pos_click_position = self.tracking_points[self.progress_slider.value()]['pos']
        neg_click_position = self.tracking_points[self.progress_slider.value()]['neg']
        
        if len(click_action) > 0 and click_action[-1] == 1 and len(pos_click_position) > 0:
            self.tracking_points[self.progress_slider.value()]['pos'].pop()
            self.tracking_points[self.progress_slider.value()]['raw_pos'].pop()
            self.tracking_points[self.progress_slider.value()]['labels'].pop()
        elif len(click_action) > 0 and click_action[-1] == -1 and len(neg_click_position) > 0:
            self.tracking_points[self.progress_slider.value()]['neg'].pop()
            self.tracking_points[self.progress_slider.value()]['raw_neg'].pop()
            self.tracking_points[self.progress_slider.value()]['labels'].pop()
        if self.last_frame is not None:
            self.draw_image()
    
    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Select a Video", "", "Video Files (*.mp4 *.avi *.mov)")
        self.video_path = video_path
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(self.frame_count):
                self.tracking_points[i] = dict(
                    pos=[], raw_pos=[], neg=[], raw_neg=[], labels=[]
                )
                self.tracking_masks[i] = []
            self.progress_slider.setMaximum(self.frame_count - 1)
            self.update_frame(0)
            self.update_keyframe_bar()  # Initialize keyframe bar

        # check if clip prepocess info is exist, load automatically
        video_name = self.video_path.split('/')[-1].split('.')[0]
        clip_data_pth = f'/Users/dingzihan/Documents/projects/tracker_tools/{video_name}.json'
        if os.path.exists(clip_data_pth):
            self.load_clip_data()
            
    def update_frame(self, frame_number):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                # Resize frame to fit the QLabel while keeping aspect ratio
                if self.vis_track_res:
                    frame = self.track_res[frame_number]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.height, self.width, channel = frame.shape

                # Scale the image to fit QLabel
                label_width = self.video_label.width()
                label_height = self.video_label.height()
                self.scale_width = label_width / self.width
                self.scale_height = label_height / self.height
                self.scale = min(self.scale_width, self.scale_height)
                new_width = int(self.width * self.scale)
                new_height = int(self.height * self.scale)

                resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Update and reposition frame position label
                self.update_frame_position_label()

                self.last_frame = resized_frame
                
                self.draw_image()
                
    def seek_video(self):
        frame_number = self.progress_slider.value()
        self.update_frame(frame_number)
    
    def toggle_playback(self):
        if self.cap is None:
            return
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
        label_x = max(slider_x, min(label_x, slider_x + slider_width - self.frame_position_label.width()))
        self.frame_position_label.move(label_x, label_y)
        self.frame_position_label.show()  # Show the label

    def get_frame_position(self):
        current_position = self.progress_slider.value()
        print(f"Current Frame Position: {current_position}")
        return current_position
    
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

    def set_sam_config(self):
        
        tracking_points = self.tracking_points
        positive_points = []
        negative_points = []
        labels = []
        
        if self.button_param_select.currentText() == 'Frame Mode':
            is_video = False
        elif self.button_param_select.currentText() == 'Video Mode':
            is_video = True
        else:
            raise ValueError('Please select the sam mode')

        select_frame = self.progress_slider.value()
        frame_pts = tracking_points[select_frame]
        if frame_pts['raw_pos'] != []:
            positive_points.extend([[pt.x(), pt.y()] for pt in frame_pts['raw_pos']])
        if frame_pts['raw_neg'] != []:
            negative_points.extend([[pt.x(), pt.y()] for pt in frame_pts['raw_neg']])
        if (frame_pts['raw_pos'] != []) or (frame_pts['raw_neg'] != []):
            labels.extend(frame_pts['labels'])

        self.sam_config['is_video'] = is_video
        self.sam_config['positive_points'] = positive_points
        self.sam_config['negative_points'] = negative_points
        self.sam_config['labels'] = labels
        self.sam_config['select_frame'] = select_frame
    
    def get_sam_result(self):
        self.set_sam_config()   
        masks, mask_images = request_sam(self.sam_config)
        self.sam_res = mask_images
        
        if self.sam_config['is_video']:
            for i, mask in enumerate(masks):
                self.tracking_masks[self.sam_config['select_frame'] + i] = mask
        else:
            self.tracking_masks[self.sam_config['select_frame']] = masks[0]
        
    def get_tap_result(self):        
        self.co_tracker_config['mode'] = self.button_param_select.currentText()
        
        if self.co_tracker_config['mode'] == 'Point Mode':
            points, select_frame = [], []
            for frame_id, frame_pts in self.tracking_points.items():
                if frame_pts['raw_pos'] != []:
                    points.extend([[pt.x(), pt.y()] for pt in frame_pts['raw_pos']])
                    select_frame.extend([[frame_id]] * len(frame_pts['labels']))
            self.co_tracker_config['points'] = points
            self.co_tracker_config['select_frame'] = select_frame
            assert len(select_frame) == len(points)
            self.co_tracker_config['mode'] = 'point' # TODO: add mode selection
        
        elif self.co_tracker_config['mode'] == 'Mask Mode':
            self.set_sam_config()
            self.co_tracker_config['select_frame'] = self.sam_config['select_frame']  
        
        elif self.co_tracker_config['mode'] == 'Grid Mode':
            self.co_tracker_config['grid_size'] = 10
        
        else:
            raise ValueError('Please select the tracker mode')
        
        pred_tracks, pred_visibility, images = request_cotracker(self.sam_config, self.co_tracker_config)
        self.tracker_res = images
    
    def mousePressEvent(self, event: QMouseEvent):
        if self.last_frame is None:
            return
        if event.button() == Qt.LeftButton and self.last_frame is not None:
            pos = self.video_label.mapFromGlobal(event.globalPos())
            gt_pos = self.get_align_point(pos.x(), pos.y())
            if gt_pos is None:
                return
            click_position = QPoint(gt_pos[0], gt_pos[1])
            original_position = QPoint(int(gt_pos[0]//self.scale), int(gt_pos[1]//self.scale))            
            self.tracking_points[self.progress_slider.value()]['raw_pos'].append(original_position)
            self.tracking_points[self.progress_slider.value()]['pos'].append(click_position)
            self.tracking_points[self.progress_slider.value()]['labels'].append(1)
            
            # Draw a point on the frame
        elif event.button() == Qt.RightButton and self.last_frame is not None:
            pos = self.video_label.mapFromGlobal(event.globalPos())
            gt_pos = self.get_align_point(pos.x(), pos.y())
            if gt_pos is None:
                return
            click_position = QPoint(gt_pos[0], gt_pos[1])
            original_position = QPoint(int(gt_pos[0]//self.scale), int(gt_pos[1]//self.scale))     
            self.tracking_points[self.progress_slider.value()]['neg'].append(click_position)
            self.tracking_points[self.progress_slider.value()]['raw_neg'].append(original_position)
            self.tracking_points[self.progress_slider.value()]['labels'].append(-1)
            
        self.draw_image()
    
    def get_align_point(self, x, y): 
        label_height, label_width = self.video_label.height(), self.video_label.width()
        resized_width = int(self.width * min(self.scale_width, self.scale_height))
        resized_height = int(self.height * min(self.scale_width, self.scale_height))
        offset_x = (label_width - resized_width) // 2
        offset_y = (label_height - resized_height) // 2
        x -= offset_x
        y -= offset_y
        
        gt_shape = self.last_frame.shape
        if x < 0 or y < 0 or x >= gt_shape[1] or y >= gt_shape[0]:
            return None
        
        return (x, y)
     
    def draw_image(self):
        frame = self.last_frame.copy()
        pos_click_position = self.tracking_points[self.progress_slider.value()]['pos']
        neg_click_position = self.tracking_points[self.progress_slider.value()]['neg']

        for point in pos_click_position:
            x, y = point.x(), point.y()
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        for point in neg_click_position:
            x, y = point.x(), point.y()
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        if len(pos_click_position)==0 and len(neg_click_position)==0:
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
        keyframe_image.fill(Qt.gray)

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