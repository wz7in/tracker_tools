import sys
import os
import json
import argparse
import pickle
from PyQt5.QtCore import QPoint, QTimer, Qt
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                             QLabel, QSlider, QFileDialog, QLineEdit, QHBoxLayout, QFrame, QButtonGroup, QRadioButton, QToolTip, QComboBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QMouseEvent
from PyQt5.QtCore import Qt, QRect, QEvent, QPoint

import yaml
from client_utils import request_sam, request_cotracker, request_video
import numpy as np

def load_anno_file(anno_file, out_file):
    with open(anno_file, 'r') as f:
        video_list = f.readlines()
    
    if os.path.exists(out_file):
        anno = pickle.load(open(out_file, 'rb'))
    else:
        anno = {}
    
    video_list = [line.strip() for line in video_list]
    video_list = sorted([line for line in video_list if line not in anno])
        
    return video_list, anno

class VideoPlayer(QWidget):
    def __init__(self, args):
        
        # load video list
        self.video_list, self.anno = load_anno_file(args.anno_file, args.out_file)
        
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
        self.progress_slider.hide()  # Hide initially
        video_layout.addWidget(self.progress_slider)

        # Keyframe indicator bar
        self.keyframe_bar = QLabel(self)
        self.keyframe_bar.setFixedHeight(20)  # Set the height of the keyframe bar
        self.keyframe_bar.setMouseTracking(True)
        self.keyframe_bar.setAlignment(Qt.AlignCenter)  # Center the QLabel
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
        
        video_control_button_layout = QHBoxLayout()
        
        # Pre video button
        self.pre_button = QPushButton("<<", self)
        self.pre_button.clicked.connect(self.pre_video)
        self.pre_button.setDisabled(True)
        video_control_button_layout.addWidget(self.pre_button)

        # Pre Frame button
        self.pre_f_button = QPushButton("<", self)
        self.pre_f_button.clicked.connect(self.pre_frame)
        self.pre_button.setDisabled(True)
        video_control_button_layout.addWidget(self.pre_f_button)

        self.video_position_label = QLabel(self)
        self.video_position_label.setStyleSheet("background-color: gray;")
        self.video_position_label.setAlignment(Qt.AlignCenter)
        self.video_position_label.setFixedSize(200, 20)
        video_control_button_layout.addWidget(self.video_position_label)

        # Next video button
        self.next_f_button = QPushButton(">", self)
        self.next_f_button.clicked.connect(self.next_frame)
        video_control_button_layout.addWidget(self.next_f_button)
        
        # Next video button
        self.next_button = QPushButton(">>", self)
        self.next_button.clicked.connect(self.next_video)
        video_control_button_layout.addWidget(self.next_button)
        video_layout.addLayout(video_control_button_layout)
        
        video_load_button_layout = QHBoxLayout()
        
        # Load video button
        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.load_video)
        video_load_button_layout.addWidget(self.load_button)
        
        self.load_clip_data_button = QPushButton("Load Clip", self)
        self.load_clip_data_button.clicked.connect(self.load_clip_data)
        video_load_button_layout.addWidget(self.load_clip_data_button)
        
        self.play_button = QPushButton("Play", self)
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_playback)
        video_load_button_layout.addWidget(self.play_button)
        
        self.remove_video_button = QPushButton("Remove Video", self)
        self.remove_video_button.clicked.connect(self.clear_video)
        video_load_button_layout.addWidget(self.remove_video_button)
        
        video_layout.addLayout(video_load_button_layout)
        
        # Add video layout to the main layout
        main_layout.addLayout(video_layout)
        
        # Separator line between video area and toolbar
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Toolbar layout
        self.toolbar_layout = QVBoxLayout()

        # Add the horizontal layout to the toolbar layout
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
        
        anno_button_layout = QHBoxLayout()
        self.anno_function_select = QComboBox()
        self.anno_function_select.addItem('Sam')
        self.anno_function_select.addItem('Tracker')
        
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

        
        # Create a horizontal layout for the title and line
        annotation_title_layout = QHBoxLayout()
        annotation_title = QLabel("Visualization Annotation", self)
        annotation_title.setAlignment(Qt.AlignLeft)  # Left align the title
        annotation_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        annotation_title_layout.addWidget(annotation_title)
        
        line = QFrame(self)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: grey;")  # Set the same color as the title
        annotation_title_layout.addWidget(line)
        
        self.toolbar_layout.addLayout(annotation_title_layout)
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
        annotation_title = QLabel("Language Annotation", self)
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
        
        self.description_input = QLineEdit(self)
        self.description_input.setPlaceholderText("Enter description...")
        self.toolbar_layout.addWidget(self.description_input)
       
        # Language description input
        self.desc_layout = QHBoxLayout()
        self.description_mode = QComboBox(self)
        self.description_mode.addItems(['Frame Mode','Video Mode'])
        self.desc_layout.addWidget(self.description_mode)
        
        # Submit button for language description
        self.submit_description_button = QPushButton("Submit Description", self)
        self.submit_description_button.clicked.connect(self.submit_description)
        self.desc_layout.addWidget(self.submit_description_button)
        
        self.toolbar_layout.addLayout(self.desc_layout)

        # Create a horizontal layout for the title and line
        annotation_title_layout = QHBoxLayout()

        # Add a label for the per-frame annotation title
        annotation_title = QLabel("Key Frame Annotation", self)
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
        # keyframe_option_layout = QHBoxLayout()
        # self.keyframe_button_group = QButtonGroup(self)
        # self.start_button = QRadioButton("Start", self)
        # self.end_button = QRadioButton("End", self)
        # self.keyframe_button_group.addButton(self.start_button)
        # self.keyframe_button_group.addButton(self.end_button)
        # keyframe_option_layout.addWidget(self.start_button)
        # keyframe_option_layout.addWidget(self.end_button)
        # self.toolbar_layout.addLayout(keyframe_option_layout)

        keyframe_button_layout = QHBoxLayout()
        self.key_frame_selector = QComboBox()
        self.key_frame_selector.addItems(['Start', 'End'])
        keyframe_button_layout.addWidget(self.key_frame_selector)
        
        # Mark keyframe button
        self.mark_keyframe_button = QPushButton("Mark Keyframe", self)
        self.mark_keyframe_button.clicked.connect(self.mark_keyframe)
        keyframe_button_layout.addWidget(self.mark_keyframe_button)
        
        # Remove keyframe button
        self.remove_keyframe_button = QPushButton("Remove Keyframe", self)
        self.remove_keyframe_button.clicked.connect(self.remove_keyframe)
        keyframe_button_layout.addWidget(self.remove_keyframe_button)
        self.toolbar_layout.addLayout(keyframe_button_layout)

        # edit mode layout
        # Add a label for the per-frame annotation title
        annotation_title_layout = QHBoxLayout()
        annotation_title = QLabel("Edit Annotation", self)
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
        
        self.control_button_layout = QHBoxLayout()
        # clear_all_button
        self.clear_all_button = QPushButton("Clear", self)
        self.clear_all_button.clicked.connect(self.clear_annotations)
        self.control_button_layout.addWidget(self.clear_all_button)
        
        # remove_last_button
        self.remove_last_button = QPushButton("Remove Last", self)
        self.remove_last_button.clicked.connect(self.remove_last_annotation)
        self.control_button_layout.addWidget(self.remove_last_button)
        
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_result)
        self.control_button_layout.addWidget(self.save_button)
        
        self.toolbar_layout.addLayout(self.control_button_layout)
        self.edit_button_layout = QHBoxLayout()

        self.edit_button = QPushButton("Edit keypoints", self)
        self.edit_button.clicked.connect(self.edit_track_pts)
        self.edit_button_layout.addWidget(self.edit_button)

        self.close_edit_button = QPushButton("Save Edit", self)
        self.close_edit_button.clicked.connect(self.close_edit)
        self.edit_button_layout.addWidget(self.close_edit_button)
        
        self.track_point_selector = QComboBox()
        self.track_point_selector.hide()
        self.edit_button_layout.addWidget(self.track_point_selector)

        self.toolbar_layout.addLayout(self.edit_button_layout)

        # Add spacer to push the items to the top
        self.toolbar_layout.addStretch()
        
        # Add toolbar layout to the main layout
        main_layout.addLayout(self.toolbar_layout)

        self.setLayout(main_layout)

        # self.cap = None
        self.cur_video_idx = 1
        self.frame_count = 0
        self.last_frame = None
        self.tracking_points = dict()
        self.tracking_masks = dict()
        # self.video_position_label.setText(f"{self.cur_video_idx}/{len(self.video_list)}, -/-")
        self.video_position_label.setText(f"Video: {self.cur_video_idx}/{len(self.video_list)}, Frame: -/-")
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_video)

        # add config
        config_path = "./config/config.yaml"
        with open(config_path, "r") as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)
        self.sam_config = self.model_config["sam"]
        self.co_tracker_config = self.model_config["cotracker"]

        # initialize
        self.vis_track_res = False
        self.sam_res = []
        self.video_cache = dict()
        self.cur_frame_idx = self.progress_slider.value()
        self.pre_f_button.setDisabled(True)
        self.next_f_button.setDisabled(True)
        self.is_edit_mode = False

    def edit_track_pts(self):
        tracked_points = self.anno[self.video_list[self.cur_video_idx-1]]['track'][0][0]
        num_key_pts = tracked_points.shape[1]
        # self.edit_choice_button = []
        # for i in range(num_key_pts):
        #     choice_button = QRadioButton(str(i))
        #     self.edit_choice_button.append(choice_button)
        #     self.toolbar_layout.addWidget(choice_button)
        self.track_point_selector.show()
        self.track_point_selector.clear()
        for i in range(num_key_pts):
            self.track_point_selector.addItem(f"Point {i+1}")
        
        self.edit_track_res = self.anno[self.video_list[self.cur_video_idx-1]]['track'][0][0].copy()
        self.is_edit_mode = True

    def close_edit(self):
        self.anno[self.video_list[self.cur_video_idx-1]]['track'][0][0] =  self.edit_track_res
        self.is_edit_mode = False
        self.track_point_selector.hide()


    def next_video(self):
        if self.cur_video_idx < len(self.video_list):
            self.cur_video_idx += 1
        if self.cur_video_idx == len(self.video_list):
            self.next_button.setDisabled(True)
        self.pre_button.setDisabled(False)
        self.video_position_label.setText(f"Video: {self.cur_video_idx}/{len(self.video_list)}, Frame: -/-")

    def pre_video(self):
        if self.cur_video_idx > 1:
            self.cur_video_idx -= 1
        if self.cur_video_idx == 1:
            self.pre_button.setDisabled(True)
        self.next_button.setDisabled(False)
        self.video_position_label.setText(f"Video: {self.cur_video_idx}/{len(self.video_list)}, Frame: -/-")

    def next_frame(self):
        if self.cur_frame_idx < self.frame_count:
            self.cur_frame_idx += 1
        if self.cur_frame_idx == self.frame_count-1:
            self.next_f_button.setDisabled(True)
        self.pre_f_button.setDisabled(False)
        self.video_position_label.setText(f"Video: {self.cur_video_idx}/{len(self.video_list)}, Frame: {self.cur_frame_idx}/{self.frame_count}")

        self.update_frame(self.cur_frame_idx)
        self.progress_slider.setValue(self.cur_frame_idx)
        

    def pre_frame(self):
        if self.cur_frame_idx >= 1:
            self.cur_frame_idx -= 1
        if self.cur_frame_idx == 0:
            self.pre_f_button.setDisabled(True)
        self.next_f_button.setDisabled(False)
        self.video_position_label.setText(f"Video: {self.cur_video_idx}/{len(self.video_list)}, Fame: {self.cur_frame_idx}/{self.frame_count}")

        self.update_frame(self.cur_frame_idx)
        self.progress_slider.setValue(self.cur_frame_idx)
    
    def request_video(self):
        if self.video_list[self.cur_video_idx-1] in self.video_cache:
            video = self.video_cache[self.cur_video_idx]
        else:
            video = request_video(self.video_list[self.cur_video_idx-1])
            self.video_cache[self.cur_video_idx-1] = video
        return video
    
    def save_result(self):
        # json.dump(self.anno, open(args.out_file, 'w'))
        pickle.dump(self.anno, open(args.out_file, 'wb'))
    
    def get_anno_result(self):
        if self.anno_function_select.currentText() == 'Sam':
            self.get_sam_result()
        elif self.anno_function_select.currentText() == 'Tracker':
            self.get_tap_result()
    
    def update_function_select(self):
        if self.anno_function_select.currentText() == 'Sam':
            self.button_param_select.clear()
            self.button_param_select.addItem('Frame Mode')
            self.button_param_select.addItem('Video Mode')
        elif self.anno_function_select.currentText() == 'Tracker':
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
        for k, _ in self.tracking_points.items():
            self.tracking_points[k]['pos'] = []
            self.tracking_points[k]['raw_pos'] = []
            self.tracking_points[k]['neg'] = []
            self.tracking_points[k]['raw_neg'] = []
            self.tracking_points[k]['labels'] = []
        # self.tracking_points[self.progress_slider.value()]['pos'] = []
        # self.tracking_points[self.progress_slider.value()]['raw_pos'] = []
        # self.tracking_points[self.progress_slider.value()]['neg'] = []
        # self.tracking_points[self.progress_slider.value()]['raw_neg'] = []
        # self.tracking_points[self.progress_slider.value()]['labels'] = []
        if self.last_frame is not None:
            self.draw_image()

    def load_clip_data(self):
        video_name = self.video_list[self.cur_video_idx-1].split('/')[-1].split('.')[0]
        clip_data_pth = f'./{video_name}.json'
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
        # if self.cap is not None:
        #     self.cap.release()
        # self.cap = None
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
        video = self.request_video()
        if video is None:
            return
        self.sam_config['video_path'] = self.video_list[self.cur_video_idx-1]
        self.co_tracker_config['video_path'] = self.video_list[self.cur_video_idx-1]
        self.frame_count = video.shape[0]
        self.ori_video = []
        for i in range(self.frame_count):
            self.tracking_points[i] = dict(
                pos=[], raw_pos=[], neg=[], raw_neg=[], labels=[]
            )
            self.tracking_masks[i] = []
        self.ori_video = np.array(video)
        self.progress_slider.setMaximum(self.frame_count - 1)
        self.update_frame(0)
        self.progress_slider.show()
        self.frame_position_label.show()
        self.update_keyframe_bar()  # Initialize keyframe bar

        self.video_position_label.setText(f"Video: {self.cur_video_idx}/{len(self.video_list)}, Frame: {self.cur_frame_idx}/{self.frame_count}")
        self.pre_f_button.setDisabled(True)

        # video_name = self.video_path.split('/')[-1].split('.')[0]
        video_name = self.video_list[self.cur_video_idx-1]
        # print(video_name)
        if video_name not in self.anno:
            self.anno[video_name] = dict(
                mask=[], track=[]
            )
        clip_data_pth = f'./{video_name}.json'
        if os.path.exists(clip_data_pth):
            self.load_clip_data()
            
    def update_frame(self, frame_number):
        # if self.cap is not None:
        #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        #     ret, frame = self.cap.read()
        #     if ret:
                # Resize frame to fit the QLabel while keeping aspect ratio
        frame = self.ori_video[frame_number]
        if self.vis_track_res:
            frame = self.track_res[frame_number]
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        self.cur_frame_idx = self.progress_slider.value()
        self.video_position_label.setText(f"Video: {self.cur_video_idx}/{len(self.video_list)}, Frame: {self.cur_frame_idx}/{self.frame_count}")

        self.pre_f_button.setDisabled(False)
        self.next_f_button.setDisabled(False)
        if self.cur_frame_idx == 0:
            self.pre_f_button.setDisabled(True)
        if self.cur_frame_idx == self.frame_count-1:
            self.next_f_button.setDisabled(True)
    
    def toggle_playback(self):
        # if self.cap is None:
        #     return
        if self.play_button.isChecked():
            self.play_button.setText("Stop")
            self.current_frame = self.progress_slider.value()
            self.timer.start(30)  # Set timer to update frame every 30 ms
        else:
            self.play_button.setText("Play")
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
        # if self.cap is not None:
        if self.current_frame < self.frame_count - 1:
            self.current_frame += 1
            self.update_frame(self.current_frame)
            self.progress_slider.setValue(self.current_frame)
        else:
            self.timer.stop()
            self.play_button.setChecked(False)
            self.play_button.setText("Play")

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
        frame_id = self.sam_config['select_frame']
        mask_images = np.array([cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB) for mask_image in mask_images])
        if mask_images.shape[0] != self.frame_count:
            if len(self.sam_res) == 0:
                self.sam_res = self.ori_video.copy()    
            self.sam_res[frame_id:frame_id+mask_images.shape[0]] = mask_images
        else:
            self.sam_res = mask_images
        
        if self.sam_config['is_video']:
            for i, mask in enumerate(masks):
                self.tracking_masks[self.sam_config['select_frame'] + i] = mask
            self.anno[self.video_list[self.cur_video_idx-1]]['mask'] = masks
        else:
            self.tracking_masks[self.sam_config['select_frame']] = masks[0]
            if len(self.anno[self.video_list[self.cur_video_idx-1]]['mask']) > 0:
                self.anno[self.video_list[self.cur_video_idx-1]]['mask'][frame_id] = masks[0]
            else:
                self.anno[self.video_list[self.cur_video_idx-1]]['mask'] = np.zeros((self.frame_count, *masks[0].shape))
                self.anno[self.video_list[self.cur_video_idx-1]]['mask'][frame_id] = masks[0]
                
    def get_tap_result(self):        
        self.co_tracker_config['mode'] = self.button_param_select.currentText()
        
        if self.co_tracker_config['mode'] == 'Point Mode':
            points, select_frame = [], []
            for frame_id, frame_pts in self.tracking_points.items():
                if frame_pts['raw_pos'] != []:
                    points.extend([[pt.x(), pt.y()] for pt in frame_pts['raw_pos']])
                    select_frame.extend([[frame_id]] * len(frame_pts['raw_pos']))
            self.co_tracker_config['points'] = points
            self.co_tracker_config['select_frame'] = select_frame
            assert len(select_frame) == len(points)
            # self.co_tracker_config['mode'] = 'point' # TODO: add mode selection
        
        elif self.co_tracker_config['mode'] == 'Mask Mode':
            self.set_sam_config()
            self.co_tracker_config['select_frame'] = self.sam_config['select_frame']  
        
        elif self.co_tracker_config['mode'] == 'Grid Mode':
            self.co_tracker_config['grid_size'] = 10
        
        else:
            raise ValueError('Please select the tracker mode')
        
        pred_tracks, pred_visibility, images = request_cotracker(self.sam_config, self.co_tracker_config)
        self.tracker_res = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images])
        self.anno[self.video_list[self.cur_video_idx-1]]['track'] = (pred_tracks, pred_visibility)
    
    def mousePressEvent(self, event: QMouseEvent):
        if self.is_edit_mode:
            pos = self.video_label.mapFromGlobal(event.globalPos())
            gt_pos = self.get_align_point(pos.x(), pos.y())
            self.cur_edit_click = np.array([int(gt_pos[0]), int(gt_pos[1])])
            original_position = QPoint(int(gt_pos[0]//self.scale), int(gt_pos[1]//self.scale))

            cur_frame = self.cur_frame_idx
            cur_pt_id = int(self.track_point_selector.currentText().split(' ')[-1]) - 1
            self.edit_track_res[cur_frame, cur_pt_id, :] = np.array([original_position.x(), original_position.y()])

            self.draw_image()

            return
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
        if self.is_edit_mode:
            x, y = int(self.cur_edit_click[0]), int(self.cur_edit_click[1])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            pos_click_position = self.tracking_points[self.progress_slider.value()]['pos']
            neg_click_position = self.tracking_points[self.progress_slider.value()]['neg']

            for point in pos_click_position:
                x, y = point.x(), point.y()
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            for point in neg_click_position:
                x, y = point.x(), point.y()
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1, lineType=cv2.LINE_AA)
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
        if self.key_frame_selector.currentText() == 'Start':
            self.keyframes[current_frame] = 'start'
        elif self.key_frame_selector.currentText() == 'End':
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
        # if self.cap is not None:
        #     self.cap.release()
        event.accept()
            
if __name__ == "__main__":
    
    # load annotation file
    args = argparse.ArgumentParser()
    args.add_argument('--anno_file', type=str, default='./data/video_list.txt')
    args.add_argument('--out_file', type=str, default='./data/annotation.json')
    args = args.parse_args()
    
    app = QApplication(sys.argv)
    player = VideoPlayer(args)
    player.resize(1000, 600)  # Adjusted size to accommodate the toolbar
    player.show()
    sys.exit(app.exec_())