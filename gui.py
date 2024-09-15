import sys
import os, time
import json
import argparse
import pickle
from PyQt5.QtCore import QPoint, QTimer, Qt
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox, QLineEdit, QDialogButtonBox, QTextEdit,
                             QLabel, QSlider, QDialog, QHBoxLayout, QFrame, QProgressDialog, QRadioButton, QToolTip, QComboBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QMouseEvent, QBrush
from PyQt5.QtCore import Qt, QRect, QEvent, QPoint, QSize, pyqtSignal, QThread
from PyQt5.QtGui import QMovie

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

class TextInputDialog(QDialog):
    def __init__(self, initial_text='', parent=None):
        super().__init__(parent)
        self.setWindowTitle('请输入语言标注')
        
        self.layout = QVBoxLayout(self)
        
        self.label = QLabel('请输入语言标注:')
        self.layout.addWidget(self.label)
        
        self.text_input = QLineEdit(self)
        self.text_input.setText(initial_text)
        self.layout.addWidget(self.text_input)
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.layout.addWidget(self.button_box)
        
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
    
    def get_text(self):
        return self.text_input.text()


class VideoPlayer(QWidget):
    def __init__(self, args):
        
        # load video list
        self.video_list, self.anno = load_anno_file(args.anno_file, args.out_file)
        
        super().__init__()
        self.setWindowTitle("浦器实验室视频标注工具")
        # 设置背景图片, 并设置模糊度，不许重复

        
        # Main layout to contain both video display and the toolbar
        main_layout = QHBoxLayout()
        # 添加背景图片       
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
        self.keyframe_bar.setFixedHeight(10)  # Set the height of the keyframe bar
        # self.keyframe_bar.setMouseTracking(True)
        self.keyframe_bar.setAlignment(Qt.AlignCenter)  # Center the QLabel
        # self.keyframe_bar.installEventFilter(self)  # Install event filter to handle mouse events
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
        self.load_button.clicked.connect(self.load_video_async)
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
        self.button_param_select.addItem('Video Mode')
        self.button_param_select.addItem('Frame Mode')
        
        self.anno_function_select.currentIndexChanged.connect(self.update_function_select)
    
        click_action_button = QPushButton("Run", self)
        # select color
        click_action_button.clicked.connect(self.get_anno_result)
        anno_button_layout.addWidget(self.anno_function_select)
        anno_button_layout.addWidget(self.button_param_select)
        anno_button_layout.addWidget(click_action_button)
        self.toolbar_layout.addLayout(anno_button_layout)
        
        self.sam_object_layout = QHBoxLayout()
        
        self.sam_pre_button = QPushButton("Pre Object", self)
        self.sam_pre_button.clicked.connect(self.pre_sam_object)
        self.sam_pre_button.setDisabled(True)
        
        self.sam_obj_pos_label = QLabel(self)
        self.sam_obj_pos_label.setStyleSheet("background-color: gray;")
        self.sam_obj_pos_label.setAlignment(Qt.AlignCenter)
        self.sam_obj_pos_label.setFixedSize(150, 20)
        
        self.sam_next_button = QPushButton("Next/Add Object", self)
        self.sam_next_button.clicked.connect(self.next_sam_object)
        self.sam_next_button.setDisabled(True)
        
        self.sam_object_layout.addWidget(self.sam_pre_button)
        self.sam_object_layout.addWidget(self.sam_obj_pos_label)
        self.sam_object_layout.addWidget(self.sam_next_button)
        self.toolbar_layout.addLayout(self.sam_object_layout)
        
        self.edit_button_layout = QHBoxLayout()

        self.edit_button = QPushButton("Edit keypoints", self)
        self.edit_button.clicked.connect(self.edit_track_pts)
        self.edit_button_layout.addWidget(self.edit_button)
        self.edit_button.hide()

        self.close_edit_button = QPushButton("Save Edit", self)
        self.close_edit_button.clicked.connect(self.close_edit)
        self.edit_button_layout.addWidget(self.close_edit_button)
        self.close_edit_button.hide()
        
        self.track_point_selector = QComboBox()
        self.track_point_selector.hide()
        self.edit_button_layout.addWidget(self.track_point_selector)
        
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
        
        # self.description_input = QLineEdit(self)
        # self.description_input.setPlaceholderText("Enter description...")
        # self.toolbar_layout.addWidget(self.description_input)
       
        # Language description input
        self.desc_layout = QHBoxLayout()
        self.description_mode = QComboBox(self)
        self.description_mode.addItems(['Video', 'Clip'])
        self.desc_layout.addWidget(self.description_mode)
        
        # Submit button for language description
        self.submit_description_button = QPushButton("Add / Motified Description", self)
        self.submit_description_button.clicked.connect(self.submit_description)
        self.desc_layout.addWidget(self.submit_description_button)
        
        self.toolbar_layout.addLayout(self.desc_layout)

        # Create a horizontal layout for the title and line
        # annotation_title_layout = QHBoxLayout()

        # # Add a label for the per-frame annotation title
        # annotation_title = QLabel("Key Frame Annotation Tips:", self)
        # annotation_title.setAlignment(Qt.AlignLeft)  # Left align the title
        # annotation_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        # annotation_title_layout.addWidget(annotation_title)

        # # Add a horizontal line to fill the remaining space
        # line = QFrame(self)
        # line.setFrameShape(QFrame.HLine)
        # line.setFrameShadow(QFrame.Sunken)
        # line.setStyleSheet("color: grey;")  # Set the same color as the title
        # annotation_title_layout.addWidget(line)

        # Add the horizontal layout to the toolbar layout
        # self.toolbar_layout.addLayout(annotation_title_layout)

        # keyframe_button_layout = QHBoxLayout()
        # self.key_frame_selector = QComboBox()
        # self.key_frame_selector.addItems(['Start', 'End'])
        # keyframe_button_layout.addWidget(self.key_frame_selector)
        
        # Mark keyframe button
        # self.mark_keyframe_button = QPushButton("Mark Keyframe", self)
        # self.mark_keyframe_button.clicked.connect(self.mark_keyframe)
        # keyframe_button_layout.addWidget(self.mark_keyframe_button)
        
        # # Remove keyframe button
        # self.remove_keyframe_button = QPushButton("Remove Keyframe", self)
        # self.remove_keyframe_button.clicked.connect(self.remove_keyframe)
        # keyframe_button_layout.addWidget(self.remove_keyframe_button)
        # self.toolbar_layout.addLayout(keyframe_button_layout)

        # edit mode layout
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
        self.remove_last_sam_button = QPushButton("Remove Last SAM", self)
        self.remove_last_sam_button.clicked.connect(self.remove_last_sam_annotation)
        self.control_button_layout.addWidget(self.remove_last_sam_button)
        
        self.remove_last_tap_button = QPushButton("Remove Last TAP", self)
        self.remove_last_tap_button.clicked.connect(self.remove_last_tap_annotation)
        self.control_button_layout.addWidget(self.remove_last_tap_button)
        
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_result)
        self.control_button_layout.addWidget(self.save_button)
        
        self.toolbar_layout.addLayout(self.control_button_layout)

        # Add spacer to push the items to the top
        self.toolbar_layout.addStretch()
        
        # Add Language Show
        lang_layout = QVBoxLayout()
        lang_title_layout = QHBoxLayout()
        lang_title = QLabel("Video Language Annotation", self)
        lang_title.setAlignment(Qt.AlignLeft)  # Left align the title
        lang_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        lang_title_layout.addWidget(lang_title)
        # Add a horizontal line to fill the remaining space
        line = QFrame(self)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: grey;")  # Set the same color as the title
        lang_title_layout.addWidget(line)
        lang_layout.addLayout(lang_title_layout)
        # Add a text edit to show the language annotation
        self.video_lang_input = QTextEdit(self)
        self.video_lang_input.setReadOnly(True)
        self.video_lang_input.setFixedHeight(30)
        lang_layout.addWidget(self.video_lang_input)
        self.toolbar_layout.addLayout(lang_layout)
        
        # Add Language Show
        lang_layout = QVBoxLayout()
        lang_title_layout = QHBoxLayout()
        lang_title = QLabel("Clip Language Annotation", self)
        lang_title.setAlignment(Qt.AlignLeft)  # Left align the title
        lang_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        lang_title_layout.addWidget(lang_title)
        # Add a horizontal line to fill the remaining space
        line = QFrame(self)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: grey;")  # Set the same color as the title
        lang_title_layout.addWidget(line)
        lang_layout.addLayout(lang_title_layout)
        # Add a text edit to show the language annotation
        self.clip_lang_input = QTextEdit(self)
        self.clip_lang_input.setReadOnly(True)
        self.clip_lang_input.setFixedHeight(30)
        lang_layout.addWidget(self.clip_lang_input)
        self.toolbar_layout.addLayout(lang_layout)
        
        # Add Tool Tip
        self.tips_layout = QVBoxLayout()
        self.tips_title_layout = QHBoxLayout()
        tips_title = QLabel("Key Frame Annotation Tips:", self)
        tips_title.setAlignment(Qt.AlignLeft)  # Left align the title
        tips_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        self.tips_title_layout.addWidget(tips_title)
        # Add a horizontal line to fill the remaining space
        line = QFrame(self)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: grey;")  # Set the same color as the title
        self.tips_title_layout.addWidget(line)
        self.tips_layout.addLayout(self.tips_title_layout)
        
        self.tips_input = QTextEdit(self)
        self.tips_input.setText(
            "Q:\tStart Key Frame \tENTER:\tAdd/Motify Language\nE:\tEnd Key Frame \tBACK:\tDelete Key Frame \nA:\tPre Frame \nD:\tNext Frame")
        self.tips_input.setReadOnly(True)
        self.tips_input.setFixedHeight(90)
        self.tips_layout.addWidget(self.tips_input)
        self.toolbar_layout.addLayout(self.tips_layout)
        
        # Add toolbar layout to the main layout
        main_layout.addLayout(self.toolbar_layout)

        self.setLayout(main_layout)

        # self.cap = None
        self.cur_video_idx = 1
        self.frame_count = 0
        self.last_frame = None
        self.tracking_points_sam = dict()
        self.tracking_points_tap = dict()
        self.tracking_masks = dict()
        # self.video_position_label.setText(f"{self.cur_video_idx}/{len(self.video_list)}, -/-")
        self.video_position_label.setText(f"Frame: -/- | Video: {self.cur_video_idx}/{len(self.video_list)}")
        self.sam_obj_pos_label.setText("Annotation Object: -/-")
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_video)

        # add config
        config_path = "./config/config.yaml"
        with open(config_path, "r") as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)
        self.sam_config = self.model_config["sam"]
        self.co_tracker_config = self.model_config["cotracker"]

        # initialize
        self.ori_video = []
        self.vis_track_res = False
        self.sam_res = []
        self.lang_anno = dict()
        self.video_cache = dict()
        self.anno_mode = 'sam'
        self.cur_frame_idx = self.progress_slider.value()
        self.pre_f_button.setDisabled(True)
        self.next_f_button.setDisabled(True)
        self.is_edit_mode = False        
    
    def pre_sam_object(self):
        if self.sam_object_id[self.progress_slider.value()] > 0:
            self.sam_object_id[self.progress_slider.value()] -= 1
        if self.sam_object_id[self.progress_slider.value()] == 0:
            self.sam_pre_button.setDisabled(True)
        
        self.sam_next_button.setDisabled(False)
        cur_id = self.sam_object_id[self.progress_slider.value()] + 1
        all_object_size = len(self.tracking_points_sam[self.progress_slider.value()])
        self.sam_obj_pos_label.setText(f"Annotation: {cur_id}/{all_object_size}")
        
        self.draw_image()
    
    def resizeEvent(self, event):
        self.seek_video()
        self.update_keyframe_bar()
        
        self.setAutoFillBackground(False)
        palette = self.palette()
        palette.setBrush(self.backgroundRole(), QBrush(QPixmap('./demo/bg.png').scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)))
        self.setPalette(palette)
    
    def next_sam_object(self):
        
        self.sam_object_id[self.progress_slider.value()] += 1
        if self.sam_object_id[self.progress_slider.value()] == len(self.tracking_points_sam[self.progress_slider.value()]):
            self.tracking_points_sam[self.progress_slider.value()].append(
                dict(pos=[], raw_pos=[], neg=[], raw_neg=[], labels=[])
            )
        self.sam_pre_button.setDisabled(False)
        
        if len(self.tracking_points_sam[self.progress_slider.value()][self.sam_object_id[self.progress_slider.value()]]['pos']) == 0:
            self.sam_next_button.setDisabled(True)
        else:
            self.sam_next_button.setDisabled(False)
        
        cur_id = self.sam_object_id[self.progress_slider.value()] + 1
        all_object_size = len(self.tracking_points_sam[self.progress_slider.value()])
        self.sam_obj_pos_label.setText(f"Annotation: {cur_id}/{all_object_size}")
        
        self.draw_image()
    
    def edit_track_pts(self):
        tracked_points = self.anno[self.video_list[self.cur_video_idx-1]]['track'][0][0]
        num_key_pts = tracked_points.shape[1]
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
        self.video_position_label.setText(f"Frame: -/- | Video: {self.cur_video_idx}/{len(self.video_list)}")

    def pre_video(self):
        if self.cur_video_idx > 1:
            self.cur_video_idx -= 1
        if self.cur_video_idx == 1:
            self.pre_button.setDisabled(True)
        self.next_button.setDisabled(False)
        self.video_position_label.setText(f"Frame: -/- | Video: {self.cur_video_idx}/{len(self.video_list)}")

    def next_frame(self):
        if self.cur_frame_idx < self.frame_count:
            self.cur_frame_idx += 1
        if self.cur_frame_idx == self.frame_count-1:
            self.next_f_button.setDisabled(True)
        self.pre_f_button.setDisabled(False)
        self.video_position_label.setText(f"Frame: {self.cur_frame_idx}/{self.frame_count} | Video: {self.cur_video_idx}/{len(self.video_list)}")
        self.sam_object_id[self.cur_frame_idx] = 0
        
        if self.sam_object_id[self.cur_frame_idx] == 0:
            self.sam_pre_button.setDisabled(True)     
        
        self.update_frame(self.cur_frame_idx)
        self.progress_slider.setValue(self.cur_frame_idx)
        
        cur_id = self.sam_object_id[self.progress_slider.value()] + 1
        all_object_size = len(self.tracking_points_sam[self.progress_slider.value()])
        self.sam_obj_pos_label.setText(f"Annotation: {cur_id}/{all_object_size}")   
        
        if (0, 0) in self.lang_anno:
            self.video_lang_input.setText(self.lang_anno[(0, 0)])
        
        anno_loc, clip_text = self.get_clip_description()
        if anno_loc is not None:
            self.clip_lang_input.setText(f"Start Frame: {anno_loc[0]} | End Frame: {anno_loc[1]} | Description: {clip_text}")
        
    def pre_frame(self):
        if self.cur_frame_idx >= 1:
            self.cur_frame_idx -= 1
        if self.cur_frame_idx == 0:
            self.pre_f_button.setDisabled(True)
        self.next_f_button.setDisabled(False)
        self.video_position_label.setText(f"Frame: {self.cur_frame_idx}/{self.frame_count} | Video: {self.cur_video_idx}/{len(self.video_list)}")
        self.sam_object_id[self.cur_frame_idx] = 0
        
        if self.sam_object_id[self.cur_frame_idx] == 0:
            self.sam_pre_button.setDisabled(True)
        
        self.sam_next_button.setDisabled(False)
        self.update_frame(self.cur_frame_idx)
        self.progress_slider.setValue(self.cur_frame_idx)
        
        cur_id = self.sam_object_id[self.progress_slider.value()] + 1
        all_object_size = len(self.tracking_points_sam[self.progress_slider.value()])
        self.sam_obj_pos_label.setText(f"Annotation: {cur_id}/{all_object_size}")
        
        if (0, 0) in self.lang_anno:
            self.video_lang_input.setText(self.lang_anno[(0, 0)])
        
        anno_loc, clip_text = self.get_clip_description()
        if anno_loc is not None:
            self.clip_lang_input.setText(f"Start Frame: {anno_loc[0]} | End Frame: {anno_loc[1]} | Description: {clip_text}")    
    
    def request_video(self):
        if self.video_list[self.cur_video_idx-1] in self.video_cache:
            video = self.video_cache[self.cur_video_idx]
        else:
            video = request_video(self.video_list[self.cur_video_idx-1])
            self.video_cache[self.cur_video_idx-1] = video
        return video
    
    def request_video_async(self):
        class VideoThread(QThread):
            finished = pyqtSignal(object)
            def __init__(self, parent):
                super().__init__()
                self.parent = parent
            def run(self):
                res = self.parent.request_video()
                self.finished.emit(res)
        video_thread = VideoThread(self)
        return video_thread
    
    def request_sam_async(self):
        class SamThread(QThread):
            finished = pyqtSignal(object)
            def __init__(self, parent):
                super().__init__()
                self.parent = parent
            def run(self):
                res = self.parent.get_sam_result()
                self.finished.emit(res)
    
        sam_thread = SamThread(self)
        return sam_thread

    def request_cotracker_async(self, sam_config, co_tracker_config):
        class CoTrackerThread(QThread):
            finished = pyqtSignal(object)
            
            def __init__(self, parent, sam_config, co_tracker_config):
                super().__init__()
                self.parent = parent
                self.sam_config = sam_config
                self.co_tracker_config = co_tracker_config
            
            def run(self):
                res = self.parent.get_tap_result(self.sam_config, self.co_tracker_config)
                self.finished.emit(res)
    
        cotracker_thread = CoTrackerThread(self, sam_config, co_tracker_config)
        return cotracker_thread
    
    def save_result(self):
        # json.dump(self.anno, open(args.out_file, 'w'))
        pickle.dump(self.anno, open(args.out_file, 'wb'))
    
    def get_anno_result(self):
        if self.anno_function_select.currentText() == 'Sam':
            self.get_sam_async()
        elif self.anno_function_select.currentText() == 'Tracker':
            self.get_tap_async()
    
    def update_function_select(self):
        
        if self.anno_function_select.currentText() == 'Sam':
            self.button_param_select.clear()
            self.button_param_select.addItem('Frame Mode')
            self.button_param_select.addItem('Video Mode')
            self.sam_pre_button.show()
            self.sam_next_button.show()
            self.toolbar_layout.removeItem(self.edit_button_layout)
            self.toolbar_layout.insertLayout(2, self.sam_object_layout)
            self.edit_button.hide()
            self.close_edit_button.hide()
            self.sam_obj_pos_label.show()
            self.anno_mode = 'sam'
            
        elif self.anno_function_select.currentText() == 'Tracker':
            self.button_param_select.clear()
            self.button_param_select.addItem('Point Mode')
            self.button_param_select.addItem('Mask Mode')
            self.button_param_select.addItem('Grid Mode')
            self.sam_pre_button.hide()
            self.sam_next_button.hide()
            self.toolbar_layout.removeItem(self.sam_object_layout)
            self.toolbar_layout.insertLayout(2, self.edit_button_layout)
            self.edit_button.show()
            self.close_edit_button.show()
            self.sam_obj_pos_label.hide()
            self.anno_mode = 'tracker'
        
        self.vis_ori.setChecked(True)
        self.load_res()
    
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
        self.sam_object_id[frame_number] = 0
        self.update_frame(frame_number)

    def clear_sam_annotations(self):
        for k, _ in self.tracking_points_sam.items():
            self.tracking_points_sam[k] = [
                dict(pos=[], raw_pos=[], neg=[], raw_neg=[], labels=[])
            ]

        if self.last_frame is not None:
            self.draw_image()
    
    def clear_tap_annotations(self):
        for k, _ in self.tracking_points_tap.items():
            self.tracking_points_tap[k] = dict(pos=[], raw_pos=[])
        
        if self.last_frame is not None:
            self.draw_image()
    
    def clear_annotations(self):
        for k, _ in self.tracking_points_sam.items():
            self.tracking_points_sam[k] = [
                dict(pos=[], raw_pos=[], neg=[], raw_neg=[], labels=[])
            ]
        for k, _ in self.tracking_points_tap.items():
            self.tracking_points_tap[k] = dict(pos=[], raw_pos=[])
        
        self.sam_next_button.setDisabled(False)
        self.sam_pre_button.setDisabled(False)
        self.sam_object_id = [0] * self.frame_count
        
        if self.last_frame is not None:
            self.draw_image()

    def clear_keyframes(self):
        self.keyframes = {}
        self.update_keyframe_bar()
        self.lang_anno = dict()
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
        self.video_label.clear()
        self.progress_slider.setValue(0)
        self.frame_position_label.hide()
        self.keyframes = {}
        self.selected_keyframe = None
        self.update_keyframe_bar()
        self.tracking_points_sam = dict()
        self.tracking_points_tap = dict()
        self.tracking_masks = dict()
        self.last_frame = None
        self.tracker_res = []
        self.sam_res = []
        self.cur_frame_idx = 0
        self.sam_object_id = [0] * self.frame_count
        self.vis_ori.setChecked(True)
        self.lang_anno = dict()
        self.video_lang_input.clear()
        self.clip_lang_input.clear()
        self.video_position_label.setText(f"Frame: -/- | Video: {self.cur_video_idx}/{len(self.video_list)}")
        self.pre_f_button.setDisabled(True)
    
    def remove_last_sam_annotation(self):
        
        sam_object_id = self.sam_object_id[self.progress_slider.value()]
        click_action = self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['labels']
        pos_click_position = self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['pos']
        neg_click_position = self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['neg']
        
        if len(click_action) > 0 and click_action[-1] == 1 and len(pos_click_position) > 0:
            self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['pos'].pop()
            self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['raw_pos'].pop()
            self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['labels'].pop()
        elif len(click_action) > 0 and click_action[-1] == -1 and len(neg_click_position) > 0:
            self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['neg'].pop()
            self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['raw_neg'].pop()
            self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['labels'].pop()
        
        if self.last_frame is not None:
            self.draw_image()
    
    def remove_last_tap_annotation(self):
        self.tracking_points_tap[self.progress_slider.value()]['pos'].pop()
        self.tracking_points_tap[self.progress_slider.value()]['raw_pos'].pop()
        if self.last_frame is not None:
            self.draw_image()
        
    def load_video_callback(self, video):
        if video is not None:
            self.load_video(video)
            QMessageBox.information(self, "Success", "视频加载完成!")
            self.progress.close()
        else:
            QMessageBox.warning(self, "Error", "视频加载失败，请检查网络设置")
            self.progress.close()
            return
        
    def load_video_async(self):
        self.progress = QProgressDialog("请等待，正在加载视频...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None)
        self.progress.setMinimumDuration(0)  # 立即显示对话框
        self.progress.show()
        
        self.video_thread = self.request_video_async()
        self.video_thread.finished.connect(self.load_video_callback)
        self.video_thread.start()
    
    def load_video(self, video):
        # video = self.request_video()
        if video is None:
            return -1
        
        self.sam_config['video_path'] = self.video_list[self.cur_video_idx-1]
        self.co_tracker_config['video_path'] = self.video_list[self.cur_video_idx-1]
        self.frame_count = video.shape[0]
        self.sam_object_id = [0] * self.frame_count
        for i in range(self.frame_count):
            self.tracking_points_sam[i] = [dict(
                    pos=[], raw_pos=[], neg=[], raw_neg=[], labels=[]
            )]
            self.tracking_points_tap[i] = dict(
                pos=[], raw_pos=[]
            )
            self.tracking_masks[i] = []
        
        self.vis_ori.setChecked(True)      
        self.ori_video = np.array(video)
        self.progress_slider.setMaximum(self.frame_count - 1)
        self.update_frame(0)
        self.progress_slider.show()
        self.frame_position_label.show()
        self.update_keyframe_bar()  # Initialize keyframe bar
        self.video_position_label.setText(f"Frame: {self.cur_frame_idx}/{self.frame_count} | Video: {self.cur_video_idx}/{len(self.video_list)}")
        self.pre_f_button.setDisabled(True)
        self.sam_obj_pos_label.setText("Annotation Object: 1/1")
        video_name = self.video_list[self.cur_video_idx-1]
        if video_name not in self.anno:
            self.anno[video_name] = dict(
                mask=[], track=[]
            )
        self.seek_video()
        return 1
            
    def update_frame(self, frame_number):
        if len(self.ori_video) == 0:
            self.smart_message('Please load video first')
            return
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
        if self.last_frame is None:
            return 
        frame_number = self.progress_slider.value()
        self.update_frame(frame_number)
        self.cur_frame_idx = self.progress_slider.value()
        self.video_position_label.setText(f"Frame: {self.cur_frame_idx}/{self.frame_count} | Video: {self.cur_video_idx}/{len(self.video_list)}")

        self.pre_f_button.setDisabled(False)
        self.next_f_button.setDisabled(False)
        
        if self.cur_frame_idx == 0:
            self.pre_f_button.setDisabled(True)
        if self.cur_frame_idx == self.frame_count-1:
            self.next_f_button.setDisabled(True)
        
        self.sam_object_id[self.cur_frame_idx] = 0
        self.sam_obj_pos_label.setText(f"Annotation Object: {self.sam_object_id[self.cur_frame_idx]+1}/{len(self.tracking_points_sam[self.cur_frame_idx])}")
        self.sam_pre_button.setDisabled(True)
        self.sam_next_button.setDisabled(False)
        
        if (0, 0) in self.lang_anno:
            self.video_lang_input.setText(self.lang_anno[(0, 0)])
        
        anno_loc, clip_text = self.get_clip_description()
        if anno_loc is not None:
            self.clip_lang_input.setText(f"Start Frame: {anno_loc[0]} | End Frame: {anno_loc[1]} | Description: {clip_text}")
            
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
        # check if the frame number has keyframe
        if frame_number in self.keyframes:
            keyframe_type = self.keyframes[frame_number]
            self.frame_position_label.setText(f"Frame: {frame_number} {keyframe_type}")
        else:
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
        
        tracking_points = self.tracking_points_sam
        positive_points_all = {}
        negative_points_all = {}
        labels_all = {}
        
        if self.button_param_select.currentText() == 'Frame Mode':
            is_video = False
        elif self.button_param_select.currentText() == 'Video Mode':
            is_video = True
        else:
            raise ValueError('Please select the sam mode')

        select_frame = self.progress_slider.value()
        try:
            frame_pts = tracking_points[select_frame]
        except:
            self.smart_message('Please load video first')
            return -1
        
        has_pos_points = False
        # select all objects
        for obj_id, obj_pts in enumerate(frame_pts):
            positive_points, negative_points, labels = [], [], []
            if obj_pts['raw_pos'] != []:
                positive_points.extend([[pt.x(), pt.y()] for pt in obj_pts['raw_pos']])
                has_pos_points = True
            if obj_pts['raw_neg'] != []:
                negative_points.extend([[pt.x(), pt.y()] for pt in obj_pts['raw_neg']])
            if (obj_pts['raw_pos'] != []) or (obj_pts['raw_neg'] != []):
                labels.extend(obj_pts['labels'])
            
            positive_points_all[obj_id] = positive_points
            negative_points_all[obj_id] = negative_points
            labels_all[obj_id] = labels

        if not has_pos_points:
            self.smart_message('Please select at least one positive point')
            return -1
        
        self.sam_config['is_video'] = is_video
        self.sam_config['positive_points'] = positive_points_all
        self.sam_config['negative_points'] = negative_points_all
        self.sam_config['labels'] = labels_all
        self.sam_config['select_frame'] = select_frame
        
        return 0
    
    def smart_message(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle('提示')
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
    
    def sam_callback(self, res):
        self.sam_thread.wait()
        if res == 1:
            self.vis_sam.setChecked(True)
            self.load_res()
            QMessageBox.information(self, "Success", "SAM处理完成!")
            self.progress.close()
        else:
            QMessageBox.warning(self, "Error", "SAM处理失败，请重试")
            self.progress.close()
            return
    
    def get_sam_async(self):
        self.progress = QProgressDialog("请等待，正在请求Sam结果...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None)
        self.progress.setMinimumDuration(0)
        self.progress.show()
        
        self.sam_thread = self.request_sam_async()
        self.sam_thread.finished.connect(self.sam_callback)
        self.sam_thread.start()
    
    def get_sam_result(self):
        res = self.set_sam_config()
        if res == -1:
            return -1
        
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
    
        return 1
    
    def tracker_callback(self, res):
        self.tracker_thread.wait()
        if res == 1:
            self.vis_tracker.setChecked(True)
            self.load_res()
            QMessageBox.information(self, "Success", "Tracker处理完成!")
            self.progress.close()
        else:
            QMessageBox.warning(self, "Error", "Tracker处理失败，请重试")
            self.progress.close()
            return
            
    def get_tap_async(self):
        self.progress = QProgressDialog("请等待，正在请求Tracker结果...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None)
        self.progress.setMinimumDuration(0)
        self.progress.show()
        
        self.set_tap_config()
        
        self.tracker_thread = self.request_cotracker_async(self.sam_config, self.co_tracker_config)
        self.tracker_thread.finished.connect(self.tracker_callback)
        self.tracker_thread.start()
    
    def set_tap_config(self):
        
        self.co_tracker_config['mode'] = self.button_param_select.currentText()
        if self.co_tracker_config['mode'] == 'Point Mode':
            points, select_frame = [], []
            for frame_id, frame_pts in self.tracking_points_tap.items():
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
    
    def get_tap_result(self, sam_config, co_tracker_config):        
        
        pred_tracks, pred_visibility, images = request_cotracker(sam_config, co_tracker_config)
        
        if pred_tracks is None:
            return -1
        
        self.tracker_res = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images])
        self.anno[self.video_list[self.cur_video_idx-1]]['track'] = (pred_tracks, pred_visibility)
        
        
        return 1
    
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
            if self.anno_mode == 'sam':
                sam_object_id = self.sam_object_id[self.progress_slider.value()]
                self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['raw_pos'].append(original_position)
                self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['pos'].append(click_position)
                self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['labels'].append(1)
                self.sam_next_button.setDisabled(False)
            else:
                self.tracking_points_tap[self.progress_slider.value()]['raw_pos'].append(original_position)
                self.tracking_points_tap[self.progress_slider.value()]['pos'].append(click_position)
            
            # Draw a point on the frame
        elif event.button() == Qt.RightButton and self.last_frame is not None:
            pos = self.video_label.mapFromGlobal(event.globalPos())
            gt_pos = self.get_align_point(pos.x(), pos.y())
            if gt_pos is None:
                return
            click_position = QPoint(gt_pos[0], gt_pos[1])
            original_position = QPoint(int(gt_pos[0]//self.scale), int(gt_pos[1]//self.scale))
            if self.anno_mode == 'sam':
                sam_object_id = self.sam_object_id[self.progress_slider.value()]
                self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['neg'].append(click_position)
                self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['raw_neg'].append(original_position)
                self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['labels'].append(-1)
                self.sam_next_button.setDisabled(False)
            else:
                self.tracking_points_tap[self.progress_slider.value()]['raw_pos'].append(original_position)
                self.tracking_points_tap[self.progress_slider.value()]['pos'].append(click_position)
            
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
        if self.last_frame is None:
            return
        frame = self.last_frame.copy()
        if self.is_edit_mode:
            x, y = int(self.cur_edit_click[0]), int(self.cur_edit_click[1])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            if self.anno_mode == 'sam':
                sam_object_id = self.sam_object_id[self.progress_slider.value()]
                pos_click_position = self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['pos']
                neg_click_position = self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['neg']
            else:
                pos_click_position = self.tracking_points_tap[self.progress_slider.value()]['pos']
                neg_click_position = []

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
        if len(self.ori_video) == 0:
            self.smart_message('请先加载视频')
            return
        print(self.description_mode.currentText())
        if self.description_mode.currentText() == 'Frame':
            self.add_frame_discribtion()
        elif self.description_mode.currentText() == 'Video':
            self.add_video_description()

    def mark_keyframe(self):
        current_frame = self.progress_slider.value()
        if self.key_frame_mode == 'Start':
            self.keyframes[current_frame] = 'start'
        elif self.key_frame_mode == 'End':
            self.keyframes[current_frame] = 'end'
        self.update_keyframe_bar()
    
    def update_lang_anno(self):
        key_frame_list = sorted(self.keyframes.keys())
        key_pairs = []
        if len(key_frame_list) <= 1:
            self.smart_message('请先标记关键帧')
            return -1
        
        for i in range(0, len(key_frame_list), 2):
            start_frame = key_frame_list[i]
            end_frame = key_frame_list[i+1]
            if self.keyframes[start_frame] != 'start' or self.keyframes[end_frame] != 'end':
                self.smart_message('请检查关键帧标记是否正确，必须是start和end交替出现')
                return -1
            key_pairs.append((start_frame, end_frame))
            
        for i in key_pairs:
            if i not in self.lang_anno:
                self.lang_anno[i] = None

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
    
    def show_tooltip(self, event):
        # Show tooltip when hovering over a keyframe
        mouse_pos = event.pos()
        for frame, key_type in self.keyframes.items():
            x_position = int((frame / self.frame_count) * self.keyframe_bar.width())
            if abs(mouse_pos.x() - x_position) <= 5:  # Small range to detect hover
                QToolTip.showText(self.keyframe_bar.mapToGlobal(mouse_pos), f"Frame: {frame}")
                return
        QToolTip.hideText()

    def closeEvent(self, event):
        # if self.cap is not None:
        #     self.cap.release()
        event.accept()
     
    def keyPressEvent(self, event):
        if len(self.ori_video) == 0:
            self.smart_message('请先加载视频')
            return        
        key = event.key()
        if key == Qt.Key_A:
            self.pre_frame()
        elif key == Qt.Key_D:
            self.next_frame()
        elif key == Qt.Key_W:
            self.key_frame_mode = 'Start'
            self.mark_keyframe()
            self.update_frame_position_label()
        elif key == Qt.Key_S:
            self.key_frame_mode = 'End'
            self.mark_keyframe()
            self.update_frame_position_label()
        elif key == Qt.Key_Backspace:
            self.selected_keyframe = self.progress_slider.value()
            self.remove_keyframe()
            self.update_frame_position_label()
        # Enter key pressed
        elif key == Qt.Key_Return:
            self.add_frame_discribtion()
     
    def add_frame_discribtion(self):
        frame_number = self.progress_slider.value()
        if self.update_lang_anno() == -1:
            return
        key_pairs = list(self.lang_anno.keys())
        has_key = [i[0] <= frame_number <= i[1] for i in key_pairs].count(True) > 0
        if not has_key:
            self.smart_message('请先标记当前所在区域的起止帧')
            return

        # load the cached description
        anno_loc = [i for i in key_pairs if i[0] < frame_number <= i[1]]
        if len(anno_loc) == 0:
            self.smart_message('请移动到所在区域的起止帧之间')
            return
        anno_loc = anno_loc[0]
        
        if self.lang_anno[anno_loc] is not None:
            cached_lang = self.lang_anno[anno_loc]
        else:
            cached_lang = ''
        # Create a dialog to get the description from the user
        dialog = TextInputDialog(cached_lang, self)
        if dialog.exec_() == QDialog.Accepted:
            cached_lang = dialog.get_text()
            print(cached_lang)
            self.lang_anno[anno_loc] = cached_lang
            self.clip_lang_input.setText(f"Start Frame: {anno_loc[0]} | End Frame: {anno_loc[1]} | Description: {cached_lang}")
        else:
            return 
        
    def add_video_description(self):
        # Create a dialog to get the description from the user
        if (0, 0) in self.lang_anno:
            cached_lang = self.lang_anno[(0, 0)]
        else:
            cached_lang = ''
        
        dialog = TextInputDialog(cached_lang, self)
        if dialog.exec_() == QDialog.Accepted:
            video_description = dialog.get_text()
            self.lang_anno[(0, 0)] = video_description
            self.video_lang_input.setText(f"Video Description: {video_description}")
        else:
            return

    def get_clip_description(self):
        # Get the description for the clip
        key_pairs = list(self.lang_anno.keys())
        frame_number = self.progress_slider.value()
        anno_loc = [i for i in key_pairs if i[0] < frame_number <= i[1]]
        if len(anno_loc) > 0:
            return anno_loc[0], self.lang_anno[anno_loc[0]]
        return None, None

            


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