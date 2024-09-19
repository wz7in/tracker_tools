import sys
import os, time
import json
import argparse
import pickle
from PyQt5.QtCore import QPoint, QTimer, Qt
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox, QLineEdit, QDialogButtonBox, QTextEdit, QGridLayout,
                             QLabel, QSlider, QDialog, QHBoxLayout, QFrame, QProgressDialog, QRadioButton, QToolTip, QComboBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QMouseEvent, QBrush
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal, QThread

import yaml
from client_utils import request_sam, request_cotracker, request_video
import numpy as np

def load_anno_file(anno_file, out_file):
    with open(anno_file, 'r') as f:
        video_list = f.readlines()
        video_list = [x.strip() for x in video_list]
    
    if os.path.exists(out_file):
        anno = pickle.load(open(out_file, 'rb'))
    else:
        anno = {}
        
    return video_list, anno

class TextInputDialog(QDialog):
    
    def __init__(self, initial_text='', parent=None, is_video=True):
        super().__init__(parent)
        self.setWindowTitle('请输入语言标注')
        self.is_video = is_video
        
        self.main_layout = QGridLayout(self)
        
        self.text_title = QLabel('请输入语言标注:', self) 
        self.text_input = QLineEdit(self)
        self.text_input.setFixedSize(300,20)
        
        self.text_input.setText(initial_text)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        
        if not is_video:
            self.prim_title = QLabel('请选择操作类型:', self)
            self.mode_select = QComboBox()
            self.mode_select.addItems(['Open', 'Close', 'Move'])
            self.main_layout.addWidget(self.prim_title, 1, 0)
            self.main_layout.addWidget(self.mode_select, 1, 1)
            self.main_layout.addWidget(self.text_title, 0, 0)
            self.main_layout.addWidget(self.text_input, 0, 1)
            self.main_layout.addWidget(self.button_box, 2, 0, 1, 2)
        else:
            self.main_layout.addWidget(self.text_title, 0, 0)
            self.main_layout.addWidget(self.text_input, 0, 1)
            self.main_layout.addWidget(self.button_box, 1, 0, 1, 2)
        
        
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
    def get_text(self):
        return self.text_input.text()
    
    def get_prim(self):
        if not self.is_video:
            return self.mode_select.currentText()
        else:
            return ''

class VideoPlayer(QWidget):
    def __init__(self, args):
        
        # load video list
        self.video_list, self.all_anno = load_anno_file(args.anno_file, args.out_file)
        
        super().__init__()
        self.setWindowTitle("浦器实验室视频标注工具")
        
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
        self.frame_position_label.setStyleSheet("background-color: #E3E3E3;")
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
        self.video_position_label.setStyleSheet("background-color: #E3E3E3;")
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
        self.anno_function_select.addItem('Interpolation')
        
        # button params for different functions
        self.button_param_select = QComboBox()
        self.button_param_select.addItem('Video Mode')
        self.button_param_select.addItem('Frame Mode')
        
        # display tracking point numbers in interpolation mode
        self.track_point_num_label = QLabel(self)
        self.track_point_num_label.setStyleSheet("background-color: #E3E3E3;")
        self.track_point_num_label.setAlignment(Qt.AlignCenter)
        self.track_point_num_label.setFixedSize(150, 20)
        self.track_point_num_label.hide()
        
        self.track_mode_selector = QComboBox()
        self.track_mode_selector.addItems(['BiDirection', 'Forward'])
        self.track_mode_selector.hide()
        
        self.anno_function_select.currentIndexChanged.connect(self.update_function_select)
    
        click_action_button = QPushButton("Run", self)
        # select color
        click_action_button.clicked.connect(self.get_anno_result)
        anno_button_layout.addWidget(self.anno_function_select)
        anno_button_layout.addWidget(self.button_param_select)
        anno_button_layout.addWidget(self.track_mode_selector)
        anno_button_layout.addWidget(self.track_point_num_label)
        anno_button_layout.addWidget(click_action_button)
        self.toolbar_layout.addLayout(anno_button_layout)
        
        self.sam_object_layout = QHBoxLayout()
        
        self.sam_pre_button = QPushButton("Pre Object", self)
        self.sam_pre_button.clicked.connect(self.pre_sam_object)
        self.sam_pre_button.setDisabled(True)
        
        self.sam_obj_pos_label = QLabel(self)
        self.sam_obj_pos_label.setStyleSheet("background-color: #E3E3E3;")
        self.sam_obj_pos_label.setAlignment(Qt.AlignCenter)
        self.sam_obj_pos_label.setFixedSize(150, 20)
        
        self.sam_next_button = QPushButton("Next/Add Object", self)
        self.sam_next_button.clicked.connect(self.next_sam_object)
        self.sam_next_button.setDisabled(True)
        
        self.sam_object_layout.addWidget(self.sam_pre_button)
        self.sam_object_layout.addWidget(self.sam_obj_pos_label)
        self.sam_object_layout.addWidget(self.sam_next_button)
        self.toolbar_layout.addLayout(self.sam_object_layout)
        
        # self.edit_button_layout = QHBoxLayout()

        # self.edit_button = QPushButton("Edit keypoints", self)
        # self.edit_button.clicked.connect(self.edit_track_pts)
        # self.edit_button_layout.addWidget(self.edit_button)
        # self.edit_button.hide()

        # self.close_edit_button = QPushButton("Save Edit", self)
        # self.close_edit_button.clicked.connect(self.close_edit)
        # self.edit_button_layout.addWidget(self.close_edit_button)
        # self.close_edit_button.hide()
        
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
        self.clear_all_button = QPushButton("Remove All Annotation", self)
        self.clear_all_button.clicked.connect(self.clear_annotations)
        self.control_button_layout.addWidget(self.clear_all_button)
        
        # remove_last_button
        self.remove_last_button = QPushButton("Remove Last Annotation", self)
        self.remove_last_button.clicked.connect(self.remove_last_annotation)
        self.control_button_layout.addWidget(self.remove_last_button)
        
        self.remove_frame_button = QPushButton("Remove Frame Annotation", self)
        self.remove_frame_button.clicked.connect(self.remove_frame_annotation)
        self.control_button_layout.addWidget(self.remove_frame_button)
        
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.load_and_save_result)
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
        self.video_lang_input.setFixedHeight(60)
        self.video_lang_input.setStyleSheet("background-color: #E3E3E3; font-weight: bold;")
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
        self.clip_lang_input.setFixedHeight(80)
        self.clip_lang_input.setStyleSheet("background-color: #E3E3E3; font-weight: bold;")
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
            "KEY_Q:\tStart Key Frame\tKEY_ENTE:\tAdd Language\nKEY_E:\tEnd Key Frame\tKEY_BACK:\tDelete Key Frame\nKEY_A:\tPre Frame\nKEY_D:\tNext Frame")
        self.tips_input.setReadOnly(True)
        self.tips_input.setFixedHeight(75)
        self.tips_input.setStyleSheet("background-color: #E3E3E3; font-weight: bold;")
        self.tips_layout.addWidget(self.tips_input)
        self.toolbar_layout.addLayout(self.tips_layout)
        
        # Add toolbar layout to the main layout
        main_layout.addLayout(self.toolbar_layout)

        self.setLayout(main_layout)

        self.cur_video_idx = 1
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
        self.frame_count = 0
        self.last_frame = None
        self.tracking_points_sam = dict()
        self.tracking_points_tap = dict()
        self.tracking_points_interp = dict()
        self.ori_video = {}
        self.anno = {}
        self.vis_track_res = False
        self.sam_res = dict()
        self.track_res = dict()
        self.lang_anno = dict()
        self.video_cache = dict()
        self.max_point_num = dict()
        self.anno_mode = 'sam'
        self.cur_frame_idx = self.progress_slider.value()
        self.pre_f_button.setDisabled(True)
        self.next_f_button.setDisabled(True)
        self.is_edit_mode = False
        self.edit_track_res = None
        self.key_frame_mode = 'Start'
        
    def pre_sam_object(self):
        if self.sam_object_id[self.progress_slider.value()] > 0:
            self.sam_object_id[self.progress_slider.value()] -= 1
        if self.sam_object_id[self.progress_slider.value()] == 0:
            self.sam_pre_button.setDisabled(True)
        
        self.sam_next_button.setDisabled(False)
        cur_id = self.sam_object_id[self.progress_slider.value()] + 1
        all_object_size = len(self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()])
        self.sam_obj_pos_label.setText(f"Annotation: {cur_id}/{all_object_size}")
        
        self.draw_image()
    
    def resizeEvent(self, event):
        self.seek_video()
        self.clear_keyframes()
        self.update_keyframe_bar()
        
        self.setAutoFillBackground(False)
        palette = self.palette()
        palette.setBrush(self.backgroundRole(), QBrush(QPixmap('./demo/bg.png').scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)))
        self.setPalette(palette)
        if self.video_list[self.cur_video_idx-1] in self.ori_video:
            self.keyframe_bar.show()
        else:
            self.keyframe_bar.hide()
    
    def next_sam_object(self):
        
        self.sam_object_id[self.progress_slider.value()] += 1
        if self.sam_object_id[self.progress_slider.value()] == len(self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]):
            self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()].append(
                dict(pos=[], raw_pos=[], neg=[], raw_neg=[], labels=[])
            )
        self.sam_pre_button.setDisabled(False)
        
        if len(self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][self.sam_object_id[self.progress_slider.value()]]['pos']) == 0:
            self.sam_next_button.setDisabled(True)
        else:
            self.sam_next_button.setDisabled(False)
        
        cur_id = self.sam_object_id[self.progress_slider.value()] + 1
        all_object_size = len(self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()])
        self.sam_obj_pos_label.setText(f"Annotation: {cur_id}/{all_object_size}")
        
        self.draw_image()
    
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
        all_object_size = len(self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()])
        self.sam_obj_pos_label.setText(f"Annotation: {cur_id}/{all_object_size}")   
        
        if (0, 0) in self.lang_anno[self.video_list[self.cur_video_idx-1]]:
            self.video_lang_input.setText(self.lang_anno[self.video_list[self.cur_video_idx-1]][(0, 0)])
        else:
            self.video_lang_input.setText('')
        
        anno_loc, (clip_text, prim) = self.get_clip_description()
        if anno_loc is not None:
            self.clip_lang_input.setText(f"Start Frame: {anno_loc[0]} | End Frame: {anno_loc[1]}\nPrim: {prim}\nDescription: {clip_text}")
        else:
            self.clip_lang_input.setText('')
        
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
        all_object_size = len(self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()])
        self.sam_obj_pos_label.setText(f"Annotation: {cur_id}/{all_object_size}")
        
        if (0, 0) in self.lang_anno[self.video_list[self.cur_video_idx-1]]:
            self.video_lang_input.setText(self.lang_anno[self.video_list[self.cur_video_idx-1]][(0, 0)])
        else:
            self.video_lang_input.setText('')
        
        anno_loc, (clip_text, prim) = self.get_clip_description()
        if anno_loc is not None:
            self.clip_lang_input.setText(f"Start Frame: {anno_loc[0]} | End Frame: {anno_loc[1]}\nPrim: {prim}\nDescription: {clip_text}")
        else:
            self.clip_lang_input.setText('') 
    
    def request_video(self):
        if self.video_list[self.cur_video_idx-1] in self.video_cache:
            video = self.video_cache[self.video_list[self.cur_video_idx-1]]
        else:
            try:
                video = request_video(self.video_list[self.cur_video_idx-1])
            except Exception as e:
                return None
            if video is None:
                return None
            self.video_cache[self.video_list[self.cur_video_idx-1]] = video
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
    
    def load_and_save_result(self):
        lang_res, sam_res, track_res = dict(), dict(), dict()
        #################### parse video language annotation ####################
        if (0, 0) in self.lang_anno[self.video_list[self.cur_video_idx-1]] and self.lang_anno[self.video_list[self.cur_video_idx-1]][(0, 0)] != '':
            lang_res['video'] = self.lang_anno[self.video_list[self.cur_video_idx-1]][(0, 0)]
        else:
            self.smart_message("Please annotation the whole video language")
            return 
        #################### parse clip language annotation ####################
        lang_res['clip'] = []
        for clip_range, lang in self.lang_anno[self.video_list[self.cur_video_idx-1]].items():
            if clip_range == (0, 0):
                continue
            if len(lang) == 0:
                self.smart_message(f"Please annotation the clip language between frame {clip_range[0]} and {clip_range[1]}")
                return
            tmp_lang = dict()
            tmp_lang['start_frame'] = clip_range[0]
            tmp_lang['end_frame'] = clip_range[1]
            tmp_lang['description'] = lang
            lang_res['clip'].append(tmp_lang)
        #################### parse sam annotation ####################
        sam_res = self.anno[self.video_list[self.cur_video_idx-1]]['sam']
        #################### parse tracker annotation ####################
        track_res['track'] = self.anno[self.video_list[self.cur_video_idx-1]]['track']
        track_res['visibility'] = self.anno[self.video_list[self.cur_video_idx-1]]['visibility']
        
        # load original annotation
        if self.video_list[self.cur_video_idx-1] in self.all_anno and len(self.all_anno[self.video_list[self.cur_video_idx-1]]) > 0:
            reply = QMessageBox.question(
                self,
                "Confirmation",
                "是否要更新原始的标注？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        self.all_anno[self.video_list[self.cur_video_idx-1]] = dict()
        self.all_anno[self.video_list[self.cur_video_idx-1]]['lang'] = lang_res
        self.all_anno[self.video_list[self.cur_video_idx-1]]['sam'] = sam_res
        self.all_anno[self.video_list[self.cur_video_idx-1]]['track'] = track_res
        
        pickle.dump(self.all_anno, open(args.out_file, 'wb'))
        self.smart_message("Save Successfully!")
    
    def get_anno_result(self):
        if self.anno_function_select.currentText() == 'Sam':
            self.get_sam_async()
        elif self.anno_function_select.currentText() == 'Tracker':
            self.get_tap_async()
        elif self.anno_function_select.currentText() == 'Interpolation':
            self.get_interp_result()
    
    def update_function_select(self):
        
        if self.anno_function_select.currentText() == 'Sam':
            self.button_param_select.clear()
            self.button_param_select.addItem('Frame Mode')
            self.button_param_select.addItem('Video Mode')
            self.sam_pre_button.show()
            self.sam_next_button.show()
            self.button_param_select.show()
            self.toolbar_layout.insertLayout(2, self.sam_object_layout)
            self.track_mode_selector.hide()
            self.track_point_num_label.hide()
            self.sam_obj_pos_label.show()
            self.anno_mode = 'sam'
            self.progress_slider.setValue(0)
            
        elif self.anno_function_select.currentText() == 'Tracker':
            self.button_param_select.clear()
            self.button_param_select.addItem('Point Mode')
            self.button_param_select.addItem('Mask Mode')
            self.button_param_select.addItem('Grid Mode')
            self.button_param_select.show()
            self.sam_pre_button.hide()
            self.sam_next_button.hide()
            self.track_mode_selector.show()
            self.toolbar_layout.removeItem(self.sam_object_layout)
            self.sam_obj_pos_label.hide()
            self.track_point_num_label.hide()
            self.anno_mode = 'tracker'
            self.progress_slider.setValue(0)
        
        elif self.anno_function_select.currentText() == 'Interpolation':
            self.button_param_select.clear()
            self.button_param_select.hide()
            self.sam_pre_button.hide()
            self.sam_next_button.hide()
            self.toolbar_layout.removeItem(self.sam_object_layout)
            self.sam_obj_pos_label.hide()
            self.track_mode_selector.hide()
            self.track_point_num_label.show()
            self.anno_mode = 'interpolation'
            self.max_point_num[self.video_list[self.cur_video_idx-1]] = 0
            self.progress_slider.setValue(0)
        
        self.vis_ori.setChecked(True)
        self.load_res()
    
    def load_res(self):
        if self.video_list[self.cur_video_idx-1] not in self.ori_video:
            self.smart_message("Please load video first!")
            return

        if self.vis_sam.isChecked():
            self.vis_track_res = True
            self.parse_res = self.sam_res[self.video_list[self.cur_video_idx-1]]
        elif self.vis_tracker.isChecked():
            self.vis_track_res = True
            self.parse_res = self.track_res[self.video_list[self.cur_video_idx-1]]
        else:
            self.vis_track_res = False
            
        frame_number = self.progress_slider.value()
        self.sam_object_id[frame_number] = 0
        self.update_frame(frame_number)
    
    def clear_annotations(self):
        if self.video_list[self.cur_video_idx-1] not in self.tracking_points_sam:
            self.smart_message("Please load video first!")
            return
        
        if self.video_list[self.cur_video_idx-1] not in self.tracking_points_tap:
            self.smart_message("Please load video first!")
            return

        if self.video_list[self.cur_video_idx-1] not in self.tracking_points_interp:
            self.smart_message("Please load video first!")
            return
        
        for k, _ in self.tracking_points_sam[self.video_list[self.cur_video_idx-1]].items():
            self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][k] = [
                dict(pos=[], raw_pos=[], neg=[], raw_neg=[], labels=[])
            ]
        for k, _ in self.tracking_points_tap[self.video_list[self.cur_video_idx-1]].items():
            self.tracking_points_tap[self.video_list[self.cur_video_idx-1]][k] = dict(pos=[], raw_pos=[])
        
        for k, _ in self.tracking_points_interp[self.video_list[self.cur_video_idx-1]].items():
            self.tracking_points_interp[self.video_list[self.cur_video_idx-1]][k] = dict(pos=[], raw_pos=[])
        
        self.anno[self.video_list[self.cur_video_idx-1]] = dict(
            sam=None, track=None, visibility=None
        )
        self.lang_anno[self.video_list[self.cur_video_idx-1]] = dict()
        
        self.sam_next_button.setDisabled(False)
        self.sam_pre_button.setDisabled(False)
        self.sam_object_id = [0] * self.frame_count
        
        if self.last_frame is not None:
            self.draw_image()

    def clear_keyframes(self):
        self.keyframes[self.video_list[self.cur_video_idx-1]] = {}
        self.lang_anno[self.video_list[self.cur_video_idx-1]] = dict()
        if self.last_frame is not None:
            self.draw_image()
        self.update_keyframe_bar()
           
    def clear_video(self):
        self.video_label.clear()
        self.progress_slider.setValue(0)
        self.frame_position_label.hide()
        self.keyframes = {}
        self.selected_keyframe = None
        self.update_keyframe_bar()
        self.keyframe_bar.hide()
        self.tracking_points_sam = dict()
        self.tracking_points_tap = dict()
        self.tracking_points_interp = dict()
        self.last_frame = None
        self.cur_frame_idx = 0
        self.sam_object_id = [0] * self.frame_count
        self.vis_ori.setChecked(True)
        self.lang_anno[self.video_list[self.cur_video_idx-1]] = dict()
        self.video_lang_input.clear()
        self.clip_lang_input.clear()
        self.video_position_label.setText(f"Frame: -/- | Video: {self.cur_video_idx}/{len(self.video_list)}")
        self.pre_f_button.setDisabled(True)
    
    def remove_last_sam_annotation(self):
        if self.video_list[self.cur_video_idx-1] not in self.tracking_points_sam:
            self.smart_message("Please load video first!")
            return
        
        sam_object_id = self.sam_object_id[self.progress_slider.value()]
        click_action = self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['labels']
        pos_click_position = self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['pos']
        neg_click_position = self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['neg']
        
        if len(click_action) > 0 and click_action[-1] == 1 and len(pos_click_position) > 0:
            self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['pos'].pop()
            self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['raw_pos'].pop()
            self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['labels'].pop()
        elif len(click_action) > 0 and click_action[-1] == -1 and len(neg_click_position) > 0:
            self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['neg'].pop()
            self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['raw_neg'].pop()
            self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['labels'].pop()
        
        if self.last_frame is not None:
            self.draw_image()
    
    def remove_last_tap_annotation(self):
        
        if self.video_list[self.cur_video_idx-1] not in self.tracking_points_tap:
            self.smart_message("Please load video first!")
            return
        
        self.tracking_points_tap[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]['pos'].pop()
        self.tracking_points_tap[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]['raw_pos'].pop()
        if self.last_frame is not None:
            self.draw_image()
    
    def remove_last_interp_annotation(self):
        if self.video_list[self.cur_video_idx-1] not in self.tracking_points_interp:
            self.smart_message("Please load video first!")
            return
        
        self.tracking_points_interp[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]['pos'].pop()
        self.tracking_points_interp[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]['raw_pos'].pop()
        
        update_max_point_num = [len(self.tracking_points_interp[self.video_list[self.cur_video_idx-1]][i]['pos']) for i in range(self.frame_count)]
        self.max_point_num[self.video_list[self.cur_video_idx-1]] = max(update_max_point_num)
        
        if self.last_frame is not None:
            self.draw_image()
        
    def remove_last_annotation(self):
        if self.anno_function_select.currentText() == 'Sam':
            self.remove_last_sam_annotation()
        elif self.anno_function_select.currentText() == 'Tracker':
            self.remove_last_tap_annotation()
        elif self.anno_function_select.currentText() == 'Interpolation':
            self.remove_last_interp_annotation()
    
    def remove_frame_annotation(self):
        if self.video_list[self.cur_video_idx-1] not in self.tracking_points_sam:
            self.smart_message("Please load video first!")
            return
        if self.anno_function_select.currentText() == 'Sam':
            self.sam_object_id[self.progress_slider.value()] = 0
            self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()] = [dict(
                pos=[], raw_pos=[], neg=[], raw_neg=[], labels=[]
            )]
        elif self.anno_function_select.currentText() == 'Tracker':
            self.tracking_points_tap[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()] = dict(
                pos=[], raw_pos=[]
            )
        elif self.anno_function_select.currentText() == 'Interpolation':
            self.tracking_points_interp[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()] = dict(
                pos=[], raw_pos=[]
            )
        if self.last_frame is not None:
            self.draw_image()
    
    def load_video_callback(self, video):
        if video is not None:
            self.load_video(video)
            self.progress.close()
            self.smart_message("视频加载完成!")
        else:
            self.progress.close()
            self.smart_message("视频加载失败，请检查网络设置")
            return
        
    def load_video_async(self):
        self.progress = QProgressDialog("请等待，正在加载视频...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None)
        self.progress.setMinimumDuration(0)  # 立即显示对话框
        self.progress.show()
        
        if self.video_list[self.cur_video_idx-1] in self.ori_video:
            video = self.video_cache[self.video_list[self.cur_video_idx-1]]
            res = self.load_video(video)
            if res == -1:
                self.progress.close()
                QMessageBox.warning(self, "Error", "视频加载失败，请检查网络设置")
            else:
                self.progress.close()
                QMessageBox.information(self, "Success", "视频加载完成!")
        else:
            self.video_thread = self.request_video_async()
            self.video_thread.finished.connect(self.load_video_callback)
            self.video_thread.start()
    
    def load_video(self, video):
        if video is None:
            return -1
        
        self.sam_config['video_path'] = self.video_list[self.cur_video_idx-1]
        self.co_tracker_config['video_path'] = self.video_list[self.cur_video_idx-1]
        self.frame_count = video.shape[0]
        self.sam_object_id = [0] * self.frame_count
        has_interpolation = False
        
        if self.video_list[self.cur_video_idx-1] not in self.lang_anno:
            self.lang_anno[self.video_list[self.cur_video_idx-1]] = dict()
            
        if self.video_list[self.cur_video_idx-1] not in self.tracking_points_sam:
            self.tracking_points_sam[self.video_list[self.cur_video_idx-1]] = dict()
        
        if self.video_list[self.cur_video_idx-1] not in self.tracking_points_tap:
            self.tracking_points_tap[self.video_list[self.cur_video_idx-1]] = dict()
        
        if self.video_list[self.cur_video_idx-1] not in self.tracking_points_interp:
            self.tracking_points_interp[self.video_list[self.cur_video_idx-1]] = dict()
        
        for i in range(self.frame_count):
            self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][i] = [dict(
                    pos=[], raw_pos=[], neg=[], raw_neg=[], labels=[]
            )]
            self.tracking_points_tap[self.video_list[self.cur_video_idx-1]][i] = dict(
                pos=[], raw_pos=[]
            )
            if i not in self.tracking_points_interp[self.video_list[self.cur_video_idx-1]]:
                self.tracking_points_interp[self.video_list[self.cur_video_idx-1]][i] = dict(
                    pos=[], raw_pos=[]
                )
        
        self.vis_ori.setChecked(True)
        self.vis_track_res = False 
        self.ori_video[self.video_list[self.cur_video_idx-1]] = np.array(video)
        if self.video_list[self.cur_video_idx-1] not in self.sam_res:
            self.sam_res[self.video_list[self.cur_video_idx-1]] = np.array(video)
        if self.video_list[self.cur_video_idx-1] not in self.track_res:
            self.track_res[self.video_list[self.cur_video_idx-1]] = np.array(video)
        
        self.progress_slider.setMaximum(self.frame_count - 1)
        self.progress_slider.show()
        self.frame_position_label.show()
        self.update_keyframe_bar()
        self.update_frame(0)
        self.progress_slider.setValue(0)
        self.anno_function_select.setCurrentIndex(0)
        self.keyframe_bar.show()
        self.video_position_label.setText(f"Frame: {self.cur_frame_idx+1}/{self.frame_count} | Video: {self.cur_video_idx}/{len(self.video_list)}")
        self.pre_f_button.setDisabled(True)
        self.sam_obj_pos_label.setText("Annotation Object: 1/1")
        if self.video_list[self.cur_video_idx-1] not in self.anno:
            self.anno[self.video_list[self.cur_video_idx-1]] = dict(
                sam=None, track=None, visibility=None
            )
        if self.video_list[self.cur_video_idx-1] not in self.max_point_num:
            self.max_point_num[self.video_list[self.cur_video_idx-1]] = 0
        self.seek_video()
        # for align the keyframe display length
        if 0 not in self.keyframes[self.video_list[self.cur_video_idx-1]]:
            self.mark_keyframe()
            self.selected_keyframe = self.progress_slider.value()
            self.remove_keyframe()

        return 1
            
    def update_frame(self, frame_number):
        if self.video_list[self.cur_video_idx-1] not in self.ori_video or len(self.ori_video[self.video_list[self.cur_video_idx-1]]) == 0:
            self.smart_message('Please load video first')
            return
        frame = self.ori_video[self.video_list[self.cur_video_idx-1]][frame_number]
        if self.vis_track_res:
            frame = self.parse_res[frame_number]
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
        self.video_position_label.setText(f"Frame: {self.cur_frame_idx+1}/{self.frame_count} | Video: {self.cur_video_idx}/{len(self.video_list)}")

        self.pre_f_button.setDisabled(False)
        self.next_f_button.setDisabled(False)
        
        if self.cur_frame_idx == 0:
            self.pre_f_button.setDisabled(True)
        if self.cur_frame_idx == self.frame_count-1:
            self.next_f_button.setDisabled(True)
        
        self.sam_object_id[self.cur_frame_idx] = 0
        self.sam_obj_pos_label.setText(f"Annotation Object: {self.sam_object_id[self.cur_frame_idx]+1}/{len(self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.cur_frame_idx])}")
        self.sam_pre_button.setDisabled(True)
        self.sam_next_button.setDisabled(False)
        
        if (0, 0) in self.lang_anno[self.video_list[self.cur_video_idx-1]]:
            self.video_lang_input.setText(self.lang_anno[self.video_list[self.cur_video_idx-1]][(0, 0)])
        else:
            self.video_lang_input.setText('')
        
        anno_loc, (clip_text, prim) = self.get_clip_description()
        if anno_loc is not None:
            self.clip_lang_input.setText(f"Start Frame: {anno_loc[0]+1} | End Frame: {anno_loc[1]+1}\nPrim: {prim}\nDescription: {clip_text}")
        else:
            self.clip_lang_input.clear()
            
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
        if self.video_list[self.cur_video_idx-1] in self.keyframes and frame_number in self.keyframes[self.video_list[self.cur_video_idx-1]]:
            keyframe_type = self.keyframes[self.video_list[self.cur_video_idx-1]][frame_number]
            self.frame_position_label.setText(f"Frame: {frame_number+1} {keyframe_type}")
        else:
            self.frame_position_label.setText(f"Frame: {frame_number+1}")

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
        
        tracking_points = self.tracking_points_sam[self.video_list[self.cur_video_idx-1]]
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
            self.progress.close()
            QMessageBox.information(self, "Success", "SAM处理完成!")
            self.remove_frame_annotation()
        else:
            self.progress.close()
            QMessageBox.warning(self, "Error", "SAM处理失败，请重试")
            return
    
    def get_sam_async(self):
        if self.video_list[self.cur_video_idx-1] not in self.ori_video:
            self.smart_message("Please load video first!")
            return
        
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
        if masks is None:
            return -1
        
        frame_id = self.sam_config['select_frame']

        mask_images = np.array([cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB) for mask_image in mask_images])  
        self.sam_res[self.video_list[self.cur_video_idx-1]][frame_id:frame_id+mask_images.shape[0]] = mask_images
        
        if self.anno[self.video_list[self.cur_video_idx-1]]['sam'] is None:
            self.anno[self.video_list[self.cur_video_idx-1]]['sam'] = np.zeros((masks.shape[0], self.frame_count, *masks[0,0].shape))
        self.anno[self.video_list[self.cur_video_idx-1]]['sam'][:, frame_id:frame_id+mask_images.shape[0]] = masks
        return 1
    
    def tracker_callback(self, res):
        self.tracker_thread.wait()
        if res == 1:
            self.vis_tracker.setChecked(True)
            self.load_res()
            self.progress.close()
            QMessageBox.information(self, "Success", "Tracker处理完成!")
            self.remove_frame_annotation()
        else:
            self.progress.close()
            QMessageBox.warning(self, "Error", "Tracker处理失败，请重试")
            return
            
    def get_tap_async(self):
        if self.video_list[self.cur_video_idx-1] not in self.ori_video:
            self.smart_message("Please load video first!")
            return
        
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
        track_mode = self.track_mode_selector.currentText()
        self.co_tracker_config['mode'] = self.button_param_select.currentText()
        self.co_tracker_config['track_mode'] = track_mode
        
        if self.co_tracker_config['mode'] == 'Point Mode':
            points, select_frame = [], []
            if track_mode == 'BiDirection':
                for frame_id, frame_pts in self.tracking_points_tap[self.video_list[self.cur_video_idx-1]].items():
                    if frame_pts['raw_pos'] != []:
                        points.extend([[pt.x(), pt.y()] for pt in frame_pts['raw_pos']])
                        select_frame.extend([[frame_id]] * len(frame_pts['raw_pos']))
            else:
                frame_pts = self.tracking_points_tap[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]
                if frame_pts['raw_pos'] != []:
                    points.extend([[pt.x(), pt.y()] for pt in frame_pts['raw_pos']])
                    select_frame.extend([[self.progress_slider.value()]] * len(frame_pts['raw_pos']))
            
            self.co_tracker_config['points'] = points
            self.co_tracker_config['select_frame'] = select_frame
            assert len(select_frame) == len(points)
        
        elif self.co_tracker_config['mode'] == 'Mask Mode':
            self.smart_message("开发中，请优先使用point模式")
            return
            # self.set_sam_config()
            # self.co_tracker_config['select_frame'] = self.sam_config['select_frame']  
        
        elif self.co_tracker_config['mode'] == 'Grid Mode':
            self.smart_message("开发中，请优先使用point模式")
            return
            # self.co_tracker_config['grid_size'] = 10
        
        else:
            raise ValueError('Please select the tracker mode')
    
    def get_tap_result(self, sam_config, co_tracker_config):        
        pred_tracks, pred_visibility, images = request_cotracker(sam_config, co_tracker_config)
        if pred_tracks is None:
            return -1
        frame_id = co_tracker_config['select_frame'][0][0]
        track_images = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images])

        if self.anno[self.video_list[self.cur_video_idx-1]]['track'] is None:
            self.anno[self.video_list[self.cur_video_idx-1]]['track'] = np.zeros((self.frame_count, *pred_tracks[0,0].shape))
            self.anno[self.video_list[self.cur_video_idx-1]]['visibility'] = np.zeros((self.frame_count, *pred_visibility[0,0].shape))
        
        if self.track_mode_selector.currentText() == 'BiDirection':
            self.track_res[self.video_list[self.cur_video_idx-1]] = track_images
            self.anno[self.video_list[self.cur_video_idx-1]]['track'][:] = pred_tracks[0]
            self.anno[self.video_list[self.cur_video_idx-1]]['visibility'][:] = pred_visibility[0]
        else:
            self.track_res[self.video_list[self.cur_video_idx-1]][frame_id:frame_id+track_images.shape[0]] = track_images
            self.anno[self.video_list[self.cur_video_idx-1]]['track'][frame_id:frame_id+track_images.shape[0]] = pred_tracks[0]
            self.anno[self.video_list[self.cur_video_idx-1]]['visibility'][frame_id:frame_id+track_images.shape[0]] = pred_visibility[0]
        
        return 1
    
    def interp(self, p1, p2, f1, f2):
        if f1 == f2:
            return [(p1, p2)]
        p = np.linspace(p1, p2, f2 - f1 + 1)
        return p[1:]
    
    def get_interp_points(self, points, select_frame):
        interp_points = []
        for key_frame_num in range(len(points)):
            if key_frame_num == 0:
                interp_points.append(points[key_frame_num])
            else:
                interp_points.extend(self.interp(points[key_frame_num-1], points[key_frame_num], select_frame[key_frame_num-1], select_frame[key_frame_num]))
        return interp_points
    
    def get_interp_result(self):
        if self.video_list[self.cur_video_idx-1] not in self.tracking_points_interp:
            self.smart_message("Please load video first!")
            return

        raw_points, points, select_frame, point_num = [], [], [], []
        for frame_id, frame_pts in self.tracking_points_interp[self.video_list[self.cur_video_idx-1]].items():
            if frame_pts['raw_pos'] != []:
                raw_points.append([[pt.x(), pt.y()] for pt in frame_pts['raw_pos']])
                points.append([[pt.x(), pt.y()] for pt in frame_pts['pos']])
                select_frame.append(frame_id)
                point_num.append(len(frame_pts['raw_pos']))
        
        if len(points) < 2:
            self.smart_message("Please select at least one point in at least two frame")
            return -1
        
        # check if the points number is the same in each frame
        wrong_frame_number_idx = [select_frame[i] for i, num in enumerate(point_num) if num != point_num[0]]
        if len(wrong_frame_number_idx) > 0:
            self.smart_message("Please select the same number of points as the first frame in each frame with the same order, the wrong frame number is: " + str(wrong_frame_number_idx))
            return -1
        
        if self.anno[self.video_list[self.cur_video_idx-1]]['track'] is not None and self.anno[self.video_list[self.cur_video_idx-1]]['track'].shape[1] != self.max_point_num[self.video_list[self.cur_video_idx-1]]:
            self.smart_message("Please select the same number of points as the tracker mode since the tracker mode has been used")
            return -1 
        
        interp_raw_points = self.get_interp_points(raw_points, select_frame)
        interp_points = self.get_interp_points(points, select_frame)
        
        
        start_frame = select_frame[0]
        if self.anno[self.video_list[self.cur_video_idx-1]]['track'] is None:
            self.anno[self.video_list[self.cur_video_idx-1]]['track'] = np.zeros((self.frame_count, len(interp_points[0]), 2))
            self.anno[self.video_list[self.cur_video_idx-1]]['visibility'] = np.zeros((self.frame_count, len(interp_points[0])))
        
        self.anno[self.video_list[self.cur_video_idx-1]]['track'][start_frame:start_frame+len(interp_points)] = np.array(interp_raw_points)
        self.anno[self.video_list[self.cur_video_idx-1]]['visibility'][start_frame:start_frame+len(interp_points)] = np.ones((len(interp_points), len(interp_points[0])))
        
        # for visualization
        for frame_id, pts in enumerate(interp_points):
            self.tracking_points_interp[self.video_list[self.cur_video_idx-1]][start_frame+frame_id]['pos'] = []
            for point_id in range(len(pts)):
                self.tracking_points_interp[self.video_list[self.cur_video_idx-1]][start_frame+frame_id]['pos'].append(QPoint(int(pts[point_id][0]), int(pts[point_id][1])))
        
        self.draw_image()
        self.smart_message("Interpolation完成!")
        self.progress_slider.setValue(start_frame)
        
        return 1
   
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
            if self.anno_mode == 'sam':
                sam_object_id = self.sam_object_id[self.progress_slider.value()]
                self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['raw_pos'].append(original_position)
                self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['pos'].append(click_position)
                self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['labels'].append(1)
                self.sam_next_button.setDisabled(False)
            elif self.anno_mode == 'tracker':
                self.tracking_points_tap[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]['raw_pos'].append(original_position)
                self.tracking_points_tap[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]['pos'].append(click_position)
            elif self.anno_mode == 'interpolation':
                self.tracking_points_interp[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]['raw_pos'].append(original_position)
                self.tracking_points_interp[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]['pos'].append(click_position)
                self.max_point_num[self.video_list[self.cur_video_idx-1]] = max(self.max_point_num[self.video_list[self.cur_video_idx-1]], len(self.tracking_points_interp[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]['pos']))
                self.track_point_num_label.setText(f"Point Number: {self.max_point_num[self.video_list[self.cur_video_idx-1]]}")
                
            
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
                self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['neg'].append(click_position)
                self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['raw_neg'].append(original_position)
                self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['labels'].append(-1)
                self.sam_next_button.setDisabled(False)
            
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

        if self.anno_mode == 'sam':
            sam_object_id = self.sam_object_id[self.progress_slider.value()]
            pos_click_position = self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['pos']
            neg_click_position = self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['neg']
        elif self.anno_mode == 'tracker':
            pos_click_position = self.tracking_points_tap[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]['pos']
            neg_click_position = []
        elif self.anno_mode == 'interpolation':
            pos_click_position = self.tracking_points_interp[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]['pos']
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
        if self.video_list[self.cur_video_idx-1] not in self.ori_video:
            self.smart_message('请先加载视频')
            return
        if self.description_mode.currentText() == 'Frame':
            self.add_frame_discribtion()
        elif self.description_mode.currentText() == 'Video':
            self.add_video_description()

    def mark_keyframe(self):
        current_frame = self.progress_slider.value()
        if self.key_frame_mode == 'Start':
            self.keyframes[self.video_list[self.cur_video_idx-1]][current_frame] = 'start'
        elif self.key_frame_mode == 'End':
            self.keyframes[self.video_list[self.cur_video_idx-1]][current_frame] = 'end'
        self.update_keyframe_bar()
    
    def update_lang_anno(self):
        key_frame_list = sorted(self.keyframes[self.video_list[self.cur_video_idx-1]].keys())
        key_pairs = []
        if len(key_frame_list) <= 1:
            self.smart_message('请先标记关键帧')
            return -1
        
        for i in range(0, len(key_frame_list), 2):
            start_frame = key_frame_list[i]
            end_frame = key_frame_list[i+1]
            if self.keyframes[self.video_list[self.cur_video_idx-1]][start_frame] != 'start' or self.keyframes[self.video_list[self.cur_video_idx-1]][end_frame] != 'end':
                self.smart_message('请检查关键帧标记是否正确，必须是start和end交替出现')
                return -1
            key_pairs.append((start_frame, end_frame))
            
        for i in key_pairs:
            if i not in self.lang_anno[self.video_list[self.cur_video_idx-1]]:
                self.lang_anno[self.video_list[self.cur_video_idx-1]][i] = (None, None)

    def remove_keyframe(self):
        # Remove the selected keyframe if any
        if self.selected_keyframe is not None:
            frame_to_remove = self.selected_keyframe
            if frame_to_remove in self.keyframes[self.video_list[self.cur_video_idx-1]]:
                del self.keyframes[self.video_list[self.cur_video_idx-1]][frame_to_remove]
                self.update_keyframe_bar()
                self.selected_keyframe = None

    def update_keyframe_bar(self):
        if self.video_list[self.cur_video_idx-1] not in self.keyframes:
            self.keyframes[self.video_list[self.cur_video_idx-1]] = {}
        # Clear the keyframe bar
        keyframe_image = QImage(self.keyframe_bar.width(), self.keyframe_bar.height(), QImage.Format_RGB32)
        keyframe_image.fill(Qt.gray)

        painter = QPainter(keyframe_image)
        for frame, key_type in self.keyframes[self.video_list[self.cur_video_idx-1]].items():
            x_position = int((frame / self.frame_count) * self.keyframe_bar.width())
            color = QColor('red') if key_type == 'start' else QColor('blue')
            painter.fillRect(QRect(x_position, 0, 5, self.keyframe_bar.height()), color)
        painter.end()

        # Set the updated image to the QLabel
        self.keyframe_bar.setPixmap(QPixmap.fromImage(keyframe_image))
    
    def show_tooltip(self, event):
        # Show tooltip when hovering over a keyframe
        mouse_pos = event.pos()
        for frame, key_type in self.keyframes[self.video_list[self.cur_video_idx-1]].items():
            x_position = int((frame / self.frame_count) * self.keyframe_bar.width())
            if abs(mouse_pos.x() - x_position) <= 5:  # Small range to detect hover
                QToolTip.showText(self.keyframe_bar.mapToGlobal(mouse_pos), f"Frame: {frame+1}")
                return
        QToolTip.hideText()

    def closeEvent(self, event):
        # if self.cap is not None:
        #     self.cap.release()
        event.accept()
     
    def keyPressEvent(self, event):
        if self.video_list[self.cur_video_idx-1] not in self.ori_video:
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
        key_pairs = list(self.lang_anno[self.video_list[self.cur_video_idx-1]].keys())
        has_key = [i[0] <= frame_number <= i[1] for i in key_pairs].count(True) > 0
        if not has_key:
            self.smart_message('请先标记当前所在区域的起止帧')
            return

        # load the cached description
        anno_loc = [i for i in key_pairs if i[0] <= frame_number <= i[1] and i[0] != i[1]]
        if len(anno_loc) == 0:
            self.smart_message('请移动到所在区域的起止帧之间')
            return
        anno_loc = anno_loc[0]
        
        if self.lang_anno[self.video_list[self.cur_video_idx-1]][anno_loc] is not None:
            cached_lang, prim = self.lang_anno[self.video_list[self.cur_video_idx-1]][anno_loc]
        else:
            cached_lang, prim = '', ''
        # Create a dialog to get the description from the user
        dialog = TextInputDialog(cached_lang, self, False)
        if dialog.exec_() == QDialog.Accepted:
            cached_lang = dialog.get_text()
            prim = dialog.get_prim()
            self.lang_anno[self.video_list[self.cur_video_idx-1]][anno_loc] = (cached_lang, prim)
            self.clip_lang_input.setText(f"Start Frame: {anno_loc[0]+1} | End Frame: {anno_loc[1]+1}\nPrim: {prim}\nDescription: {cached_lang}")
        else:
            return 
        
    def add_video_description(self):
        # Create a dialog to get the description from the user
        if (0, 0) in self.lang_anno[self.video_list[self.cur_video_idx-1]]:
            cached_lang = self.lang_anno[self.video_list[self.cur_video_idx-1]][(0, 0)]
        else:
            cached_lang = ''
        
        dialog = TextInputDialog(cached_lang, self)
        if dialog.exec_() == QDialog.Accepted:
            video_description = dialog.get_text()
            self.lang_anno[self.video_list[self.cur_video_idx-1]][(0, 0)] = video_description
            self.video_lang_input.setText(f"Video Description: {video_description}")
        else:
            return

    def get_clip_description(self):
        # Get the description for the clip
        key_pairs = list(self.lang_anno[self.video_list[self.cur_video_idx-1]].keys())
        frame_number = self.progress_slider.value()
        anno_loc = [i for i in key_pairs if i[0] <= frame_number <= i[1] and i[0] != i[1]]
        if len(anno_loc) > 0:
            return anno_loc[0], self.lang_anno[self.video_list[self.cur_video_idx-1]][anno_loc[0]]
        return None, (None, None)

if __name__ == "__main__":
    
    # load annotation file
    args = argparse.ArgumentParser()
    args.add_argument('--anno_file', type=str, default='./data/video_list.txt')
    args.add_argument('--out_file', type=str, default='./data/annotation.json')
    args = args.parse_args()
    
    app = QApplication(sys.argv)
    player = VideoPlayer(args)
    player.resize(1150, 600)  # Adjusted size to accommodate the toolbar
    player.show()
    sys.exit(app.exec_())