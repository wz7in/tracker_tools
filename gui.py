import sys
import os
import json
import argparse
import pickle
from PyQt5.QtCore import QPoint, QTimer, Qt
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox, QLineEdit, QDialogButtonBox, QTextEdit, QGridLayout,
                             QLabel, QSlider, QDialog, QHBoxLayout, QFrame, QProgressDialog, QRadioButton, QToolTip, QComboBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QMouseEvent, QBrush
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal, QThread, QThreadPool, pyqtSlot, QRunnable, QObject

import yaml
from client_utils import request_sam, request_cotracker, request_video
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT_DIR = '/mnt/hwfile/OpenRobotLab/wangziqin/data/rh20t/'

def load_anno_file(anno_file, out_file):
    if anno_file.endswith('.json'):
        video_anno = json.load(open(anno_file, 'r'))
    elif anno_file.endswith('.npy'):
        video_anno = np.load(anno_file, allow_pickle=True).item()
    if os.path.exists(out_file):
        anno = pickle.load(open(out_file, 'rb'))
    else:
        anno = {}
    return sorted(list(video_anno.keys()))[:12], video_anno, anno

class TextInputDialog(QDialog):
    
    def __init__(self, initial_text='', parent=None, is_video=True, video_anno_json=None, ann_id=None):
        super().__init__(parent)
        self.setWindowTitle('请输入语言标注')
        self.is_video = is_video
        
        self.main_layout = QGridLayout(self)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        
        if not is_video:
            global_instruction_C = video_anno_json['instructionC']
            clip_lang_C_options = video_anno_json['clip_langC']
            primitive_action_options_C = [i[1] for i in video_anno_json['primitive_action']]
            
            self.prim_title = QLabel('请选择语言标注:', self)
            self.prim_select = QComboBox()
            self.prim_select.addItems([i for i in clip_lang_C_options.keys() if clip_lang_C_options[i] is None])
            self.prim_select.addItems(['空'])
            
            initial_action = None
            for i in clip_lang_C_options.keys():
                if clip_lang_C_options[i] is not None and clip_lang_C_options[i] == ann_id:
                    initial_action = i
                    break
            if initial_action and len(initial_action) > 0:
                self.prim_select.addItems([initial_action])
                self.prim_select.setCurrentText(initial_action)
            else:
                self.prim_select.setCurrentIndex(-1)
            
            self.mode_title = QLabel('请选择原子动作:', self)
            self.mode_select = QComboBox()
            self.mode_select.addItems(primitive_action_options_C)
            self.mode_select.setCurrentIndex(0)
            self.language_edit = QTextEdit()
            self.language_title = QLabel('请确认语言标注:', self)
            self.language_title.hide()
            if initial_text is not None and len(initial_text) > 0:
                self.language_edit.setText(initial_text)
            else:
                self.language_edit.hide()
                self.language_edit.setText('')
            
            self.language_edit.setFixedHeight(100)
            self.prim_select.currentIndexChanged.connect(self.language_select)
            self.main_layout.addWidget(self.language_edit, 2, 1)
            self.main_layout.addWidget(self.language_title, 2, 0)
            self.main_layout.addWidget(self.mode_title, 1, 0)
            self.main_layout.addWidget(self.mode_select, 1, 1)
            self.main_layout.addWidget(self.prim_title, 0, 0)
            self.main_layout.addWidget(self.prim_select, 0, 1)
            self.main_layout.addWidget(self.button_box, 3, 0, 1, 2)
        else:
            global_instruction_C = video_anno_json['instructionC']
            self.text_title = QLabel('请输入语言标注:', self)
            self.text_input = QLineEdit(self)
            self.text_input.setText(global_instruction_C if initial_text is None or len(initial_text) == 0 else initial_text)
            self.text_input.setFixedSize(300,20)
            self.text_input.setText(initial_text)
            self.main_layout.addWidget(self.text_title, 0, 0)
            self.main_layout.addWidget(self.text_input, 0, 1)
            self.main_layout.addWidget(self.button_box, 1, 0, 1, 2)
                
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
    def get_text(self):
        return self.language_edit.toPlainText() if not self.is_video else self.text_input.text()
    
    def get_prim(self):
        if not self.is_video:
            return self.mode_select.currentText()
        else:
            return ''
    
    def get_select_lang(self):
        if not self.is_video:
            return self.prim_select.currentText()
        else:
            return ''
    
    def language_select(self):
        if not self.is_video:
            self.language_title.show()
            self.language_edit.show()
            self.language_edit.setText(self.prim_select.currentText())

class DictQueue():
    def __init__(self):
        self.max_size = 10
        self.queue = []
        self.memory = {}
    
    def pop(self):
        key = self.queue.pop(0)
        value = self.memory[key]
        del self.memory[key]
        return key, value
    
    def get(self, key):
        if key not in self.memory:
            return None
        return self.memory[key]
    
    def get_locate(self, key):
        return self.queue.index(key)
    
    def get_last(self):
        return self.queue[-1]
    
    def __len__(self):
        return len(self.queue)
    
    def add(self, key, value):
        self.memory[key] = value
        self.queue.append(key)
        if len(self.queue) > self.max_size:
            self.pop()
    
    def clear(self):
        self.queue = []
        self.memory = {}
    

class VideoPlayer(QWidget):
    def __init__(self, args):
        self.video_list, self.video_anno_list, self.all_anno = load_anno_file(args.anno_file, args.out_file)
        super().__init__()
        self.setWindowTitle("视频标注工具")
        ###########################################################
        #################### Main Area Layout ####################
        ###########################################################
        main_layout = QHBoxLayout()     
        
        ###########################################################
        #################### Video Area Layout ####################
        ###########################################################
        video_layout = QVBoxLayout()
        # Video display label
        self.video_label = QLabel(self)
        self.video_label.setMouseTracking(True)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)
        # Progress slider
        self.progress_slider = QSlider(self)
        self.progress_slider.setOrientation(Qt.Horizontal)  
        self.progress_slider.valueChanged.connect(self.seek_video)
        self.progress_slider.hide()  # Hide initially
        video_layout.addWidget(self.progress_slider)
        # Keyframe indicator bar
        self.keyframe_bar = QLabel(self)
        self.keyframe_bar.setFixedHeight(10) 
        self.keyframe_bar.setMouseTracking(True)
        self.keyframe_bar.setAlignment(Qt.AlignCenter)  
        self.keyframe_bar.installEventFilter(self)  
        video_layout.addWidget(self.keyframe_bar)
        # Dynamic frame position label that floats above the slider
        self.frame_position_label = QLabel(self)
        self.frame_position_label.setStyleSheet("background-color: #E3E3E3;")
        self.frame_position_label.setAlignment(Qt.AlignCenter)
        self.frame_position_label.setFixedSize(80, 20)
        self.frame_position_label.hide()
        video_control_button_layout = QHBoxLayout()
        # Pre video button
        self.pre_button = QPushButton("<<", self)
        self.pre_button.clicked.connect(self.pre_video)
        self.pre_button.setDisabled(True)
        video_control_button_layout.addWidget(self.pre_button)
        # Video position label
        self.video_position_label = QLabel(self)
        self.video_position_label.setStyleSheet("background-color: #E3E3E3;")
        self.video_position_label.setAlignment(Qt.AlignCenter)
        self.video_position_label.setFixedSize(350, 20)
        video_control_button_layout.addWidget(self.video_position_label)
        # Next video button
        self.next_button = QPushButton(">>", self)
        self.next_button.clicked.connect(self.next_video)
        video_control_button_layout.addWidget(self.next_button)
        video_layout.addLayout(video_control_button_layout)
        video_load_button_layout = QHBoxLayout()
        # Load video button
        self.load_button = QPushButton("加载视频", self)
        self.load_button.clicked.connect(self.load_video_async)
        video_load_button_layout.addWidget(self.load_button)
        # Play button
        self.play_button = QPushButton("播放", self)
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_playback)
        video_load_button_layout.addWidget(self.play_button)
        # Remove video button
        self.remove_video_button = QPushButton("移除视频", self)
        self.remove_video_button.clicked.connect(self.clear_video)
        video_load_button_layout.addWidget(self.remove_video_button)
        video_layout.addLayout(video_load_button_layout)
        main_layout.addLayout(video_layout)
        ###########################################################
        ######################## Separator ########################
        ###########################################################
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)
        ###########################################################
        #################### Toolbar Area Layout ##################
        ###########################################################
        self.toolbar_layout = QVBoxLayout()
        
        # Auto-annotation function title layout
        function_title_layout = QHBoxLayout()
        function_title = QLabel("分割标注工具区", self)
        function_title.setAlignment(Qt.AlignLeft)  # Left align the title
        function_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        function_title_layout.addWidget(function_title)
        fline = QFrame(self)
        fline.setFrameShape(QFrame.HLine)
        fline.setFrameShadow(QFrame.Sunken)
        fline.setStyleSheet("color: grey;")  # Set the same color as the title
        function_title_layout.addWidget(fline)
        self.toolbar_layout.addLayout(function_title_layout)
        # Auto-annotation function selection layout
        anno_button_layout = QHBoxLayout()
        self.anno_function_select = QComboBox()
        self.anno_function_select.addItem('分割模型')
        self.anno_function_select.addItem('跟踪模型')
        self.anno_function_select.addItem('插值模型')
        self.anno_function_select.hide()
        # mode select button
        self.button_param_select = QComboBox()
        self.button_param_select.addItem('双向视频模式')
        self.button_param_select.addItem('前向视频模式')
        self.button_param_select.addItem('反向视频模式')
        self.button_param_select.addItem('单帧模式')
        # tracking point numbers displayer in interpolation mode
        self.track_point_num_label = QLabel(self)
        self.track_point_num_label.setStyleSheet("background-color: #E3E3E3;")
        self.track_point_num_label.setAlignment(Qt.AlignCenter)
        self.track_point_num_label.setFixedSize(150, 20)
        self.track_point_num_label.hide()
        # track mode selector
        self.track_mode_selector = QComboBox()
        self.track_mode_selector.addItems(['双向', '前向'])
        self.track_mode_selector.hide()
        self.anno_function_select.currentIndexChanged.connect(self.update_function_select)
        # run button
        # click_action_button = QPushButton("运行", self)
        click_action_button = QPushButton("保存SAM标注", self)
        # click_action_button.clicked.connect(self.get_anno_result)
        click_action_button.clicked.connect(self.save_sam_input)
        anno_button_layout.addWidget(self.anno_function_select)
        anno_button_layout.addWidget(self.button_param_select)
        anno_button_layout.addWidget(self.track_mode_selector)
        anno_button_layout.addWidget(self.track_point_num_label)
        anno_button_layout.addWidget(click_action_button)
        self.toolbar_layout.addLayout(anno_button_layout)
        self.sam_object_layout = QHBoxLayout()
        # sam pre object button
        self.sam_pre_button = QPushButton("上一个物体", self)
        self.sam_pre_button.clicked.connect(self.pre_sam_object)
        self.sam_pre_button.setDisabled(True)
        # sam object position label
        self.sam_obj_pos_label = QLabel(self)
        self.sam_obj_pos_label.setStyleSheet("background-color: #E3E3E3;")
        self.sam_obj_pos_label.setAlignment(Qt.AlignCenter)
        self.sam_obj_pos_label.setFixedSize(150, 20)
        # sam next object button
        self.sam_next_button = QPushButton("下一个/添加物体", self)
        self.sam_next_button.clicked.connect(self.next_sam_object)
        self.sam_next_button.setDisabled(True)
        self.sam_object_layout.addWidget(self.sam_pre_button)
        self.sam_object_layout.addWidget(self.sam_obj_pos_label)
        self.sam_object_layout.addWidget(self.sam_next_button)
        self.toolbar_layout.addLayout(self.sam_object_layout)
        
        # Visualization area layout
        annotation_title_layout = QHBoxLayout()
        annotation_vis_title = QLabel("标注及结果可视化区域", self)
        annotation_vis_title.setAlignment(Qt.AlignLeft)  # Left align the title
        annotation_vis_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        annotation_title_layout.addWidget(annotation_vis_title)
        visline = QFrame(self)
        visline.setFrameShape(QFrame.HLine)
        visline.setFrameShadow(QFrame.Sunken)
        visline.setStyleSheet("color: grey;")  # Set the same color as the title
        annotation_title_layout.addWidget(visline)
        self.toolbar_layout.addLayout(annotation_title_layout)
        # Visualization button layout
        vis_button_layout = QHBoxLayout()
        self.vis_button = QPushButton("可视化", self)
        self.vis_button.clicked.connect(self.load_res)
        vis_button_layout.addWidget(self.vis_button)
        self.vis_ori = QRadioButton("原始视频", self)
        vis_button_layout.addWidget(self.vis_ori)
        self.vis_sam = QRadioButton("分割结果视频", self)
        vis_button_layout.addWidget(self.vis_sam)
        self.vis_tracker = QRadioButton("跟踪结果视频", self)
        vis_button_layout.addWidget(self.vis_tracker)
        self.toolbar_layout.addLayout(vis_button_layout)
        self.vis_tracker.hide()

        # edit mode are layout
        annotation_title_layout = QHBoxLayout()
        annotation_title = QLabel("标注编辑区域", self)
        annotation_title.setAlignment(Qt.AlignLeft)  # Left align the title
        annotation_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        annotation_title_layout.addWidget(annotation_title)
        annoline = QFrame(self)
        annoline.setFrameShape(QFrame.HLine)
        annoline.setFrameShadow(QFrame.Sunken)
        annoline.setStyleSheet("color: grey;")  # Set the same color as the title
        annotation_title_layout.addWidget(annoline)
        self.toolbar_layout.addLayout(annotation_title_layout)
        # clear all button
        self.control_button_layout = QHBoxLayout()
        self.clear_all_button = QPushButton("删除所有标注", self)
        self.clear_all_button.clicked.connect(self.clear_annotations)
        self.control_button_layout.addWidget(self.clear_all_button)
        # remove last button
        self.remove_last_button = QPushButton("删除上一个标注", self)
        self.remove_last_button.clicked.connect(self.remove_last_annotation)
        self.control_button_layout.addWidget(self.remove_last_button)
        # remove frame button
        self.remove_frame_button = QPushButton("删除当前帧标注", self)
        self.remove_frame_button.clicked.connect(self.remove_frame_annotation)
        self.control_button_layout.addWidget(self.remove_frame_button)
        # save button
        self.save_button = QPushButton("保存语言标注", self)
        # self.save_button.clicked.connect(self.save_annotation)
        self.save_button.clicked.connect(self.save_lang_anno)
        self.control_button_layout.addWidget(self.save_button)
        self.toolbar_layout.addLayout(self.control_button_layout)

        # Video language annotation area
        lang_layout = QVBoxLayout()
        lang_title_layout = QHBoxLayout()
        lang_title = QLabel("视频语言标注展示区域", self)
        lang_title.setAlignment(Qt.AlignLeft)  # Left align the title
        lang_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        lang_title_layout.addWidget(lang_title)
        videoline = QFrame(self)
        videoline.setFrameShape(QFrame.HLine)
        videoline.setFrameShadow(QFrame.Sunken)
        videoline.setStyleSheet("color: grey;")  # Set the same color as the title
        lang_title_layout.addWidget(videoline)
        lang_layout.addLayout(lang_title_layout)
        # Video Language annotation show area
        self.video_lang_input = QTextEdit(self)
        self.video_lang_input.setReadOnly(True)
        self.video_lang_input.setFixedHeight(60)
        self.video_lang_input.setStyleSheet("background-color: #E3E3E3; font-weight: bold;")
        lang_layout.addWidget(self.video_lang_input)
        self.toolbar_layout.addLayout(lang_layout)
        
        # Video clip language annotation area 
        clip_layout = QVBoxLayout()
        lang_title_layout = QHBoxLayout()
        clip_title = QLabel("视频段语言标注展示区域", self)
        clip_title.setAlignment(Qt.AlignLeft)  # Left align the title
        clip_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        lang_title_layout.addWidget(clip_title)
        clipline = QFrame(self)
        clipline.setFrameShape(QFrame.HLine)
        clipline.setFrameShadow(QFrame.Sunken)
        clipline.setStyleSheet("color: grey;")  # Set the same color as the title
        lang_title_layout.addWidget(clipline)
        clip_layout.addLayout(lang_title_layout)
        # Video clip Language annotation show area
        self.clip_lang_input = QTextEdit(self)
        self.clip_lang_input.setReadOnly(True)
        self.clip_lang_input.setFixedHeight(80)
        self.clip_lang_input.setStyleSheet("background-color: #E3E3E3; font-weight: bold;")
        clip_layout.addWidget(self.clip_lang_input)
        self.toolbar_layout.addLayout(clip_layout)
        
        # choose mode
        mode = self.mode_choose()
        if mode is None:
            sys.exit()
        
        if mode == 'SAM标注':
            # 删除语言标注区域
            self.video_lang_input.hide()
            self.clip_lang_input.hide()
            clip_title.hide()
            lang_title.hide()
            clipline.hide()
            videoline.hide()
            # line.hide()
            self.save_button.hide()
            tips_items = ['A: 上一帧', 'D: 下一帧']
        
        elif mode == '语言标注':
            visline.hide()
            fline.hide()
            click_action_button.hide()
            self.sam_pre_button.hide()
            self.sam_next_button.hide()
            self.sam_obj_pos_label.hide()
            self.anno_function_select.hide()
            self.button_param_select.hide()
            function_title.hide()
            annotation_title.hide()
            annotation_vis_title.hide()
            annoline.hide()
            self.vis_button.hide()
            self.vis_ori.hide()
            self.vis_sam.hide()
            self.vis_tracker.hide()
            self.track_point_num_label.hide()
            self.track_mode_selector.hide()
            self.clear_all_button.hide()
            self.remove_last_button.hide()
            self.remove_frame_button.hide()
            tips_items = ['W: 标志开始帧','L: 添加段语言标注','S: 标记结束帧','删除: 删除标记帧','A: 上一帧','回车: 添加视频标注','D: 下一帧']
            
            
        self.tips_layout = QVBoxLayout()
        self.tips_title_layout = QHBoxLayout()
        tips_title = QLabel("关键帧快捷键", self)
        tips_title.setAlignment(Qt.AlignLeft)  # Left align the title
        tips_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        self.tips_title_layout.addWidget(tips_title)
        line = QFrame(self)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: grey;")  # Set the same color as the title
        self.tips_title_layout.addWidget(line)
        self.tips_layout.addLayout(self.tips_title_layout)
        self.toolbar_layout.addLayout(self.tips_layout)
        self.tips_text_layout = QGridLayout()
        
        for i, item in enumerate(tips_items):
            tips_input = QTextEdit(self)
            tips_input.setText(item)
            tips_input.setReadOnly(True)
            tips_input.setSizeAdjustPolicy(QTextEdit.AdjustToContents)
            tips_input.setFixedHeight(20)
            if i >= 4:
                self.tips_text_layout.addWidget(tips_input, 1, i-4)
            else:
                self.tips_text_layout.addWidget(tips_input, 0, i)
        self.toolbar_layout.addLayout(self.tips_text_layout)
        main_layout.addLayout(self.toolbar_layout)
        self.setLayout(main_layout)

        self.cur_video_idx = 1
        self.video_position_label.setText(f"帧: -/- | 视频: {self.cur_video_idx}/{len(self.video_list)}")
        self.sam_obj_pos_label.setText("标注物体: -/-")
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_video)

        ##############################################################
        ##################### initialize Configs #####################
        ##############################################################
        config_path = "./config/config.yaml"
        with open(config_path, "r") as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)
        self.sam_config = self.model_config["sam"]
        self.co_tracker_config = self.model_config["cotracker"]

        ##############################################################
        #################### initialize Variables ####################
        ##############################################################
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
        self.video_cache = DictQueue()
        self.max_point_num = dict()
        self.anno_mode = 'sam'
        self.cur_frame_idx = self.progress_slider.value()
        self.keyframes = {} 
        self.selected_keyframe = None 
        self.is_edit_mode = False
        self.edit_track_res = None
        self.key_frame_mode = 'start'
        self.requesting_item = None
        self.threadpool = QThreadPool()
        self.sam_point_anno = dict()
        self.lang_only_anno = dict()
        
        
    def mode_choose(self):
        # 在主窗口上直接弹出对话框，选择模式
        mode_dialog = QDialog(self)
        mode_dialog.setWindowTitle("选择模式")
        mode_dialog.setFixedSize(300, 200)
        mode_layout = QVBoxLayout()
        mode_dialog.setLayout(mode_layout)
        mode_select = QComboBox()
        mode_select.addItem('SAM标注')
        mode_select.addItem('语言标注')
        # 返回选择的模式
        mode_layout.addWidget(mode_select)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        button_box.accepted.connect(mode_dialog.accept)
        button_box.rejected.connect(mode_dialog.reject)
        mode_layout.addWidget(button_box)
        if mode_dialog.exec_() == QDialog.Accepted:
            return mode_select.currentText()
        return None
    
    def pre_sam_object(self):
        if self.sam_object_id[self.progress_slider.value()] > 0:
            self.sam_object_id[self.progress_slider.value()] -= 1
        if self.sam_object_id[self.progress_slider.value()] == 0:
            self.sam_pre_button.setDisabled(True)
        
        self.sam_next_button.setDisabled(False)
        cur_id = self.sam_object_id[self.progress_slider.value()] + 1
        all_object_size = len(self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()])
        self.sam_obj_pos_label.setText(f"标注物体: {cur_id}/{all_object_size}")
        
        self.draw_image()
    
    def resizeEvent(self, event):
        self.seek_video()
        self.clear_keyframes()
        self.update_keyframe_bar()
        
        self.setAutoFillBackground(False)
        palette = self.palette()
        # palette.setBrush(self.backgroundRole(), QBrush(QPixmap('./demo/bg.png').scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)))
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
        self.sam_obj_pos_label.setText(f"标注物体: {cur_id}/{all_object_size}")
        
        self.draw_image()
    
    def next_video(self):
        if self.cur_video_idx < len(self.video_list):
            self.cur_video_idx += 1
        if self.cur_video_idx == len(self.video_list):
            self.next_button.setDisabled(True)
        self.pre_button.setDisabled(False)
        self.video_position_label.setText(f"帧: -/- | 视频: {self.cur_video_idx}/{len(self.video_list)}")

    def pre_video(self):
        if self.cur_video_idx > 1:
            self.cur_video_idx -= 1
        if self.cur_video_idx == 1:
            self.pre_button.setDisabled(True)
        self.next_button.setDisabled(False)
        self.video_position_label.setText(f"帧: -/- | 视频: {self.cur_video_idx}/{len(self.video_list)}")

    def next_frame(self):
        if self.cur_frame_idx < self.frame_count - 1:
            self.cur_frame_idx += 1
        else:
            return
        self.video_position_label.setText(f"帧: {self.cur_frame_idx}/{self.frame_count} | 视频: {self.cur_video_idx}/{len(self.video_list)}")
        self.sam_object_id[self.cur_frame_idx] = 0
        
        if self.sam_object_id[self.cur_frame_idx] == 0:
            self.sam_pre_button.setDisabled(True)     
        
        self.update_frame(self.cur_frame_idx)
        self.progress_slider.setValue(self.cur_frame_idx)
        
        cur_id = self.sam_object_id[self.progress_slider.value()] + 1
        all_object_size = len(self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()])
        self.sam_obj_pos_label.setText(f"标注物体: {cur_id}/{all_object_size}")   
        
        if (0, 0) in self.lang_anno[self.video_list[self.cur_video_idx-1]]:
            self.video_lang_input.setText(self.lang_anno[self.video_list[self.cur_video_idx-1]][(0, 0)])
        else:
            self.video_lang_input.setText('')
        
        anno_loc, (clip_text, prim) = self.get_clip_description()
        if anno_loc is not None:
            self.clip_lang_input.setText(f"开始帧: {anno_loc[0]+1} | 结束帧: {anno_loc[1]+1}\n原子动作: {prim}\n动作描述: {clip_text}")
        else:
            self.clip_lang_input.setText('')
        
    def pre_frame(self):
        if self.cur_frame_idx >= 1:
            self.cur_frame_idx -= 1
        else:
            return
        self.video_position_label.setText(f"帧: {self.cur_frame_idx+1}/{self.frame_count} | 视频: {self.cur_video_idx}/{len(self.video_list)}")
        self.sam_object_id[self.cur_frame_idx] = 0
        
        if self.sam_object_id[self.cur_frame_idx] == 0:
            self.sam_pre_button.setDisabled(True)
        
        self.sam_next_button.setDisabled(False)
        self.update_frame(self.cur_frame_idx)
        self.progress_slider.setValue(self.cur_frame_idx)
        
        cur_id = self.sam_object_id[self.progress_slider.value()] + 1
        all_object_size = len(self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()])
        self.sam_obj_pos_label.setText(f"物体标注: {cur_id}/{all_object_size}")
        
        if (0, 0) in self.lang_anno[self.video_list[self.cur_video_idx-1]]:
            self.video_lang_input.setText(self.lang_anno[self.video_list[self.cur_video_idx-1]][(0, 0)])
        else:
            self.video_lang_input.setText('')
        
        anno_loc, (clip_text, prim) = self.get_clip_description()
        if anno_loc is not None:
            self.clip_lang_input.setText(f"开始帧: {anno_loc[0]+1} | 结束帧: {anno_loc[1]+1}\n原子动作: {prim}\n动作描述: {clip_text}")
        else:
            self.clip_lang_input.setText('') 
    
    def request_video(self):
        if self.video_cache.get(self.video_list[self.cur_video_idx-1]) is not None:
            video = self.video_cache.get(self.video_list[self.cur_video_idx-1])       
        else:
            try:
                self.requesting_item = self.video_list[self.cur_video_idx-1]
                video = request_video(os.path.join(ROOT_DIR, self.video_list[self.cur_video_idx-1]))
                self.requesting_item = None
            except Exception as e:
                self.requesting_item = None
                return None
            if video is None:
                return None
            self.video_cache.add(self.video_list[self.cur_video_idx-1], video)
        return video
    
    def request_video_by_name(self, name):
        if self.video_cache.get(name) is None:
            self.requesting_item = name
            print(f"Loading video {name}")
            video = request_video(os.path.join(ROOT_DIR, name))
            self.requesting_item = None
        return video, name
    
    def request_video_quiet(self, name):
        video_thread = self.request_video_async_quiet(name)
        # 传入参数
        video_thread.signals.finished.connect(self.load_video_quiet_callback)
        self.threadpool.start(video_thread)
    
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
    
    @pyqtSlot(object)
    def request_video_async_quiet(self, video_name):
        class WorkerSignals(QObject):
            finished = pyqtSignal(tuple)
        class Worker(QRunnable):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent
                self.signals = WorkerSignals()
            @pyqtSlot()
            def run(self):
                result = self.parent.request_video_by_name(video_name)
                self.signals.finished.emit(result)
        
        worker = Worker(self)
        return worker
    
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
    
    def save_annotation(self):
        lang_res, sam_res, track_res = dict(), dict(), dict()
        #################### parse video language annotation ####################
        if (0, 0) in self.lang_anno[self.video_list[self.cur_video_idx-1]] and self.lang_anno[self.video_list[self.cur_video_idx-1]][(0, 0)] != '':
            lang_res['video'] = self.lang_anno[self.video_list[self.cur_video_idx-1]][(0, 0)]
        else:
            self.smart_message("请标注整体视频的语言描述")
            return 
        #################### parse clip language annotation ####################
        lang_res['clip'] = []
        for clip_range, lang in self.lang_anno[self.video_list[self.cur_video_idx-1]].items():
            if clip_range == (0, 0):
                continue
            if len(lang) == 0:
                self.smart_message(f"请完成帧{clip_range[0]}到帧{clip_range[1]}之前的语言标注")
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
        
        for i in self.video_cache:
            if self.video_cache[i] is not None:
                if i not in self.all_anno:
                    self.all_anno[i] = dict()
                self.all_anno[i]['video'] = self.video_cache[i]
        
        pickle.dump(self.all_anno, open(args.out_file, 'wb'))
        self.smart_message("保存成功!")
    
    def save_lang_anno(self):
        self.progress = QProgressDialog("请等待，正在储存标注结果...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None)
        self.progress.setMinimumDuration(0)
        self.progress.show()
        
        for cur_video_name in self.lang_anno:
        #################### parse video language annotation ####################
            lang_res = dict()
            if (0, 0) in self.lang_anno[cur_video_name] and self.lang_anno[cur_video_name][(0, 0)] != '':
                lang_res['video'] = self.lang_anno[cur_video_name][(0, 0)]
            else:
                self.smart_message("请标注整体视频的语言描述")
                return 
            #################### parse clip language annotation ####################
            lang_res['clip'] = []
            for clip_range, lang in self.lang_anno[cur_video_name].items():
                if clip_range == (0, 0):
                    continue
                if len(lang) == 0:
                    self.smart_message(f"请完成帧{clip_range[0]}到帧{clip_range[1]}之前的语言标注")
                    return
                tmp_lang = dict()
                tmp_lang['start_frame'] = clip_range[0]
                tmp_lang['end_frame'] = clip_range[1]
                tmp_lang['description'] = lang
                lang_res['clip'].append(tmp_lang)
            
            self.lang_only_anno[cur_video_name] = lang_res
        
        # if os.path.exists(args.lang_out_file):
        #     lang_res = pickle.load(open(args.lang_out_file, 'rb'))
        #     for k, v in self.lang_only_anno.items():
        #         if k not in lang_res:
        #             lang_res[k] = v
        # else:
        lang_res = self.lang_only_anno
        pickle.dump(lang_res, open(args.lang_out_file, 'wb'))
        self.progress.close()
    
    def get_anno_result(self):
        if self.anno_function_select.currentText() == '分割模型':
            self.get_sam_async()
        elif self.anno_function_select.currentText() == '跟踪模型':
            self.get_tap_async()
        elif self.anno_function_select.currentText() == '插值模型':
            self.get_interp_result()
    
    def update_function_select(self):
        
        if self.anno_function_select.currentText() == '分割模型':
            self.button_param_select.clear()
            self.button_param_select.addItem('双向视频模式')
            self.button_param_select.addItem('前向视频模式')
            self.button_param_select.addItem('反向视频模式')
            self.button_param_select.addItem('单帧模式')
            self.sam_pre_button.show()
            self.sam_next_button.show()
            self.button_param_select.show()
            self.toolbar_layout.insertLayout(2, self.sam_object_layout)
            self.track_mode_selector.hide()
            self.track_point_num_label.hide()
            self.sam_obj_pos_label.show()
            self.anno_mode = 'sam'
            self.progress_slider.setValue(0)
            
        elif self.anno_function_select.currentText() == '跟踪模型':
            self.button_param_select.clear()
            self.button_param_select.addItem('Point Mode')
            self.button_param_select.addItem('Mask Mode')
            self.button_param_select.addItem('Grid Mode')
            self.button_param_select.hide()
            self.sam_pre_button.hide()
            self.sam_next_button.hide()
            self.track_mode_selector.show()
            self.toolbar_layout.removeItem(self.sam_object_layout)
            self.sam_obj_pos_label.hide()
            self.track_point_num_label.hide()
            self.anno_mode = 'tracker'
            self.progress_slider.setValue(0)
        
        elif self.anno_function_select.currentText() == '插值模型':
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
            self.smart_message("请先加载视频!")
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
            self.smart_message("请先加载视频!")
            return
        
        if self.video_list[self.cur_video_idx-1] not in self.tracking_points_tap:
            self.smart_message("请先加载视频!")
            return

        if self.video_list[self.cur_video_idx-1] not in self.tracking_points_interp:
            self.smart_message("请先加载视频!")
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
        self.video_position_label.setText(f"帧: -/- | 视频: {self.cur_video_idx}/{len(self.video_list)}")
        # self.pre_f_button.setDisabled(True)
    
    def remove_last_sam_annotation(self):
        if self.video_list[self.cur_video_idx-1] not in self.tracking_points_sam:
            self.smart_message("请先加载视频!")
            return
        
        sam_object_id = self.sam_object_id[self.progress_slider.value()]
        click_action = self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['labels']
        pos_click_position = self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['pos']
        neg_click_position = self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['neg']
        
        if len(click_action) > 0 and click_action[-1] == 1 and len(pos_click_position) > 0:
            if len(pos_click_position) > 0:
                self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['pos'].pop()
                self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['raw_pos'].pop()
                self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['labels'].pop()
        elif len(click_action) > 0 and click_action[-1] == -1 and len(neg_click_position) > 0:
            if len(neg_click_position) > 0:
                self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['neg'].pop()
                self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['raw_neg'].pop()
                self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()][sam_object_id]['labels'].pop()
        
        if self.last_frame is not None:
            self.draw_image()
    
    def remove_last_tap_annotation(self):
        
        if self.video_list[self.cur_video_idx-1] not in self.tracking_points_tap:
            self.smart_message("请先加载视频!")
            return
        if len(self.tracking_points_tap[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]['pos']) > 0:
            self.tracking_points_tap[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]['pos'].pop()
            self.tracking_points_tap[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]['raw_pos'].pop()
        if self.last_frame is not None:
            self.draw_image()
    
    def remove_last_interp_annotation(self):
        if self.video_list[self.cur_video_idx-1] not in self.tracking_points_interp:
            self.smart_message("请先加载视频!")
            return
        
        if len(self.tracking_points_interp[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]['pos']) > 0:
            self.tracking_points_interp[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]['pos'].pop()
            self.tracking_points_interp[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()]['raw_pos'].pop()
        
        update_max_point_num = [len(self.tracking_points_interp[self.video_list[self.cur_video_idx-1]][i]['pos']) for i in range(self.frame_count)]
        self.max_point_num[self.video_list[self.cur_video_idx-1]] = max(update_max_point_num)
        
        if self.last_frame is not None:
            self.draw_image()
        
    def remove_last_annotation(self):
        if self.anno_function_select.currentText() == '分割模型':
            self.remove_last_sam_annotation()
        elif self.anno_function_select.currentText() == '跟踪模型':
            self.remove_last_tap_annotation()
        elif self.anno_function_select.currentText() == '插值模型':
            self.remove_last_interp_annotation()
    
    def remove_frame_annotation(self):
        if self.video_list[self.cur_video_idx-1] not in self.tracking_points_sam:
            self.smart_message("请先加载视频!")
            return
        if self.anno_function_select.currentText() == '分割模型':
            self.sam_object_id[self.progress_slider.value()] = 0
            self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()] = [dict(
                pos=[], raw_pos=[], neg=[], raw_neg=[], labels=[]
            )]
        elif self.anno_function_select.currentText() == '跟踪模型':
            self.tracking_points_tap[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()] = dict(
                pos=[], raw_pos=[]
            )
        elif self.anno_function_select.currentText() == '插值模型':
            self.tracking_points_interp[self.video_list[self.cur_video_idx-1]][self.progress_slider.value()] = dict(
                pos=[], raw_pos=[]
            )
        if self.last_frame is not None:
            self.draw_image()
    
    def load_video_callback(self, video):
        if video is not None:
            self.load_video(video)
            self.progress.close()
            if self.cur_frame_idx >= len(self.video_list):
                return
            if self.video_cache.get(self.video_list[self.cur_video_idx]) is None and \
                self.video_cache.get_locate(self.video_list[self.cur_video_idx-1]) < 5 and \
                    self.video_cache.__len__() < 10:
                        self.request_video_quiet(self.video_list[self.cur_video_idx])
        else:
            self.progress.close()
            self.smart_message("视频加载失败，请检查网络设置")
            return
    
    def load_video_quiet_callback(self, res):
        video, name = res
        if video is not None:
            self.video_cache.add(name, video)
            print(f'{name} video load success!')
            for video_id in range(self.cur_video_idx, len(self.video_list)):
                if self.video_cache.get_locate(self.video_list[self.cur_video_idx-1]) == 5:
                    self.video_cache.pop()
                if self.video_cache.__len__() >= 10:
                    break
                if self.video_cache.get(self.video_list[video_id]) is None and \
                    self.video_cache.get_locate(self.video_list[self.cur_video_idx-1]) < 5 and \
                        self.video_cache.__len__() < 10:
                            self.request_video_quiet(self.video_list[video_id])
                            break
        else:
            print(f'Error: {name} video load failed!')
            return
        
    def load_video_async(self):
        self.progress = QProgressDialog("请等待，正在加载视频...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None)
        self.progress.setMinimumDuration(0)  # 立即显示对话框
        self.progress.show()
        
        video = self.video_cache.get(self.video_list[self.cur_video_idx-1])
        if video is not None:
            res = self.load_video(video)
            if res == -1:
                self.progress.close()
                self.smart_message("视频加载失败，请检查网络设置")
            else:
                self.progress.close()
                for video_id in range(self.cur_video_idx, len(self.video_list)):
                    
                    if self.video_cache.get_locate(self.video_list[self.cur_video_idx-1]) == 5:
                        self.video_cache.pop()
                    
                    if self.video_cache.__len__() >= 10:
                        break
                    
                    if self.video_cache.get(self.video_list[video_id]) is None and \
                        self.video_cache.get_locate(self.video_list[self.cur_video_idx-1]) < 5 and \
                            self.video_cache.__len__() < 10:
                                print('Loading video:', video_id)
                                self.request_video_quiet(self.video_list[video_id])
                                break
        else:
            self.video_cache.clear()
            self.video_thread = self.request_video_async()
            self.video_thread.finished.connect(self.load_video_callback)
            self.video_thread.start()
    
    def load_video(self, video):
        if video is None:
            return -1
        
        self.sam_config['video_path'] = os.path.join(ROOT_DIR, self.video_list[self.cur_video_idx-1])
        self.co_tracker_config['video_path'] = os.path.join(ROOT_DIR, self.video_list[self.cur_video_idx-1])
        self.frame_count = video.shape[0]
        self.sam_object_id = [0] * self.frame_count
        
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
        self.video_position_label.setText(f"帧: {self.cur_frame_idx+1}/{self.frame_count} | 视频: {self.cur_video_idx}/{len(self.video_list)}")
        # self.pre_f_button.setDisabled(True)
        self.sam_obj_pos_label.setText("标注物体: 1/1")
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
        
        # load global language annotation
        video_anno_json = self.video_anno_list[self.video_list[self.cur_video_idx-1]]
        global_instruction_C = video_anno_json['instructionC'] if 'instructionC' in video_anno_json else ''
        if (0, 0) not in self.lang_anno[self.video_list[self.cur_video_idx-1]] or self.lang_anno[self.video_list[self.cur_video_idx-1]][(0, 0)] == '':
            self.lang_anno[self.video_list[self.cur_video_idx-1]][(0, 0)] = global_instruction_C
        self.video_lang_input.setText(self.lang_anno[self.video_list[self.cur_video_idx-1]][(0, 0)])

        return 1
            
    def update_frame(self, frame_number):
        if self.video_list[self.cur_video_idx-1] not in self.ori_video or len(self.ori_video[self.video_list[self.cur_video_idx-1]]) == 0:
            self.smart_message('请先加载视频！')
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
        self.video_position_label.setText(f"帧: {self.cur_frame_idx+1}/{self.frame_count} | 视频: {self.cur_video_idx}/{len(self.video_list)}")
        
        self.sam_object_id[self.cur_frame_idx] = 0
        self.sam_obj_pos_label.setText(f"标注物体: {self.sam_object_id[self.cur_frame_idx]+1}/{len(self.tracking_points_sam[self.video_list[self.cur_video_idx-1]][self.cur_frame_idx])}")
        self.sam_pre_button.setDisabled(True)
        self.sam_next_button.setDisabled(False)
        
        if (0, 0) in self.lang_anno[self.video_list[self.cur_video_idx-1]]:
            self.video_lang_input.setText(self.lang_anno[self.video_list[self.cur_video_idx-1]][(0, 0)])
        else:
            self.video_lang_input.setText('')
        
        anno_loc, (clip_text, prim) = self.get_clip_description()
        if anno_loc is not None:
            self.clip_lang_input.setText(f"开始帧: {anno_loc[0]+1} | 结束帧: {anno_loc[1]+1}\n原子动作: {prim}\n动作描述: {clip_text}")
        else:
            self.clip_lang_input.clear()
            
    def toggle_playback(self):
        if self.play_button.isChecked():
            self.play_button.setText("暂停")
            self.current_frame = self.progress_slider.value()
            self.timer.start(30)  # Set timer to update frame every 30 ms
        else:
            self.play_button.setText("播放")
            self.timer.stop()

    def update_frame_position_label(self):
        # Update the text of the label to show the current frame position
        frame_number = self.progress_slider.value()
        # check if the frame number has keyframe
        if self.video_list[self.cur_video_idx-1] in self.keyframes and frame_number in self.keyframes[self.video_list[self.cur_video_idx-1]]:
            keyframe_type = self.keyframes[self.video_list[self.cur_video_idx-1]][frame_number]
            keyframe_type = '开始' if keyframe_type.lower() == 'start' else '结束'
            self.frame_position_label.setText(f"帧: {frame_number+1}({keyframe_type})")
        else:
            self.frame_position_label.setText(f"帧: {frame_number+1}")

        # Calculate the position for the label above the slider handle
        slider_x = self.progress_slider.x()
        slider_width = self.progress_slider.width()
        value_ratio = frame_number / (self.progress_slider.maximum() - self.progress_slider.minimum())
        label_x = slider_x + int(value_ratio * slider_width) - self.frame_position_label.width() // 2
        
        # Set the position of the label
        label_y = self.progress_slider.y() - self.frame_position_label.height() - 4
        label_x = max(slider_x, min(label_x, slider_x + slider_width - self.frame_position_label.width()))
        self.frame_position_label.move(label_x, label_y)
        self.frame_position_label.show()  # Show the label

    def get_frame_position(self):
        current_position = self.progress_slider.value()
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
            self.play_button.setText("播放")

    def set_sam_config(self):
        
        tracking_points = self.tracking_points_sam[self.video_list[self.cur_video_idx-1]]
        positive_points_all = {}
        negative_points_all = {}
        labels_all = {}
        
        if self.button_param_select.currentText() == '单帧模式':
            is_video = False
            direction = None
        elif '视频' in self.button_param_select.currentText():
            is_video = True
            if '双向' in self.button_param_select.currentText():
                direction = 'bidirection'
            elif '前向' in self.button_param_select.currentText():
                direction = 'forward'
            elif '反向' in self.button_param_select.currentText():
                direction = 'backward'

        select_frame = self.progress_slider.value()
        frame_pts = tracking_points[select_frame]
        
        # select all objects
        for obj_id, obj_pts in enumerate(frame_pts):
            positive_points, negative_points, labels = [], [], []
            if obj_pts['raw_pos'] != []:
                positive_points.extend([[pt.x(), pt.y()] for pt in obj_pts['raw_pos']])
            if obj_pts['raw_neg'] != []:
                negative_points.extend([[pt.x(), pt.y()] for pt in obj_pts['raw_neg']])
            if (obj_pts['raw_pos'] != []) or (obj_pts['raw_neg'] != []):
                labels.extend(obj_pts['labels'])
            
            positive_points_all[obj_id] = positive_points
            negative_points_all[obj_id] = negative_points
            labels_all[obj_id] = labels
        
        self.sam_config['is_video'] = is_video
        self.sam_config['direction'] = direction
        self.sam_config['positive_points'] = positive_points_all
        self.sam_config['negative_points'] = negative_points_all
        self.sam_config['labels'] = labels_all
        self.sam_config['select_frame'] = select_frame
        
        
        return 0
    
    
        return 0
    
    def smart_message(self, message, auto_close=True):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle('提示')
        msg.setText(message)
        msg.exec_()
    
    def sam_callback(self, res):
        self.sam_thread.wait()
        if res == 1:
            self.vis_sam.setChecked(True)
            self.load_res()
            self.progress.close()
            # QMessageBox.information(self, "Success", "分割处理完成!")
            self.remove_frame_annotation()
        else:
            self.progress.close()
            QMessageBox.warning(self, "Error", "分割处理失败，请重试")
            return
    
    def get_sam_async(self):
        
        if self.video_list[self.cur_video_idx-1] not in self.ori_video:
            self.smart_message("请先加载视频!")
            return
        
        curr_frame = self.progress_slider.value()
        tracking_points = self.tracking_points_sam[self.video_list[self.cur_video_idx-1]]
        
        if curr_frame not in tracking_points or len(tracking_points[curr_frame]) == 0:
            self.smart_message("请先标注!")
            return
        
        if len(tracking_points[curr_frame][0]['raw_pos']) == 0:
            self.smart_message("请先标注!")
            return
        
        self.set_sam_config()
        # self.progress = QProgressDialog("请等待，正在请求分割模型...", None, 0, 0, self)
        # self.progress.setWindowModality(Qt.WindowModal)
        # self.progress.setCancelButton(None)
        # self.progress.setMinimumDuration(0)
        # self.progress.show()
        
        # self.sam_thread = self.request_sam_async()
        # self.sam_thread.finished.connect(self.sam_callback)
        # self.sam_thread.start()
    
    def save_sam_input(self):
        self.progress = QProgressDialog("请等待，正在储存标注结果...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None)
        self.progress.setMinimumDuration(0)
        self.progress.show()
        self.get_sam_async()
        self.sam_point_anno[self.video_list[self.cur_video_idx-1]] = self.sam_config.copy()
        if os.path.exists(args.sam_input_anno):
            old_file = pickle.load(open(args.sam_input_anno, 'rb'))
            for k, v in old_file.items():
                if k not in self.sam_point_anno:
                    self.sam_point_anno[k] = v
        pickle.dump(self.sam_point_anno, open(args.sam_input_anno, 'wb'))
        self.progress.close()
    
    def get_sam_result(self):
        self.set_sam_config()
        
        masks = request_sam(self.sam_config)
        if masks is None:
            return -1
        
        frame_id = self.sam_config['select_frame']
        direction = self.sam_config['direction']
        
        video = self.ori_video[self.video_list[self.cur_video_idx-1]]
        model_config = self.sam_config.copy()
    
        if direction == 'bidirection':
            mask_images = self.get_sam_mask_on_image_bidirection(model_config, masks, video)
        else:
            mask_images = self.get_sam_mask_on_image_forward(model_config, masks, video)
        
        if mask_images is None:
            return -1
        
        mask_images = np.array([cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB) for mask_image in mask_images])
        if self.anno[self.video_list[self.cur_video_idx-1]]['sam'] is None:
            self.anno[self.video_list[self.cur_video_idx-1]]['sam'] = np.zeros((masks.shape[0], self.frame_count, *masks[0,0].shape))  
        elif self.anno[self.video_list[self.cur_video_idx-1]]['sam'].shape[0] != masks.shape[0]:
            self.anno[self.video_list[self.cur_video_idx-1]]['sam'] = np.zeros((masks.shape[0], self.frame_count, *masks[0,0].shape))
        
        if direction == 'backward':
            assert masks.shape[1] == frame_id + 1, f"masks.shape[1]: {masks.shape[1]}, frame_id: {frame_id}"
            self.anno[self.video_list[self.cur_video_idx-1]]['sam'][:, :frame_id+1] = masks[:, ::-1]
            self.sam_res[self.video_list[self.cur_video_idx-1]][:frame_id+1] = mask_images[::-1]
        elif direction == 'bidirection':
            assert masks.shape[1] == self.frame_count, f"masks.shape[1]: {masks.shape[1]}, frame_count: {self.frame_count}"
            self.sam_res[self.video_list[self.cur_video_idx-1]] = mask_images
            self.anno[self.video_list[self.cur_video_idx-1]]['sam'][:] = masks
        else:
            self.sam_res[self.video_list[self.cur_video_idx-1]][frame_id:frame_id+mask_images.shape[0]] = mask_images
            self.anno[self.video_list[self.cur_video_idx-1]]['sam'][:, frame_id:frame_id+mask_images.shape[0]] = masks
        
        return 1
    
    def tracker_callback(self, res):
        self.tracker_thread.wait()
        if res == 1:
            self.vis_tracker.setChecked(True)
            self.load_res()
            self.progress.close()
            # QMessageBox.information(self, "Success", "跟踪处理完成!")
            self.remove_frame_annotation()
        else:
            self.progress.close()
            QMessageBox.warning(self, "Error", "跟踪处理失败，请重试")
            return
            
    def get_tap_async(self):
        if self.video_list[self.cur_video_idx-1] not in self.ori_video:
            self.smart_message("请先加载视频！")
            return
        
        if self.progress_slider.value() > 100:
            self.smart_message("请务必在0-100帧的位置进行选点!")
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
        self.co_tracker_config['mode'] = 'Point Mode'
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
        
        # elif self.co_tracker_config['mode'] == 'Mask Mode':
        #     self.smart_message("开发中，请优先使用point模式")
        #     return
        #     # self.set_sam_config()
        #     # self.co_tracker_config['select_frame'] = self.sam_config['select_frame']  
        
        # elif self.co_tracker_config['mode'] == 'Grid Mode':
        #     self.smart_message("开发中，请优先使用point模式")
        #     return
        #     # self.co_tracker_config['grid_size'] = 10
            
    def get_tap_result(self, sam_config, co_tracker_config):        
        pred_tracks, pred_visibility, images = request_cotracker(sam_config, co_tracker_config)
        if pred_tracks is None:
            return -1
        frame_id = co_tracker_config['select_frame'][0][0]
        track_images = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images])

        if self.anno[self.video_list[self.cur_video_idx-1]]['track'] is None:
            self.anno[self.video_list[self.cur_video_idx-1]]['track'] = np.zeros((self.frame_count, *pred_tracks[0,0].shape))
            self.anno[self.video_list[self.cur_video_idx-1]]['visibility'] = np.zeros((self.frame_count, *pred_visibility[0,0].shape))
        
        if self.track_mode_selector.currentText() == '双向':
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
            self.smart_message("请先加载视频!")
            return

        raw_points, points, select_frame, point_num = [], [], [], []
        for frame_id, frame_pts in self.tracking_points_interp[self.video_list[self.cur_video_idx-1]].items():
            if frame_pts['raw_pos'] != []:
                raw_points.append([[pt.x(), pt.y()] for pt in frame_pts['raw_pos']])
                points.append([[pt.x(), pt.y()] for pt in frame_pts['pos']])
                select_frame.append(frame_id)
                point_num.append(len(frame_pts['raw_pos']))
        
        if len(points) < 2:
            self.smart_message("请至少在两帧上选择至少一个点进行插值!")
            return -1
        
        # check if the points number is the same in each frame
        wrong_frame_number_idx = [select_frame[i] for i, num in enumerate(point_num) if num != point_num[0]]
        if len(wrong_frame_number_idx) > 0:
            self.smart_message("请选择与起始帧数量和顺序一致的点, 发生错误的帧位置：" + str(wrong_frame_number_idx))
            return -1
        
        if self.anno[self.video_list[self.cur_video_idx-1]]['track'] is not None and self.anno[self.video_list[self.cur_video_idx-1]]['track'].shape[1] != self.max_point_num[self.video_list[self.cur_video_idx-1]]:
            self.smart_message("由于插值点数与Tracker不一致，插值失败!")
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
        # self.smart_message("插值完成!")
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
                self.track_point_num_label.setText(f"点数量: {self.max_point_num[self.video_list[self.cur_video_idx-1]]}")
                
            
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
        self.add_video_description()

    def mark_keyframe(self):
        current_frame = self.progress_slider.value()
        if self.key_frame_mode == 'start':
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
        
        if len(key_frame_list) % 2 != 0:
            self.smart_message('请检查关键帧标记是否正确，必须是start和end交替出现')
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
            self.key_frame_mode = 'start'
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
        elif key == Qt.Key_L:
            self.add_frame_discribtion()
        elif key == Qt.Key_Return:
            self.submit_description()
     
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
        anno_loc = [(idx, i) for idx, i in enumerate(key_pairs) if i[0] <= frame_number <= i[1] and i[0] != i[1]]
        if (0, 0) in self.lang_anno[self.video_list[self.cur_video_idx-1]]:
            anno_loc = [(i[0]-1, i[1]) for i in anno_loc]
        
        if len(anno_loc) == 0:
            self.smart_message('请移动到所在区域的起止帧之间')
            return
        anno_id, anno_loc = anno_loc[0]
        
        if self.lang_anno[self.video_list[self.cur_video_idx-1]][anno_loc] is not None:
            cached_lang, prim = self.lang_anno[self.video_list[self.cur_video_idx-1]][anno_loc]
        else:
            cached_lang, prim = '', ''
        # Create a dialog to get the description from the user
        dialog = TextInputDialog(cached_lang, self, False, self.video_anno_list[self.video_list[self.cur_video_idx-1]], ann_id=anno_id)
        select_lang = dialog.get_select_lang()
        if dialog.exec_() == QDialog.Accepted:
            if select_lang == '空':
                select_gt_id = [i for i in self.video_anno_list[self.video_list[self.cur_video_idx-1]]['clip_langC'] if self.video_anno_list[self.video_list[self.cur_video_idx-1]]['clip_langC'][i] == anno_id]
                if len(select_gt_id) > 0:
                    self.video_anno_list[self.video_list[self.cur_video_idx-1]]['clip_langC'][select_gt_id[0]] = None
            elif len(select_lang) > 0:
                self.video_anno_list[self.video_list[self.cur_video_idx-1]]['clip_langC'][select_lang] = None
            
            cached_lang = dialog.get_text()
            prim = dialog.get_prim()
            select_lang = dialog.get_select_lang()
            if select_lang != '空':
                self.video_anno_list[self.video_list[self.cur_video_idx-1]]['clip_langC'][select_lang] = anno_id
            self.lang_anno[self.video_list[self.cur_video_idx-1]][anno_loc] = (cached_lang, prim)
            self.clip_lang_input.setText(f"开始帧: {anno_loc[0]+1} | 结束帧: {anno_loc[1]+1}\n原子动作: {prim}\n动作描述: {cached_lang}")
        else:
            return 
        
    def add_video_description(self):
        # Create a dialog to get the description from the user
        if (0, 0) in self.lang_anno[self.video_list[self.cur_video_idx-1]]:
            cached_lang = self.lang_anno[self.video_list[self.cur_video_idx-1]][(0, 0)]
        else:
            cached_lang = ''
        
        dialog = TextInputDialog(cached_lang, self, True, self.video_anno_list[self.video_list[self.cur_video_idx-1]])
        if dialog.exec_() == QDialog.Accepted:
            video_description = dialog.get_text()
            self.lang_anno[self.video_list[self.cur_video_idx-1]][(0, 0)] = video_description
            self.video_lang_input.setText(f"视频描述: {video_description}")
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

    def synthesis_image(self, masks_list, video, positive_points_dict):
        obj_id = list(positive_points_dict.keys())
        cmap = plt.get_cmap("tab10")
        colors = [np.array([*cmap(int(i))[:3]]) for i in obj_id]
        mask_image = [torch.tensor(masks_list[i]).permute(2, 3, 1, 0).numpy() * (
            colors[i].reshape(1, 1, -1)[:, :, :, None]
        ) for i in range(len(masks_list))]
        
        mask_image = [
            (torch.tensor(mask_image[i]).permute(3, 0, 1, 2) * 255)
            .numpy()
            .astype(np.uint8) for i in range(len(mask_image))
        ]
        mix_image_list = []

        # add mask to video
        width, height = mask_image[0][0].shape[1], mask_image[0][0].shape[0]
        text_scale = width / 800
        assert video.shape[0] == mask_image[0].shape[0], f"video shape: {video.shape[0]}, mask shape: {mask_image[0].shape[0]}"
        for i in range(video.shape[0]):      
            for obj_id in range(len(masks_list)):
                mix_mask = masks_list[obj_id][i][0][:, :, None].repeat(3, axis=2)
                mix_image = np.where(mix_mask, mask_image[obj_id][i], video[i]) if obj_id == 0 else np.where(mix_mask, mask_image[obj_id][i], mix_image)
                # write number on the mask in the image by cv2
                loc = np.where(mix_mask[:,:,0])
                if len(loc[0]) == 0:
                    continue
                loc = (np.mean(loc[0]).astype(int), np.mean(loc[1]).astype(int))
                
                if loc[0] < 10:
                    loc = (10, loc[1])
                if loc[1] < 10:
                    loc = (loc[0], 10)
                if loc[0] > height - 10:
                    loc = (height - 10, loc[1])
                if loc[1] > width - 10:
                    loc = (loc[0], width - 10)
                
                cv2.putText(mix_image, str(obj_id+1), (loc[1], loc[0]), cv2.FONT_HERSHEY_TRIPLEX, text_scale, (255, 255, 255), 1, cv2.LINE_AA)
            
            mix_image_list.append(mix_image)
        
        return mix_image_list

    def get_sam_mask_on_image_forward(self, model_config, masks_list, video):
        is_video = model_config["is_video"]
        select_frame = model_config["select_frame"]
        direction = model_config["direction"]
        positive_points_dict = model_config["positive_points"]
        if not is_video:
            video = video[select_frame:select_frame + 1]
            mask_list = masks_list[:, select_frame:select_frame + 1]
        elif direction == "forward":
            video = video[select_frame:]
            mask_list = masks_list[:, select_frame:]
        elif direction == "backward":
            video = video[:select_frame+1][::-1]
            mask_list = masks_list[:, :select_frame+1]
        select_frame = 0
        
        mask_image = self.synthesis_image(mask_list, video, positive_points_dict)
        if mask_image is None:
            return None
        
        return mask_image

    def get_sam_mask_on_image_bidirection(self, model_config, masks_list, video):
        model_config["direction"] = "forward"
        mask_images_forward = self.get_sam_mask_on_image_forward(model_config, masks_list, video)
        if model_config['select_frame'] == 0:
            mask_image = mask_images_forward
        else:
            model_config["direction"] = "backward"
            mask_images_backward = self.get_sam_mask_on_image_forward(model_config, masks_list, video)
            mask_image = mask_images_backward[::-1][:-1] + mask_images_forward
        
        return mask_image
        
if __name__ == "__main__":
    
    # load annotation file
    args = argparse.ArgumentParser()
    args.add_argument('--anno_file', type=str, default='./data/ann_all_new.npy')
    args.add_argument('--out_file', type=str, default='./data/annotation.pkl')
    args.add_argument('--sam_input_anno', type=str, default='./data/sam_input_anno.pkl')
    args.add_argument('--lang_out_file', type=str, default='./data/lang_anno.pkl')
    args = args.parse_args()
    
    app = QApplication(sys.argv)
    player = VideoPlayer(args)
    player.resize(1080, 500)  # Adjusted size to accommodate the toolbar
    player.show()
    sys.exit(app.exec_())