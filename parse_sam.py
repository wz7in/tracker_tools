import pickle
import numpy as np
import cv2, os, yaml
import multiprocessing
import concurrent.futures
from tap_sam.vis_utils import extract_frames
from sam_tools import predict_sam_video, predict_sam_video_multiframe, get_sam_mask_on_image_forward, get_sam_mask_on_image_forward_mutli
from tqdm import tqdm
import argparse, json
from tap_sam.sam import Sam

USER_PATH = '/mnt/hwfile/OpenRobotLab/Annotation4Manipulation/user_config/sam/' 
CONFIG_PATH = '/mnt/hwfile/OpenRobotLab/Annotation4Manipulation/{mode}/data/ann_human/{time}/sam/'
# USER_CHECK_PATH = '/mnt/hwfile/OpenRobotLab/Annotation4Manipulation/check/sam/'
SAM_SAVE_PATH = '/mnt/hwfile/OpenRobotLab/Annotation4Manipulation/{mode}/data/ann_human/{time}/sam_mask/'
VIDEO_SAVE_PATH = '/mnt/hwfile/OpenRobotLab/Annotation4Manipulation/{mode}/data/ann_human/{time}/sam_video'
ROOT_DIR = '/mnt/hwfile/OpenRobotLab/Annotation4Manipulation'
RH20T = 'RH20T'
DROID = 'OXE_DROID'
UPDATE_VIDEO_LIST = []

def load_sam_input_config(path):
    sam_config = pickle.load(open(path, "rb"))
    return sam_config

def multi_process_predict_sam_m(sam_config, num=-1):
    config_list = [sam_config[key] for key in sam_config]
    if num != -1:
        config_list = config_list[:num]
    with multiprocessing.Pool(4) as pool:
        pool.map(predict_sam_video, config_list)
    
    return

def multi_process_predict_sam_t(sam_config, num=-1):
    config_list = [sam_config[key] for key in sam_config]
    if num != -1:
        config_list = config_list[:num]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(predict_sam_video, config) for config in config_list]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    return

def load_new_config(path):
    return pickle.loads(np.load(path)['arr_0'])

def check_person(name, time, model_sam):
    if time == 0:
        time = ''
    elif time == 1:
        time = '_1'
    elif time == 2:
        time = '_2'
    elif time == 3:
        time = '_3'
        
    with open(os.path.join(USER_PATH, name + time + '.txt')) as f:
        user_ann_list = f.readlines()
    user_ann_list = [line.strip() for line in user_ann_list]
    for line in tqdm(user_ann_list, desc='Parsing ' + name):
        parse_and_save_results(line, model_sam)

def parse_and_save_results(line, model_sam):
    mode = 'RH20T' if 'RH20T' in line else 'OXE_DROID'
    if 'ann_human' in line:
        time = int(line.split('/')[-3]) + 1
    else:
        time = 0
    video_save_path = os.path.join(VIDEO_SAVE_PATH.format(mode=mode, time=str(time)),  line.split('/')[-1])
    sam_save_path = os.path.join(SAM_SAVE_PATH.format(mode=mode, time=str(time)), line.split('/')[-1].replace('.mp4', '.npz'))
    if os.path.exists(video_save_path) and os.path.exists(sam_save_path):
        return
    
    model_config_path = os.path.join(CONFIG_PATH.format(mode=mode, time=str(time)), line.split('/')[-1].replace('.mp4', '.npz'))
    model_config = load_new_config(model_config_path)
    if model_config['is_finished']:
        return
    if time == 0:
        mask_list = predict_sam_video_multiframe(model_config, model_sam, sam_save_path, time=time, combined_mask=False)
    elif time == 1:
        mask_list = predict_sam_video_multiframe(model_config, model_sam, sam_save_path, time=time, combined_mask=True)
    elif time == 2:
        mask_list = predict_sam_video_multiframe(model_config, model_sam, sam_save_path, time=time, combined_mask=True)
    elif time == 3:
        mask_list = predict_sam_video_multiframe(model_config, model_sam, sam_save_path, time=time, combined_mask=True)
        return
    
    # mask_list = np.load(sam_save_path)['masks']
    video_path = model_config['video_path']
    video_name = video_path.split('/')[-1]
    origin_video_path = os.path.join(ROOT_DIR, mode, 'data', 'video', video_name)
    video = extract_frames(origin_video_path)
    # video, width, height = get_sam_mask_on_image_bidirection(model_config, mask_list, video, select_frame)
    video_new, width, height = get_sam_mask_on_image_forward_mutli(model_config, mask_list, video)
    # if model_config["direction"] == "forward":
    #     video_new = np.concatenate([video[:select_frame], np.array(video_new)], axis=0)
    # elif model_config["direction"] == "backward":
    #     video_new = np.concatenate([np.array(video_new), video[select_frame+1:]], axis=0)
    
    result = cv2.VideoWriter(
        video_save_path, cv2.VideoWriter_fourcc(*"XVID"), 20, (width, height)
    )
    for i in range(len(video_new)):
        result.write(video_new[i])
    result.release()
    
    UPDATE_VIDEO_LIST.append(video_save_path)
        
    return

if __name__ == "__main__":
    with open("./config/config.yaml") as f:
        sam_config = yaml.load(f, Loader=yaml.FullLoader)
    sam_config = sam_config["sam"]
    model_sam = Sam(
        sam_config["sam_ckpt_path"],
        sam_config["model_config"],
        sam_config["threshold"],
        False,
        sam_config["device"],
    )
    args = argparse.ArgumentParser()
    args.add_argument('--name', type=str)
    args.add_argument('--time', type=int, default=0)
    args = args.parse_args()
    
    check_person(args.name, args.time, model_sam)
    
    # save UPDATE_VIDEO_LIST to a file
    add_file_number = 0
    with open(os.path.join(ROOT_DIR, 'no_annotation_sam.json'), 'r+') as f:
        no_annotation = json.load(f)
        for video_path in UPDATE_VIDEO_LIST:
            assert video_path not in no_annotation
            mode = RH20T if RH20T in video_path else DROID
            time = int(video_path.split('/')[-3])
            add_file_number += 1
            video_name = video_path.split('/')[-1]
            save_path = os.path.join(ROOT_DIR, RH20T, 'data', 'ann_human', str(time+1), 'sam', video_name.replace('.mp4', '.npz'))
            no_annotation[video_path] = {
                'anno_path': '',
                'save_path': save_path
            }
        f.seek(0)
        f.truncate()
        json.dump(no_annotation, f)
    
    print(f'Add {add_file_number} files to no_annotation_sam')
    