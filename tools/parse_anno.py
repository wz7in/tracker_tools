import os
import json
from tqdm import tqdm

ROOT_DIR = '/mnt/hwfile/OpenRobotLab/Annotation4Manipulation'
RH20T = 'RH20T'
DROID = 'OXE_DROID'
TIME = 0

def save_anno(mode):
    has_annotation = dict()
    no_annotation = dict()
    
    # add RH20T data
    RH20T_VIDEO_DIR = os.path.join(ROOT_DIR, RH20T, 'data', 'video')
    for i in tqdm(os.listdir(RH20T_VIDEO_DIR)):
        if i.endswith('.mp4'):
            video_idx = i.rsplit('_', 1)[0]
            video_name = i.split('.')[0]
            video_path = os.path.join(RH20T_VIDEO_DIR, i)
            anno_path = os.path.join(ROOT_DIR, RH20T, 'data', 'ann', video_idx + '.npz')
            if mode == 'sam':
                save_path = os.path.join(ROOT_DIR, RH20T, 'data', 'ann_human', str(TIME), 'sam', video_name + '.npz')
            elif mode == 'lang':
                save_path = os.path.join(ROOT_DIR, RH20T, 'data', 'ann_human', 'lang', video_idx + '.npz')
            assert os.path.exists(anno_path), anno_path
            no_annotation[video_path] = {
                'anno_path': anno_path,
                'save_path': save_path
            }
            # print(save_path)
    print('RH20T:', len(no_annotation))
    # add DROID data
    DROID_VIDEO_DIR = os.path.join(ROOT_DIR, DROID, 'data', 'video')
    for i in tqdm(os.listdir(DROID_VIDEO_DIR)):
        if i.endswith('.mp4'):
            video_idx = i.split('_')[0]
            video_name = i.split('.')[0]
            video_path = os.path.join(DROID_VIDEO_DIR, i)
            anno_path = os.path.join(ROOT_DIR, DROID, 'data', 'ann', video_idx + '.npz')
            if mode == 'sam':
                save_path = os.path.join(ROOT_DIR, DROID, 'data', 'ann_human', str(TIME), 'sam', video_name + '.npz')
            elif mode == 'lang':
                save_path = os.path.join(ROOT_DIR, DROID, 'data', 'ann_human', 'lang', video_idx + '.npz')
            assert os.path.exists(anno_path), anno_path
            no_annotation[video_path] = {
                'anno_path': anno_path,
                'save_path': save_path
            }
            # print(save_path)
    print('DROID:', len(no_annotation))
    
    # save all data
    with open(os.path.join(ROOT_DIR, f'no_annotation_{mode}.json'), 'w') as f:
        json.dump(no_annotation, f)
    with open(os.path.join(ROOT_DIR, f'has_annotation_{mode}.json'), 'w') as f:
        json.dump(has_annotation, f)


if __name__ == '__main__':
    for mode in ['sam']:
        print(f"Processing {mode} data...")
        save_anno(mode)