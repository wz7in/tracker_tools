import pickle
import random, json
import numpy as np
import requests, io, zipfile
import imageio
from cotracker.utils.visualizer import Visualizer

base_url = 'http://{ip}:{port}'

def request_sam(ip, port, config, mode):
    root_url = base_url.format(ip=ip, port=port)
    if mode == "online":
        url = f"{root_url}/predict_sam"
    else:
        url = f"{root_url}/get_mask"
    # add parameters here
    response = requests.post(
        url, data=json.dumps(config), headers={"content-type": "application/json"}
    )
    if response.status_code == 200:
        zip_io = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_io, "r") as zf:
            if mode == "online":
                with zf.open("masks.npy") as f:
                    masks = np.load(f)
                return masks
            if mode == "offline":
                with zf.open("config.json") as f:
                    config = json.load(f)
                with zf.open("masks.npy") as f:
                    masks = np.load(f)['masks']
                return config, masks
    else:
        print("Error:", response)
        return None, None

def request_video(ip, port, video_path):
    root_url = base_url.format(ip=ip, port=port)
    url = f"{root_url}/get_video"
    config = {
        "video_path": video_path,
    }
    response = requests.post(
        url, data=json.dumps(config), headers={"content-type": "application/json"}
    )
    if response.status_code == 200:
        zip_io = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_io, "r") as zf:
            with zf.open("video.mp4") as f:
                video = f.read()
        frames = []
        reader = imageio.get_reader(video, "mp4")
        for i, im in enumerate(reader):
            frames.append(np.array(im))
        return np.stack(frames)
    else:
        print("Error:", response)
        return None

def request_video_and_anno(ip, port, mode, username, button_mode, last_video_path, re_anno=0):
    root_url = base_url.format(ip=ip, port=port)
    if mode == 'lang':
        url = f"{root_url}/get_video_and_anno_lang"
    else:
        url = f"{root_url}/get_video_and_anno_sam"
    
    config = {
        "username": username,
        "mode": button_mode,
        "last_video_path": last_video_path,
        "re_anno": re_anno
    }
    
    response = requests.post(
        url, data=json.dumps(config), headers={"content-type": "application/json"}, stream=True
    )
    if response.status_code == 200:
        zip_io = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_io, "r") as zf:
            with zf.open("is_finished") as f:
                is_finished = f.read().decode("utf-8")
                
            is_finished = is_finished == 'True'
            if is_finished:
                return 0
            
            if not is_finished:
                with zf.open("video.mp4") as f:
                    video = f.read()
                if mode == 'lang':
                    with zf.open("anno.npz") as f:
                        anno = np.load(f)['anno_file']
                        anno = pickle.loads(anno)
                with zf.open("save_path") as f:
                    save_path = f.read().decode("utf-8")
                with zf.open("video_path") as f:
                    video_path = f.read().decode("utf-8")
                with zf.open("history_number") as f:
                    history_number = f.read().decode("utf-8")
                if mode != 'lang':
                    with zf.open("all_one_anno_num") as f:
                        all_one_anno_num = int(f.read().decode("utf-8"))
                    with zf.open("one_anno_num") as f:
                        one_anno_num = int(f.read().decode("utf-8"))
                    with zf.open("all_two_anno_num") as f:
                        all_two_anno_num = int(f.read().decode("utf-8"))
                    with zf.open("two_anno_num") as f:
                        two_anno_num = int(f.read().decode("utf-8"))
                    with zf.open("all_three_anno_num") as f:
                        all_three_anno_num = int(f.read().decode("utf-8"))
                    with zf.open("three_anno_num") as f:
                        three_anno_num = int(f.read().decode("utf-8"))
        
        frames = []
        reader = imageio.get_reader(video, "mp4")
        for _, im in enumerate(reader):
            frames.append(np.array(im))
        if mode == 'lang':
            return np.stack(frames), anno, save_path, video_path, int(history_number)
        else:
            return np.stack(frames), save_path, video_path, int(history_number), \
                one_anno_num, all_one_anno_num, two_anno_num, all_two_anno_num, three_anno_num, all_three_anno_num
    else:
        print("Error:", response)
        if mode == 'lang':
            return None
        else:
            return None

def save_anno(ip, port, save_path, anno):
    root_url = base_url.format(ip=ip, port=port)
    url = f"{root_url}/save_anno"
    # save as binary file
    anno_bytes = io.BytesIO()
    np.savez_compressed(anno_bytes, anno_file=anno)
    anno_bytes.seek(0)
    files = {
        "file": ("anno.npz", anno_bytes, "application/octet-stream"),
        "save_path": (None, save_path),
    }
    response = requests.post(url, files=files)
    if response.status_code == 200:
        return True
    else:
        print("Error:", response)
        return False

def drawback_video(ip, port, video_path, mode):
    root_url = base_url.format(ip=ip, port=port)
    if mode == 'lang':
        url = f"{root_url}/drawback_video_lang"
    else:
        url = f"{root_url}/drawback_video_sam"
    config = {
        "video_path": video_path,
    }
    response = requests.post(
        url, data=json.dumps(config), headers={"content-type": "application/json"}
    )
    if response.status_code == 200:
        return True
    else:
        print("Error:", response)
        return False

def get_avaiable_username(ip, port, username):
    root_url = base_url.format(ip=ip, port=port)
    url = f"{root_url}/is_available_user"
    config = {
        "user_name": username,
    }
    response = requests.post(
        url, data=json.dumps(config), headers={"content-type": "application/json"}
    )
    if response.status_code == 200:
        zip_io = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_io, "r") as zf:
            with zf.open("user_name") as f:
                username = f.read().decode("utf-8")
        return username.strip()
    else:
        print("Error:", response)
        return None

if __name__ == "__main__":
    # a,b = request_video_and_anno('sam')
    # print(a.shape, b)
    # save_anno(c, b)
    
    print(get_avaiable_username('10.140.0.131', 'test1'))
    