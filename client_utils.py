import pickle, json
import numpy as np
import requests, io, zipfile
import imageio
from cotracker.utils.visualizer import Visualizer

root_url = 'http://10.140.0.204:10087'

def request_sam(config, mode):
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

def request_cotracker(sam_config, co_tracker_config):
    url = f"{root_url}/predict_cotracker"    

    model_config = {
        "sam": sam_config,
        "cotracker": co_tracker_config,
    }
    response = requests.post(
        url, data=json.dumps(model_config), headers={"content-type": "application/json"}
    )

    if response.status_code == 200:
        zip_io = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_io, "r") as zf:
            with zf.open("pred_tracks.npy") as f:
                pred_tracks = np.load(f)
            with zf.open("pred_visibility.npy") as f:
                pred_visibility = np.load(f)
            with zf.open("images.npy") as f:
                images = np.load(f)
        return pred_tracks, pred_visibility, np.transpose(np.squeeze(images, axis=0), (0,2,3,1))[...,::-1]
    else:
        print("Error:", response)
        return None, None, None

def request_video(video_path):
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

def request_video_and_anno(mode, username, button_mode):
    if mode == 'lang':
        url = f"{root_url}/get_video_and_anno_lang"
    else:
        url = f"{root_url}/get_video_and_anno_sam"
    
    config = {
        "username": username,
        "mode": button_mode,
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
        
        if not is_finished:
            frames = []
            reader = imageio.get_reader(video, "mp4")
            for _, im in enumerate(reader):
                frames.append(np.array(im))
            if mode == 'lang':
                return np.stack(frames), anno, save_path, video_path, int(history_number)
            else:
                return np.stack(frames), save_path, video_path, int(history_number)
    else:
        print("Error:", response)
        if mode == 'lang':
            return None, None, None, None, 0
        else:
            return None, None, None, 0

def save_anno(save_path, anno):
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

if __name__ == "__main__":
    a,b = request_video_and_anno('sam')
    print(a.shape, b)
    # save_anno(c, b)
    