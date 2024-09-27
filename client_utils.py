import yaml, json
import numpy as np
import requests, io, zipfile
import imageio
from cotracker.utils.visualizer import Visualizer

root_url = 'http://10.140.0.146:10087'

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

if __name__ == "__main__":
    config_path = "./config/config.yaml"
    with open(config_path, "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    sam_config = model_config["sam"]
    co_tracker_config = model_config["cotracker"]
    # request_sam(sam_config, is_video=True)
    request_cotracker(sam_config, co_tracker_config)