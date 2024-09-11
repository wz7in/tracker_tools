import yaml, json
import numpy as np
import requests, io, zipfile
import cv2, torch
from cotracker.utils.visualizer import Visualizer

# root_url = 'http://10.140.1.22:10087'
root_url = 'http://10.140.0.145:10088'
# root_url = 'http://127.0.0.1:10086'

def request_sam(config):
    url = f"{root_url}/predict_sam"

    # add parameters here
    is_video = config["is_video"]

    response = requests.post(
        url, data=json.dumps(config), headers={"content-type": "application/json"}
    )

    if response.status_code == 200:
        zip_io = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_io, "r") as zf:
            with zf.open("masks.npy") as f:
                masks = np.load(f)
            with zf.open("mask_images.npy") as f:
                mask_images = np.load(f)
        # for save
        if not is_video:
            cv2.imwrite(
                f"./demo.png", mask_images[0]
            )
        else:
            width, height = mask_images[0].shape[1], mask_images[0].shape[0]
            result = cv2.VideoWriter(
                f"./demo.avi",
                cv2.VideoWriter_fourcc(*"MJPG"),
                10,
                (width, height),
            )
            for i in range(len(mask_images)):
                result.write(mask_images[i])
            result.release()
        return masks, mask_images
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
    # add parameters here
    config = {
        "video_path": video_path,
    }
    response = requests.post(
        url, data=json.dumps(config), headers={"content-type": "application/json"}
    )
    if response.status_code == 200:
        zip_io = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_io, "r") as zf:
            with zf.open("video.npy") as f:
                video = np.load(f)
        return video
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