import yaml, json
import numpy as np
import requests, io, zipfile
import cv2, torch
from cotracker.utils.visualizer import Visualizer

root_url = 'http://10.140.0.185:10086'

def request_sam(config, is_video=False):
    url = f"{root_url}/predict_sam"

    # add parameters here
    config["is_video"] = is_video
    config["positive_points"] = np.array(
        [
            [158, 23],
            [158, 35],
        ]
    ).tolist()
    config["negative_points"] = np.array(
        [
            [170, 39],
        ]
    ).tolist()
    config["labels"] = np.array([1, 1, -1]).tolist()

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
                f"/home/PJLAB/wangziqin/Desktop/tracker_tools/demo.png", mask_images[0]
            )
        else:
            width, height = mask_images[0].shape[1], mask_images[0].shape[0]
            result = cv2.VideoWriter(
                f"/home/PJLAB/wangziqin/Desktop/tracker_tools/demo.avi",
                cv2.VideoWriter_fourcc(*"MJPG"),
                10,
                (width, height),
            )
            for i in range(len(mask_images)):
                result.write(mask_images[i])
            result.release()
    else:
        print("Error:", response)

def request_cotracker(sam_config, co_tracker_config):
    url = f"{root_url}/predict_cotracker"

    # add parameters here
    sam_config["is_video"] = False
    sam_config["positive_points"] = np.array(
        [
            [158, 23],
            [158, 35],
        ]
    ).tolist()
    sam_config["negative_points"] = np.array(
        [
            [170, 39],
        ]
    ).tolist()
    sam_config["labels"] = np.array([1, 1, -1]).tolist()
    co_tracker_config["points"] = np.array(
        [
            [158, 23],
            [158, 35],
        ]
    ).tolist()

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
    else:
        print("Error:", response)

    # for save
    import ipdb; ipdb.set_trace()
    Visualizer().save_video(torch.from_numpy(images), "demo")

if __name__ == "__main__":
    config_path = "/home/PJLAB/wangziqin/Desktop/tracker_tools/config/config.yaml"
    with open(config_path, "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    sam_config = model_config["sam"]
    co_tracker_config = model_config["cotracker"]
    # request_sam(sam_config, is_video=True)
    request_cotracker(sam_config, co_tracker_config)