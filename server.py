import io, zipfile, os, json
import numpy as np
import time, yaml, torch
from flask import Flask, request, send_file
from tap_sam.sam import Sam
from tap_sam.vis_utils import extract_frames, save_multi_frames
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import read_video_from_path, Visualizer

app = Flask(__name__)
model_sam, model_cotracker = None, None

def forward_sam(model_config):
    video_path = model_config["video_path"]
    is_video = model_config["is_video"]
    select_frame = model_config["select_frame"]

    temp_image_list_save_dir = video_path.rsplit(".", 1)[0]
    video = extract_frames(video_path)
    if not is_video:
        video = video[select_frame : select_frame + 1]
        select_frame = 0
    save_multi_frames(video, temp_image_list_save_dir)
    # get positave points and negative points
    positive_points = np.array(model_config["positive_points"])
    negative_points = np.array(model_config["negative_points"])
    labels = np.array(model_config["labels"])
    print(positive_points.shape, negative_points.shape, labels.shape)
    if len(negative_points) != 0:
        positive_points = np.concatenate([positive_points, negative_points], axis=0)

    frame_length = len(os.listdir(temp_image_list_save_dir))
    sam_start_time = time.time()
    
    global model_sam
    masks = model_sam(temp_image_list_save_dir, positive_points, labels, select_frame)
    sam_end_time = time.time()
    print(
        f"SAM processing {frame_length} frames in {sam_end_time-sam_start_time} s. {(sam_end_time-sam_start_time)/frame_length} s per frame."
    )

    mask_images, masks = model_sam.get_mask_on_image(
        masks, video, save_path=model_config["save_path"]
    )
    os.system(f"rm -rf {temp_image_list_save_dir}")
    
    return mask_images, masks

def forward_co_tracker(model_config):
    video_path = model_config["cotracker"]["video_path"]
    device = model_config["cotracker"]["device"]
    mode = model_config["cotracker"]["mode"]
    select_frame = model_config["cotracker"]["select_frame"]
    
    print(video_path)
    video = read_video_from_path(video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    
    global model_cotracker
    model_cotracker = model_cotracker.to(device)
    
    if mode == "Mask Mode":
        assert model_config["sam"]["is_video"] == False, "mask mode only support single frame"
        sample_points_number = model_config["cotracker"]["sample_points_number"]
        _, masks = forward_sam(model_config["sam"])
        points = np.argwhere(masks[0][0] > 0)
        
        # switch x and y, and sample sample_points_number
        points = points[:, [1, 0]]
        if points.shape[0] > sample_points_number:
            idx = np.random.choice(points.shape[0], sample_points_number, replace=False)
            points = points[idx]
        
        # padding with frame id
        points = np.concatenate([np.array([[select_frame]] * points.shape[0]), points], axis=-1)
        points = torch.from_numpy(points).float().to(device)
        pred_tracks, pred_visibility = model_cotracker(video, queries=points[None], backward_tracking=True)
    elif mode == "Point Mode":
        points = np.array(model_config["cotracker"]["points"])
        points = np.concatenate([np.array(select_frame), points], axis=-1)
        # points = np.concatenate([np.array([[select_frame]] * points.shape[0]), points], axis=-1)
        points = torch.from_numpy(points).float().to(device)
        pred_tracks, pred_visibility = model_cotracker(video, queries=points[None], backward_tracking=True)
    elif mode == "Grid Mode":
        grid_size = model_config["cotracker"]["grid_size"]
        pred_tracks, pred_visibility = model_cotracker(video, grid_size=grid_size, grid_query_frame=select_frame, backward_tracking=True)
    else: 
        raise ValueError("mode should be mask or point")
        
    # save_path = model_config.get("save_path")
    # save_dir, file_name = save_path.rsplit("/", 1)
    vis = Visualizer()
    res_video = vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, save_video=False)

@app.route("/predict_sam", methods=["POST"])
def predict_sam_video():
    # get parameters
    model_config = json.loads(request.data)
    mask_images, masks = forward_sam(model_config)
    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, "w") as zf:
        with zf.open("mask_images.npy", "w") as f:
            np.save(f, mask_images)
        with zf.open("masks.npy", "w") as f:
            np.save(f, masks)

    zip_io.seek(0)
    return send_file(
        zip_io,
        mimetype="application/zip",
        as_attachment=True,
        download_name="arrays.zip",
    )

@app.route("/predict_cotracker", methods=["POST"])
def predict_cotracker():
    model_config = json.loads(request.data)
    pred_tracks, pred_visibility, images = forward_co_tracker(model_config)
    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, "w") as zf:
        with zf.open("pred_tracks.npy", "w") as f:
            np.save(f, pred_tracks.cpu().numpy())
        with zf.open("pred_visibility.npy", "w") as f:
            np.save(f, pred_visibility.cpu().numpy())
        with zf.open("images.npy", "w") as f:
            np.save(f, images)

    zip_io.seek(0)
    return send_file(
        zip_io,
        mimetype="application/zip",
        as_attachment=True,
        download_name="arrays.zip",
    )
      

if __name__ == "__main__":

    with open("./config/config.yaml") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    sam_config = model_config["sam"]
    co_tracker_config = model_config["cotracker"]
    
    model_sam = Sam(
        sam_config["sam_ckpt_path"],
        sam_config["model_config"],
        sam_config["threshold"],
        False,
        sam_config["device"],
    )

    model_cotracker = CoTrackerPredictor(
        checkpoint=co_tracker_config["cotracker_ckpt_path"]
    )

    app.run(host="0.0.0.0", port=10086)

    # model_cotracker = CoTrackerPredictor(
    #     checkpoint='/mnt/petrelfs/wangziqin/project/tracker_tools/co-tracker/checkpoints/cotracker2.pth'
    # ).to('cuda')
    # video = read_video_from_path('/mnt/petrelfs/wangziqin/project/tracker_tools/demo/demo.mp4')
    # video = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0).float().to('cuda')
    
    # points = torch.tensor([
    #     [0, 284, 125], 
    #     [0, 301, 107]
    # ]).float().to('cuda')
    # pred_tracks, pred_visibility = model_cotracker(video, queries=points[None], backward_tracking=True)
    # # pred_tracks, pred_visibility = model_cotracker(video, grid_size=10)
    # vis = Visualizer(save_dir='./videos', pad_value=100)
    # vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename='teaser')