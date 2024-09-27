import io, zipfile, json
import numpy as np
import yaml, torch
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
    direction = model_config["direction"]

    temp_image_list_save_dir = video_path.rsplit(".", 1)[0]
    video = extract_frames(video_path)
    if not is_video:
        video = video[select_frame:select_frame + 1]
    elif direction == "forward":
        video = video[select_frame:]     
    elif direction == "backward":
        video = video[:select_frame+1][::-1]
    select_frame = 0
    
    # os.system(f"rm -rf {temp_image_list_save_dir}")
    # save_multi_frames(video, temp_image_list_save_dir)
    
    positive_points_dict = model_config["positive_points"]
    negative_points_dict = model_config["negative_points"]
    labels_dict = model_config["labels"]
    
    mask_all = []
    global model_sam
    model_sam.set_video_list(video, temp_image_list_save_dir)
    for obj_idx in positive_points_dict.keys():
        positive_points = np.array(positive_points_dict[obj_idx])
        negative_points = np.array(negative_points_dict[obj_idx])
        labels = np.array(labels_dict[obj_idx])

        if len(negative_points) != 0:
            positive_points = np.concatenate([positive_points, negative_points], axis=0)
    
        masks = model_sam(positive_points, labels, select_frame, obj_idx)
        mask_all.append(masks)
   
    # mask_images = model_sam.get_mask_on_image(
    #     mask_all, video, save_path=model_config["save_path"], obj_id=list(positive_points_dict.keys())
    # )
    # os.system(f"rm -rf {temp_image_list_save_dir}")
    
    return mask_all

def bidirectional_sam(model_config):
    model_config["direction"] = "forward"
    masks_forward = forward_sam(model_config)
    if model_config['select_frame'] == 0:
        return masks_forward
    model_config["direction"] = "backward"
    masks_backward = forward_sam(model_config)
    
    # mask_image = mask_images_backward[::-1][:-1] + mask_images_forward
    masks = []
    for obj_id in range(len(masks_forward)):
        masks.append(np.concatenate([masks_backward[obj_id][::-1][:-1], masks_forward[obj_id]], axis=0))
    
    return masks
        
def forward_co_tracker(model_config):
    video_path = model_config["cotracker"]["video_path"]
    device = model_config["cotracker"]["device"]
    mode = model_config["cotracker"]["mode"]
    select_frame = model_config["cotracker"]["select_frame"]
    track_mode = model_config["cotracker"]["track_mode"]
    
    video = read_video_from_path(video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    
    if track_mode == 'Forward':
        video = video[:, select_frame[0][0]:]
        select_frame = [[0] for _ in range(len(select_frame))]
    
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
        final_pred_tracks, final_pred_visibility = [], []
        # max 100 frames per time
        curr_frame = 100
        while curr_frame < video[0].shape[0]:
            pred_tracks, pred_visibility = model_cotracker(video[:, curr_frame-100:curr_frame], queries=points[None], backward_tracking=True if curr_frame == 100 else False)
            final_pred_tracks.append(pred_tracks)
            final_pred_visibility.append(pred_visibility)
            points = pred_tracks[0, -1, :, ].cpu().numpy()
            points = np.concatenate([np.zeros((points.shape[0], 1)), points], axis=-1)
            points = torch.from_numpy(points).float().to(device)
            curr_frame += 99
        
        pred_tracks, pred_visibility = model_cotracker(video[:, curr_frame-100:], queries=points[None], backward_tracking=True if curr_frame == 100 else False)
        
        final_pred_tracks.append(pred_tracks)
        final_pred_visibility.append(pred_visibility)
        pred_tracks = torch.cat(final_pred_tracks, dim=1)
        pred_visibility = torch.cat(final_pred_visibility, dim=1)
                    
    elif mode == "Grid Mode":
        grid_size = model_config["cotracker"]["grid_size"]
        pred_tracks, pred_visibility = model_cotracker(video, grid_size=grid_size, grid_query_frame=select_frame, backward_tracking=True)
    
    else: 
        raise ValueError("mode should be mask or point")
        
    vis = Visualizer(show_first_frame=0)
    res_video = vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, save_video=False)
        
    return pred_tracks, pred_visibility, res_video

@app.route("/predict_sam", methods=["POST"])
def predict_sam_video():
    # get parameters
    model_config = json.loads(request.data)
    if model_config["direction"] == "bidirection":
        masks = bidirectional_sam(model_config)
    else:
        masks = forward_sam(model_config)
    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, "w") as zf:
        # with zf.open("mask_images.npy", "w") as f:
        #     np.save(f, mask_images)
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

@app.route("/get_video", methods=["POST"])
def get_video():
    model_config = json.loads(request.data)
    video_path = model_config["video_path"]
    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, "w") as zf:
        # send video
        with zf.open("video.mp4", "w") as f:
            with open(video_path, "rb") as video_file:
                f.write(video_file.read())
                
    zip_io.seek(0)
    return send_file(
        zip_io,
        mimetype="application/zip",
        as_attachment=True,
        download_name="video.zip",
    )

@app.route("/get_mask", methods=["POST"])
def get_mask():
    config = json.loads(request.data)
    video_path = config["video_path"]
    mask_path = video_path.rsplit(".", 1)[0] + "mask.npz"
    masks = np.load(mask_path)['masks']
    config = np.load('/mnt/petrelfs/wangziqin/project/tracker_tools/data/sam_input_anno.pkl', allow_pickle=True)
    config = config[video_path.split("/")[-1]]
    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, "w") as zf:
        npz_io = io.BytesIO()
        np.savez_compressed(npz_io, masks=masks)
        npz_io.seek(0)
        zf.writestr("masks.npy", npz_io.getvalue())
        zf.writestr("config.json", json.dumps(config))
    
    zip_io.seek(0)
    return send_file(
        zip_io,
        mimetype="application/zip",
        as_attachment=True,
        download_name="mask.zip",
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

    app.run(host="0.0.0.0", port=10087)

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