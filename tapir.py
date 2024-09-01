import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from tapnet.models import tapir_model
from tapnet.utils import transforms
from tapnet.utils import viz_utils
from tapnet.utils import model_utils

from sam2.build_sam import build_sam2_video_predictor

MODE = 'SAM' # ['TAP','SAM']

def get_points(select_frame, video, mode='TAP'):
    colormap = viz_utils.get_colors(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(video[select_frame])
    ax.axis('off')
    if mode == 'TAP':
      ax.set_title('You can select more than 1 point by left mouse button and then close the window.')
    else:
      ax.set_title('You can select 1 point by left mouse button, and negative points by right mouse button, then close the window.')

    labels = []
    select_points = []
    negative_points = []

    # Event handler for mouse clicks
    def on_click(event):
        if event.button == 1 and event.inaxes == ax:  # Left mouse button clicked
            x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
            select_points.append(np.array([x, y]))
            labels.append(1)
            color = colormap[len(select_points) - 1]
            color = tuple(np.array(color) / 255.0)
            ax.plot(x, y, 'p', color=color, markersize=5)
            plt.draw()
        # Right mouse button clicked
        elif event.button == 3:
            x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
            negative_points.append(np.array([x, y]))
            labels.append(-1)
            color = (1, 0, 0)
            ax.plot(x, y, 'x', color=color, markersize=5)
            plt.draw()         
            
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    print('Selected points:', select_points)
    if mode == 'TAP':
      return np.array(select_points)
    print('Negative points:', negative_points)
    return np.array(select_points), np.array(negative_points), np.array(labels)

def extract_frames(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    frames = []
    success, frame = video.read()
    # Iterate over each frame in the video
    while success:
        # Append the current frame to the list
        frames.append(frame)
        # Read the next frame
        success, frame = video.read()

    # Release the video capture object
    video.release()
    return np.array(frames)

def save_multi_frames(image_list, save_path):
  if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
    for i, image in enumerate(image_list):
        cv2.imwrite(f'{save_path}/{i}.jpg', image)
    
class Tapir:
  def __init__(self, ckpt_path, select_frame, save_visualization):
    self.ckpt_path = ckpt_path
    self.select_frame = select_frame
    self.save_visualization = save_visualization
    self.model = self.load_checkpoint()

  def load_checkpoint(self):
      ckpt_state = np.load(self.ckpt_path, allow_pickle=True).item()
      params, state = ckpt_state['params'], ckpt_state['state']

      kwargs = dict(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
      kwargs.update(dict(
          pyramid_level=1,
          extra_convs=True,
          softmax_temperature=10.0
      ))
      tapir = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=kwargs)
      return tapir

  def sample_random_points(self, frame_max_idx, height, width, num_points):
    """Sample random points with (time, height, width) order."""
    y = np.random.randint(0, height, (num_points, 1))
    x = np.random.randint(0, width, (num_points, 1))
    t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
    points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
    return points

  def convert_select_points_to_query_points(self, frame, points):
    """Convert select points to query points.

    Args:
      points: [num_points, 2], in [x, y]
    Returns:
      query_points: [num_points, 3], in [t, y, x]
    """
    points = np.stack(points)
    query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
    query_points[:, 0] = frame
    query_points[:, 1] = points[:, 1]
    query_points[:, 2] = points[:, 0]
    return query_points

  def inference(self, frames, query_points):
    """Inference on one video.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8
      query_points: [num_points, 3], [0, num_frames/height/width], [t, y, x]

    Returns:
      tracks: [num_points, 3], [-1, 1], [t, y, x]
      visibles: [num_points, num_frames], bool
    """
    # Preprocess video to match model inputs format
    print('Start inference...')
    frames = model_utils.preprocess_frames(frames)
    query_points = query_points.astype(np.float32)
    frames, query_points = frames[None], query_points[None]  # Add batch dimension

    outputs = self.model(video=frames, query_points=query_points, is_training=False, query_chunk_size=32)
    tracks, occlusions, expected_dist = outputs['tracks'], outputs['occlusion'], outputs['expected_dist']

    # Binarize occlusions
    visibles = model_utils.postprocess_occlusions(occlusions, expected_dist)
    print('Inference done.')
    return tracks[0], visibles[0]

  def __call__(self, video, track_points):
    height, width = video.shape[1:3]
    resize_height = 256
    resize_width = 256 

    # resize video
    video_resized = np.array([cv2.resize(frame, (resize_width, resize_height)) for frame in video])
    query_points = self.convert_select_points_to_query_points(SELECT_FRAME, track_points)
    query_points = transforms.convert_grid_coordinates(query_points, (1, height, width), (1, resize_height, resize_width), coordinate_format='tyx')
    tracks, visibles = self.inference(video_resized, query_points)
    tracks = np.array(tracks)
    visibles = np.array(visibles)

    # Trajectory resulta
    tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))
    
    # Visualize the result
    if SAVE_VISUALIZATION:
      video_track = viz_utils.plot_tracks_v2(video, tracks, np.logical_not(visibles))
      width, height = video_track[0].shape[1], video_track[0].shape[0]
      result = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (width, height))
      for i in range(len(video_track)):
          result.write(video_track[i])
      result.release()
    return tracks, visibles

class Sam:
  def __init__(self, sam2_checkpoint, model_cfg, ann_frame_idx, device='cpu'):
    self.sam2_checkpoint = sam2_checkpoint
    self.model_cfg = model_cfg
    self.ann_frame_idx = ann_frame_idx
    self.ann_obj_id = 1
    self.threshold = 0.0
    self.predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint, device)
  
  def get_mask_on_image(self, masks, video, obj_id=None, random_color=False, visualize=False):
    if random_color:
        color = np.concatenate([np.random.random(3)], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3]])

    mask_image = torch.tensor(masks).permute(2,3,1,0).numpy() * (color.reshape(1, 1, -1)[:,:,:,None])
    mask_image = (torch.tensor(mask_image).permute(3, 0, 1, 2) * 255).numpy().astype(np.uint8)
    
    # add mask to video
    if SAVE_VISUALIZATION:
      width, height = mask_image[0].shape[1], mask_image[0].shape[0]
      result = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (width, height))
      for i in range(len(mask_image)):
        mix_mask = masks[i][0][:,:,None].repeat(3, axis=2)
        mix_image = np.where(mix_mask, mask_image[i], video[i])
        result.write(mix_image)
      result.release()
    
    return masks
  
  def __call__(self, video_path, object_points, labels):
    masks = []
    inference_state = self.predictor.init_state(video_path)
    self.predictor.reset_state(inference_state)
    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=self.ann_frame_idx,
        obj_id=self.ann_obj_id,
        points=object_points,
        labels=labels,
    )
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0)
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    masks = [video_segments[frame_idx][self.ann_obj_id] for frame_idx in sorted(video_segments.keys())]
    mask = np.stack(masks, axis=0)
    return mask
  
if __name__ == "__main__":
  
  # tap config
  TAP_CKPT_PATH = '/Users/wzqin/Library/CloudStorage/OneDrive-个人/BUAA/project/robot/tapnet/checkpoints/bootstapir_checkpoint_v2.npy'
  VIDEO_PATH = '/Users/wzqin/Library/CloudStorage/OneDrive-个人/BUAA/project/robot/tapnet/videos/1.mp4'
  IMAGE_LIST_SAVE_PATH = '/Users/wzqin/Library/CloudStorage/OneDrive-个人/BUAA/project/robot/tapnet/1'
  SELECT_FRAME = 0
  SAVE_VISUALIZATION = True
  
  # sam config
  SAM_CKPT_PATH = "//Users/wzqin/Library/CloudStorage/OneDrive-个人/BUAA/project/robot/tapnet/segment-anything-2/checkpoints/sam2_hiera_large.pt"
  MODEL_CONFIG = "//Users/wzqin/Library/CloudStorage/OneDrive-个人/BUAA/project/robot/tapnet/segment-anything-2/sam2_configs/sam2_hiera_l.yaml"
  DEVICE = 'cpu'
  
  if MODE == 'SAM':
    video = extract_frames(VIDEO_PATH)
    save_multi_frames(video, IMAGE_LIST_SAVE_PATH)
    object_points, negative_points, labels = get_points(SELECT_FRAME, video, mode='SAM')
    object_points = np.concatenate([object_points, negative_points], axis=0)
    model = Sam(SAM_CKPT_PATH, MODEL_CONFIG, SELECT_FRAME, DEVICE)
    masks = model(IMAGE_LIST_SAVE_PATH, object_points, labels)
    mask_images = model.get_mask_on_image(masks, video, visualize=True)
    os.removedirs(IMAGE_LIST_SAVE_PATH)
  else:
    video = extract_frames(VIDEO_PATH)
    object_points = get_points(SELECT_FRAME, video, mode='TAP')
    model = Tapir(TAP_CKPT_PATH, SELECT_FRAME, SAVE_VISUALIZATION)
    model(video, object_points)
    
    
    
    
    
    
    