
import cv2
import numpy as np
from tapnet.models import tapir_model
from tapnet.utils import transforms
from tapnet.utils import viz_utils
from tapnet.utils import model_utils

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

  def __call__(self, video, track_points, save_path=None):
    
    if self.save_visualization:
        assert save_path is not None, 'while save_visualizationis True, save_path must be provided.'  
    
    height, width = video.shape[1:3]
    resize_height = 256
    resize_width = 256 

    # resize video
    video_resized = np.array([cv2.resize(frame, (resize_width, resize_height)) for frame in video])
    query_points = self.convert_select_points_to_query_points(self.select_frame, track_points)
    query_points = transforms.convert_grid_coordinates(query_points, (1, height, width), (1, resize_height, resize_width), coordinate_format='tyx')
    tracks, visibles = self.inference(video_resized, query_points)
    tracks = np.array(tracks)
    visibles = np.array(visibles)

    # Trajectory resulta
    tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))
    
    # Visualize the result
    if self.save_visualization:
      video_track = viz_utils.plot_tracks_v2(video, tracks, np.logical_not(visibles))
      width, height = video_track[0].shape[1], video_track[0].shape[0]
      result = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (width, height))
      for i in range(len(video_track)):
          result.write(video_track[i])
      result.release()
    return tracks, visibles
