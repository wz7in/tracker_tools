import os
import yaml
import argparse
import numpy as np
from tap_sam.sam import Sam
from tap_sam.tapir import Tapir
from tap_sam.vis_utils import get_points, extract_frames, save_multi_frames

if __name__ == "__main__":
  
  args = argparse.ArgumentParser()
  args.add_argument('--mode', type=str, default='sam', help='tap or sam')
  args.add_argument('--config', type=str, default='config.yaml', help='config file path')
  args.add_argument('--device', type=str, default='cpu', help='device')
  args = args.parse_args()
  
  # read config from config file
  with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
  
  assert config['mode'] == args.mode, 'mode of config file and args should be same'
  
  if args.mode == 'sam':
    if config['save_visualization']:
      assert 'save_path' in config, 'save_path should be provided in config file if save_visualization is True'
    temp_image_list_save_dir = config['video_path'].rsplit('.', 1)[0]
    video = extract_frames(config['video_path'])
    save_multi_frames(video, temp_image_list_save_dir)
    
    # get positave points and negative points
    positive_points, negative_points, labels = get_points(config['select_frame'], video, mode='SAM')
    if len(negative_points)!=0:
      positive_points = np.concatenate([positive_points, negative_points], axis=0)
    
    model = Sam(config['sam_ckpt_path'], config['model_config'], config['select_frame'], config['threshold'], config['save_visualization'], args.device)
    masks = model(temp_image_list_save_dir, positive_points, labels)
    mask_images = model.get_mask_on_image(masks, video, save_path=config['save_path'])
    os.system(f'rm -rf {temp_image_list_save_dir}')
  
  elif args.mode == 'tap':
    if config['save_visualization']:
      assert 'save_path' in config, 'save_path should be provided in config file if save_visualization is True'
    
    video = extract_frames(config['video_path'])
    object_points = get_points(config['select_frame'], video, mode='TAP')
    
    model = Tapir(config['tap_ckpt_path'], config['select_frame'], config['save_visualization'])
    model(video, object_points, save_path=config['save_path'])
  
  else:
    raise ValueError('mode should be either sam or tap')
    
    
    
    
    
    
    