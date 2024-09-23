
import numpy as np
import cv2
import pickle

def draw_mask_on_video(video, masks):
    for obj_id in range(len(masks)):
        for frame_id in range(len(video)):
            m = np.array(masks[obj_id][frame_id]).squeeze(0)[:,:,None].repeat(3, axis=2)
            video[frame_id] = np.where(m, np.full(video[frame_id].shape, 0), video[frame_id])
    return video

def draw_point_on_video(video, points):
    for frame_id in range(len(video)):
        for point in points[frame_id]:
            video[frame_id] = cv2.circle(video[frame_id], (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
    return video

if __name__ == '__main__':
    file = pickle.load(open('/Users/wzqin/Library/CloudStorage/OneDrive-个人/BUAA/project/robot/tracker_tools/data/annotation.json', 'rb'))['/mnt/hwfile/OpenRobotLab/wangziqin/data/oxe/example_data/berkeley_cable_routing/0.mp4']
    video = file['video']
    mask = file['sam']
    track = file['track']['track']
    video = draw_mask_on_video(video, mask)
    video = draw_point_on_video(video, track)
    
    # save to video
    out = cv2.VideoWriter('/Users/wzqin/Library/CloudStorage/OneDrive-个人/BUAA/project/robot/tracker_tools/data/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), 25, (video[0].shape[1], video[0].shape[0]))
    for frame in video:
        out.write(frame)
    out.release()
    print('save success')
    