import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tapnet.utils import viz_utils

def get_points(select_frame, video, mode='TAP'):
    colormap = viz_utils.get_colors(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(video[select_frame])
    ax.axis('off')
    if mode == 'TAP':
      ax.set_title('You can select more than 1 point by left mouse button \n and then close the window.')
    else:
      ax.set_title('You can track multi objects by select more than 1 point by left mouse button \n and negative points by right mouse button, then close the window.')

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
