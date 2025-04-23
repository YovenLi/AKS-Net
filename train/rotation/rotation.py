import numpy as np
import time
import open3d as o3d
import os
from PIL import Image
from torchvision import transforms
import cv2
import argparse
import math


def generate_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# Camera Rotation
def camera_rotation(path, frame_path, video_path):
    pcd = o3d.io.read_point_cloud(path)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    ctrl = vis.get_view_control()
    tmp = 0
    interval = 5.82

    start = time.time()

    # begin rotation
    while tmp < 120:
        tmp += 1
        if tmp < 30:
            ctrl.rotate(12 * interval, 0)
        elif tmp >= 30 and tmp < 60:
            ctrl.rotate(0, 12 * interval)
        elif tmp >= 60 and tmp < 90:
            ctrl.rotate(12 * interval / math.sqrt(2), 12 * interval / math.sqrt(2))
        elif tmp >= 90 and tmp < 120:
            ctrl.rotate(12 * interval / math.sqrt(2), -12 * interval / math.sqrt(2))
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(True)
        img = Image.fromarray((np.asarray(img) * 255).astype(np.uint8))

        img.save(frame_path + '/' + str(tmp).zfill(3) + '.png')

        # save videos
        # video.write(cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR))

    end = time.time()
    print("time consuming: ", end - start)
    vis.destroy_window()
    del ctrl
    del vis


def projection(path, frame_path, video_path):
    # find all the objects
    objs = os.walk(path)
    for path, dir_list, file_list in objs:
        for obj in file_list:
            one_object_path = os.path.join(path, obj)
            camera_rotation(one_object_path, 
                            generate_dir(os.path.join(frame_path, obj)), 
                            generate_dir(os.path.join(video_path, obj)))


def main(config):
    frame_path = config.frame_path
    video_path = config.video_path
    generate_dir(frame_path)
    generate_dir(video_path)
    projection(config.path, frame_path, video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='./ply/')  # path to the file that contain .ply models
    parser.add_argument('--frame_path', type=str, default='./frames/')  # path to the generated frames
    parser.add_argument('--video_path', type=str, default='./videos/')  # path to the generated videos (optional)
    
    config = parser.parse_args()
    main(config)
