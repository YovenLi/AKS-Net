import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

import torch
from torch.utils import data


class AKSDataset(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, datainfo_path, transform, crop_size, key_frame_num=4):
        super(AKSDataset, self).__init__()


        dataInfo = pd.read_csv(datainfo_path, header=0, sep=',', index_col=False, encoding="utf-8-sig")

        self.video_names = dataInfo['name']
        self.moss = dataInfo['mos']

        self.crop_size = crop_size
        self.data_dir = data_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.video_length_read = key_frame_num

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names.iloc[idx]
        frames_dir = os.path.join(self.data_dir, video_name)

        mos = self.moss.iloc[idx]

        video_channel = 3
        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        video_length_read = self.video_length_read
        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])
        files = os.listdir(frames_dir)
        video_read_index = 0
        for file in files:

            frame_name = os.path.join(frames_dir, file)

            if os.path.exists(frame_name):
                read_frame = Image.open(frame_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_video[video_read_index] = read_frame

                video_read_index += 1
            else:
                print(frame_name)
                print('120 frames do not exist!')

        if video_read_index < video_length_read:
            for j in range(video_read_index, video_length_read):
                transformed_video[j] = transformed_video[video_read_index - 1]

             

        return transformed_video, mos, video_name



