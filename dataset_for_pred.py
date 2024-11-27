import numpy as np
import glob
import cv2
import torch
from torch.utils.data import Dataset
import os
import re

class SimpleDataset(Dataset):
    def __init__(self, video_path, vid_pad):
        self.video_path = video_path
        self.vid_pad = vid_pad
        self.data = self._load_video_files(video_path)
        
        if not self.data:
            raise ValueError("No valid video files found")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid_path = self.data[idx]
        vid = self._load_vid(vid_path)
        if vid is None:
            raise ValueError(f"Failed to load video: {vid_path}")
        
        vid_len = min(vid.shape[0], self.vid_pad)
        return {'vid': torch.FloatTensor(vid.transpose(3, 0, 1, 2)), 'vid_len': vid_len}

    def _load_video_files(self, path):
        # 指定されたパスの全ビデオファイルを取得
        video_files = glob.glob(os.path.join(path, '*'))
        return sorted(video_files)

    def _load_vid(self, p):
        # pはフレーム画像が格納されているディレクトリである必要があります
        frame_directory = os.path.dirname(p)
        
        # フルパスを生成して確認
        full_path = os.path.abspath(frame_directory)
        print(f"Full video directory path: {full_path}")
        
        print(f"Loading video frames from: {frame_directory}")

        # ディレクトリ内の全ファイルを表示（デバッグ用）
        all_files = os.listdir(frame_directory)
        print(f"Files found in directory: {all_files}")

        # 正しいファイル名パターンに修正（必要に応じて）
        files = sorted(glob.glob(os.path.join(frame_directory, 'frame_*.jpg')), 
                    key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
        
        if not files:
            print(f"Warning: No frame images found in {frame_directory}")
            return None

        frames = []
        for file in files:
            img = cv2.imread(file)
            if img is None:
                print(f"Failed to load image: {file}")
                continue
            if img.shape[:2] != (64, 128):
                img = cv2.resize(img, (128, 64), interpolation=cv2.INTER_LANCZOS4)
            frames.append(img)
        
        if not frames:
            return None

        array = np.stack(frames, axis=0).astype(np.float32)

        if array.shape[0] < self.vid_pad:
            pad_width = ((0, self.vid_pad - array.shape[0]), (0, 0), (0, 0), (0, 0))
            array = np.pad(array, pad_width, mode='constant', constant_values=0)
        elif array.shape[0] > self.vid_pad:
            indices = np.linspace(0, array.shape[0] - 1, self.vid_pad, dtype=np.int32)
            array = array[indices]

        return array

