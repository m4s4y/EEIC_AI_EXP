import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
from cvtransforms import *
import torch
import re
import copy
import json
import random
import editdistance

class MyDataset(Dataset):
    letters = ['I', 'N', 'U', 'a', 'b', 'by', 'ch', 'cl', 
               'd', 'dy', 'e', 'f', 'fy', 'g', 'gw', 'gy', 
               'h', 'hy', 'i', 'j', 'k', 'kw', 'ky', 'm', 
               'my', 'n', 'ny', 'o', 'p', 'py', 'r', 
               'ry', 's', 'sh', 't', 'ts', 'ty', 'u', 
               'v', 'w', 'y', 'z', 'sil', 'pau'
    ]  # pauとsil取り除いた

    def __init__(self, video_path, anno_path, file_list, vid_pad, txt_pad, phase):
        self.anno_path = anno_path
        self.vid_pad = vid_pad
        self.txt_pad = txt_pad
        self.phase = phase

        # 音素インデックス辞書を作成
        self.phoneme_dict = {phoneme: idx for idx, phoneme in enumerate(self.letters)}
        self.sil_index = self.phoneme_dict['sil']  # 'sil'のインデックス
        self.pau_index = self.phoneme_dict['pau']  # 'pau'のインデックス
        
        print(f"\nInitializing dataset...")
        print(f"Video path: {video_path}")
        print(f"Anno path: {anno_path}")
        print(f"File list: {file_list}")
        print(f"sil index: {self.sil_index}")
        # データリストの初期化
        self.data = []
        
        # ファイルリストを読み込む
        with open(file_list, 'r') as f:
            for line in f.readlines():
                video_name = line.strip()
                video_name = video_name.replace("LFROI_", "")  # LFROIがついてないファイル構成の場合
                video_scene_dir_pattern = os.path.join(video_path, video_name, "scene_*")
                scene_dirs = glob.glob(video_scene_dir_pattern)
                
                # シーンごとのディレクトリが存在するか確認
                if not scene_dirs:
                    print(f"Skipping {video_name}: no scene directories found matching pattern {video_scene_dir_pattern}")
                    continue
                
                # アノテーションファイルのパスを構築
                base_name = video_name  # ROHAN4600_****
                anno_file_pattern = os.path.join(anno_path, base_name, "scene_*.txt")
                anno_files = glob.glob(anno_file_pattern)
                
                if not anno_files:
                    print(f"Skipping {video_name}: no annotation files found matching pattern {anno_file_pattern}")
                    continue

                # シーンごとのディレクトリとアノテーションファイルをペアにしてリストに追加
                for scene_dir in scene_dirs:
                    # シーン番号を取得
                    scene_number = os.path.basename(scene_dir).split('_')[1]
                    # 対応するアノテーションファイルを探す
                    corresponding_anno_file = os.path.join(anno_path, base_name, f"scene_{scene_number}.txt")
                    
                    if corresponding_anno_file in anno_files:
                        self.data.append((scene_dir, corresponding_anno_file, video_name))
                    else:
                        print(f"Skipping scene {scene_number}: annotation not found for {scene_dir}")
        
        if not self.data:
            raise ValueError("No valid scenes found with corresponding annotations")
            
        print(f"Successfully loaded {len(self.data)} scenes")

    def __len__(self):
        """データセットの長さを返す"""
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            (vid_path, anno_file, name) = self.data[idx]
            
            # ビデオフレームの読み込み
            vid = self._load_vid(vid_path)
            if vid is None:
                raise ValueError(f"Failed to load video: {vid_path}")
            
            # アノテーションの読み込み
            anno = self._load_anno(anno_file)
            if len(anno) == 0:
                raise ValueError(f"Failed to load annotation: {anno_file}")

            if self.phase == 'train':
                vid = HorizontalFlip(vid)
            
            vid = ColorNormalize(vid)
            vid_len = min(vid.shape[0], self.vid_pad)  # 実際の長さとパディング長の小さい方
            anno_len = len(anno)
            anno = self._padding(anno, self.txt_pad)
            
            return {'vid': torch.FloatTensor(vid.transpose(3, 0, 1, 2)),
                   'txt': torch.LongTensor(anno),
                   'txt_len': anno_len,
                   'vid_len': vid_len}
                   
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            raise


    def _load_vid(self, p): 
        """
        フレーム画像を読み込み、指定された長さにパディングまたはダウンサンプリング
        """
        files = sorted(glob.glob(os.path.join(p, 'frame_*.jpg')), 
                    key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
        if not files:
            print(f"Warning: No frame images found in {p}")
            return None
        
        # フレーム画像を読み込む
        frames = []
        for file in files:
            img = cv2.imread(file)
            if img is not None:
                # サイズが異なる場合はリサイズ
                if img.shape[:2] != (64, 128):
                    img = cv2.resize(img, (128, 64), interpolation=cv2.INTER_LANCZOS4)
                frames.append(img)
            else:
                print(f"Warning: Could not read image: {file}")
        
        if not frames:
            return None

        # フレームを numpy 配列に変換
        array = np.stack(frames, axis=0).astype(np.float32)
        
        # フレーム数の調整
        if array.shape[0] < self.vid_pad:
            # 不足分を0で埋める
            pad_width = ((0, self.vid_pad - array.shape[0]), (0, 0), (0, 0), (0, 0))
            array = np.pad(array, pad_width, mode='constant', constant_values=0)
        elif array.shape[0] > self.vid_pad:
            # フレーム数が多い場合はダウンサンプリング
            indices = np.linspace(0, array.shape[0] - 1, self.vid_pad, dtype=np.int32)
            array = array[indices]
        
        return array

    def _load_anno(self, name):
        """
        時間情報付きのアノテーションファイルを読み込む
        """
        try:
            if not os.path.exists(name):
                print(f"Annotation file not found: {name}")
                return np.array([])
                
            with open(name, 'r') as f:
                lines = [line.strip().split() for line in f.readlines()]
                if not lines:
                    print(f"Empty annotation file: {name}")
                    return np.array([])
                
                phonemes = [line[2] for line in lines]
                #phonemes = [p for p in phonemes if p.upper() not in ['SIL', 'PAU']]
                phonemes = [p for p in phonemes]

                
                if not phonemes:
                    print(f"No valid phonemes found in: {name}")
                    return np.array([])
                
                txt = ' '.join(phonemes)
                #print(f"Processed annotation: {txt}")
                
                return self.txt2arr(txt, 1)
                
        except Exception as e:
            print(f"Error loading annotation {name}: {str(e)}")
            return np.array([])

    def _padding(self, array, length):
        """配列のパディング処理"""
        if not isinstance(array, (list, np.ndarray)) or len(array) == 0:
            raise ValueError("Invalid array for padding")
        
        if isinstance(array, list):
            array = np.array(array)
        
        if len(array) >= length:
            return array[:length]
        
        pad_width = length - len(array)
        return np.pad(array, (0, pad_width), mode='constant', constant_values=0)
    
    @staticmethod
    def txt2arr(txt, start):
        """
        文字列を数値配列に変換
        """
        try:
            arr = []
            for c in txt.split():  # スペースで分割して処理
                if c in MyDataset.letters:  # 音素が存在するか確認
                    arr.append(MyDataset.letters.index(c) + start)
                else:
                    print(f"Warning: unknown phoneme '{c}'")
            return np.array(arr)
        except Exception as e:
            print(f"Error in txt2arr: {str(e)}")
            raise
        
    @staticmethod
    def arr2txt(arr, start):
        txt = []
        for n in arr:
            if(n >= start):
                txt.append(MyDataset.letters[n - start])     
        return ''.join(txt).strip()
    
    '''@staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= start):                
                if(len(txt) > 0 and txt[-1] == ' ' and MyDataset.letters[n - start] == ' '):
                    pass
                else:
                    txt.append(MyDataset.letters[n - start])                
            pre = n
        return ''.join(txt).strip()'''
    
    @staticmethod
    def ctc_arr2txt(arr, start): #いったんお試し用
        txt = []
        for n in arr:
            if n >= start:  # start以上のラベルのみを処理
                # 直前と同じラベルでも出力するために条件を変更
                txt.append(MyDataset.letters[n - start])  # ラベルをテキストに追加
        return ''.join(txt).strip()  # 最後に結合して前後の空白を削除して返す

            
    @staticmethod
    def wer(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return wer
        
    @staticmethod
    def cer(predict, truth):      
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return cer