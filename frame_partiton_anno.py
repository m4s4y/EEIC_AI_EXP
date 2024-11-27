import os
import glob
import numpy as np

# フレームレート
fps = 29.96
frame_duration = 10**7 / fps  # ミリ秒からフレームに変換するためのスケール（1000ミリ秒を割る）

# モーラ情報（母音音素、pau）
vowels = ['a', 'i', 'u', 'e', 'o', 'I', 'U']
pau = 'pau'
sil = 'sil'

# ラベルデータをファイルパスから読み込む関数
def load_labels_from_file(file_path):
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            start = int(parts[0])  # ミリ秒単位
            end = int(parts[1])    # ミリ秒単位
            phoneme = parts[2]
            labels.append((start, end, phoneme))
    return labels

# def split_scenes(labels, min_length_frames, max_length_frames, overlap_ratio=0.5):
#     scenes = []
#     current_scene = []
#     scene_start = 0
#     accumulated_frames = 1

#     # ラベルのフレーム範囲を取得（ミリ秒をフレームに変換）
#     label_frames = [(int(start / frame_duration), int(end / frame_duration), phoneme) for start, end, phoneme in labels]

#     i = 0
#     while i < len(label_frames):
#         start_frame, end_frame, phoneme = label_frames[i]
#         current_scene.append((start_frame, end_frame, phoneme))
#         scene_length = end_frame - start_frame
#         accumulated_frames += scene_length

#         # シーンの終了位置が適切かチェック
#         if (min_length_frames <= accumulated_frames <= max_length_frames) or (i == len(label_frames) - 1):
#             # 最後の音素が母音、pau、silでない場合は次の音素まで延長
#             while current_scene and not (current_scene[-1][2] in vowels or current_scene[-1][2] in [pau, sil]):
#                 if i + 1 < len(label_frames):
#                     i += 1
#                     next_start_frame, next_end_frame, next_phoneme = label_frames[i]
#                     current_scene.append((next_start_frame, next_end_frame, next_phoneme))
#                     accumulated_frames += (next_end_frame - next_start_frame)
#                     end_frame = next_end_frame
#                 else:
#                     break  # 最後まで到達した場合は終了

#             # シーンが max_length_frames を超える場合、強制的に分割
#             if accumulated_frames > max_length_frames:
#                 # シーンを強制的に max_length_frames で切り分ける
#                 split_end = scene_start + max_length_frames
#                 scenes.append((scene_start, split_end))
                
#                 # 次のシーンの開始位置をオーバーラップ比率を考慮して計算
#                 scene_start = int(split_end - overlap_ratio * max_length_frames)
#                 accumulated_frames = accumulated_frames - (split_end - scene_start)
#             else:
#                 # 残ったシーンが適切な長さの場合、そのまま追加
#                 scenes.append((scene_start, end_frame))

#             # 次のシーンの開始位置を設定
#             scene_start = int(end_frame - overlap_ratio * (end_frame - scene_start))
#             accumulated_frames = 1
#             current_scene = []

#         # 最後の要素の場合、超えているなら強制的に打ち切る
#         if i == len(label_frames) - 1 and accumulated_frames > max_length_frames:
#             # シーンを強制的に max_length_frames で切り分ける
#             split_end = scene_start + max_length_frames
#             scenes.append((scene_start, split_end))

#         i += 1

#     return scenes
# def split_scenes(labels, min_length_frames, max_length_frames, overlap_ratio=0.5):
#     scenes = []
    
#     # ラベルのフレーム範囲を取得（ミリ秒をフレームに変換）
#     label_frames = [(int(start / frame_duration), int(end / frame_duration), phoneme) 
#                    for start, end, phoneme in labels]
    
#     current_pos = 0  # 現在の処理位置
    
#     while current_pos < len(label_frames):
#         scene_start_frame = label_frames[current_pos][0]
#         current_scene = []
#         accumulated_frames = 0
#         i = current_pos
        
#         # シーンを構築
#         while i < len(label_frames):
#             start_frame, end_frame, phoneme = label_frames[i]
#             frame_length = end_frame - start_frame
            
#             # シーンに追加した場合の長さをチェック
#             if accumulated_frames + frame_length > max_length_frames:
#                 # 最後の音素が母音かポーズでない場合、次の母音かポーズまで探索
#                 if not (phoneme in vowels or phoneme in [pau, sil]):
#                     next_vowel_idx = i
#                     while (next_vowel_idx < len(label_frames) and 
#                            not (label_frames[next_vowel_idx][2] in vowels or 
#                                 label_frames[next_vowel_idx][2] in [pau, sil])):
#                         next_vowel_idx += 1
                    
#                     # 近い位置に母音やポーズがある場合は、そこまで含める
#                     if (next_vowel_idx < len(label_frames) and 
#                             accumulated_frames + (label_frames[next_vowel_idx][1] - start_frame) 
#                             <= max_length_frames * 1.1):  # 10%の余裕を持たせる
#                         while i <= next_vowel_idx:
#                             current_scene.append(label_frames[i])
#                             accumulated_frames += label_frames[i][1] - label_frames[i][0]
#                             i += 1
#                 break
            
#             current_scene.append((start_frame, end_frame, phoneme))
#             accumulated_frames += frame_length
#             i += 1
            
#             # 最小長を超え、かつ母音またはポーズで終わる場合は分割可能
#             if (accumulated_frames >= min_length_frames and 
#                     (phoneme in vowels or phoneme in [pau, sil])):
#                 break
        
#         if current_scene:
#             scene_end_frame = current_scene[-1][1]
#             scenes.append((scene_start_frame, scene_end_frame))
            
#             # 次のシーンの開始位置を計算（オーバーラップを考慮）
#             overlap_frames = int((scene_end_frame - scene_start_frame) * overlap_ratio)
#             next_start_frame = scene_end_frame - overlap_frames
            
#             # 次の開始位置を探す
#             while (current_pos < len(label_frames) and 
#                    label_frames[current_pos][0] < next_start_frame):
#                 current_pos += 1
            
#             # 開始位置が見つからない場合は次のラベルから開始
#             if current_pos >= len(label_frames):
#                 break
#         else:
#             # シーンを構築できない場合は次のラベルへ
#             current_pos += 1
    
#     return scenes

def split_scenes(labels, min_length_frames, max_length_frames, overlap_ratio=0.5):
    scenes = []
    
    # ラベルのフレーム範囲を取得（ミリ秒をフレームに変換）
    label_frames = [(int(start / frame_duration), int(end / frame_duration), phoneme) 
                   for start, end, phoneme in labels]
    
    current_pos = 0
    
    while current_pos < len(label_frames):
        scene_start_frame = label_frames[current_pos][0]
        current_scene = []
        accumulated_frames = 0
        i = current_pos
        
        # シーンを構築
        while i < len(label_frames):
            start_frame, end_frame, phoneme = label_frames[i]
            frame_length = end_frame - start_frame
            
            # 1つのフレームが最大長を超える場合は、強制的に分割
            if frame_length > max_length_frames:
                if not current_scene:  # 現在のシーンが空の場合
                    scenes.append((start_frame, start_frame + max_length_frames))
                    current_pos = i + 1  # 次のフレームへ
                    break
            
            # 累積フレーム数が最大長を超える場合
            if accumulated_frames + frame_length > max_length_frames:
                if not current_scene:  # 現在のシーンが空の場合は、最低1つは追加
                    current_scene.append((start_frame, end_frame, phoneme))
                break
            
            current_scene.append((start_frame, end_frame, phoneme))
            accumulated_frames += frame_length
            i += 1
            
            # 最小長を超え、かつ適切な分割位置（母音またはポーズ）の場合
            if (accumulated_frames >= min_length_frames and 
                    (phoneme in vowels or phoneme in [pau, sil])):
                break
        
        if current_scene:
            scene_end_frame = current_scene[-1][1]
            scenes.append((scene_start_frame, scene_end_frame))
            
            # 次のシーンの開始位置を計算（オーバーラップを考慮）
            overlap_frames = int((scene_end_frame - scene_start_frame) * overlap_ratio)
            next_start_frame = max(scene_end_frame - overlap_frames, scene_start_frame)
            
            # 次の開始位置を探す
            next_pos = current_pos
            while (next_pos < len(label_frames) and 
                   label_frames[next_pos][0] < next_start_frame):
                next_pos += 1
            
            # 進捗がない場合は強制的に次に進む
            if next_pos <= current_pos:
                current_pos += 1
            else:
                current_pos = next_pos
        else:
            # シーンを構築できない場合は次のラベルへ
            current_pos += 1
        
        # 安全装置：最後の要素を処理したら確実に終了
        if current_pos >= len(label_frames):
            break
    
    return scenes

# シーンごとにラベルデータをファイルに保存する関数
def save_label_scenes_to_files(labels, scenes, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (scene_start, scene_end) in enumerate(scenes):
        # シーンの範囲内に収まるラベルを抽出
        scene_labels = [(start, end, phoneme) for start, end, phoneme in labels if scene_start * frame_duration <= start < scene_end * frame_duration]
        
        # シーンのラベルデータをファイルに保存
        scene_file_path = os.path.join(output_dir, f'scene_{idx + 1}.txt')
        with open(scene_file_path, 'w') as scene_file:
            for start, end, phoneme in scene_labels:
                scene_file.write(f"{start} {end} {phoneme}\n")
                
        print(f"Scene {idx + 1} saved to {scene_file_path}")

# data/annotation フォルダ内のすべての .lab ファイルを取得
lab_files = glob.glob('data/annotation/*.lab')

# 各ファイルについて処理
for file_path in lab_files:
    # ラベルデータをファイルパスから読み込む
    labels = load_labels_from_file(file_path)

    # シーン分割の実行
    min_scene_length_frames = 50
    max_scene_length_frames = 75
    scenes = split_scenes(labels, min_scene_length_frames, max_scene_length_frames)

    # 出力ディレクトリを設定（****に対応する部分を自動で生成）
    annotation_file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join('data/scenes_list_pre', annotation_file_name)

    # ラベルデータをシーンごとにファイルに保存
    save_label_scenes_to_files(labels, scenes, output_dir)
