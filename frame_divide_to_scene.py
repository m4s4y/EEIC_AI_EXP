import os
import shutil
import glob
import numpy as np

# フレームレート（前のコードと同じ値を使用）
fps = 29.96
frame_duration = 10**7 / fps

def load_scene_info(scene_file):
    """シーンファイルからフレーム情報を読み込む"""
    scenes = []
    with open(scene_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if lines:
            # 最初と最後のタイムスタンプからフレーム範囲を計算
            start_time = int(lines[0].split()[0])
            end_time = int(lines[-1].split()[1])
            start_frame = int(start_time / frame_duration)
            end_frame = int(end_time / frame_duration)
            scenes = (start_frame, end_frame)
    return scenes

def save_frames_by_scene(scene_data, frame_dir, output_dir):
    """シーンごとのフレーム画像を整理する関数"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    start_frame, end_frame = scene_data
    
    # シーンの開始フレームと終了フレームに基づいて画像をコピー
    for frame_num in range(start_frame, end_frame + 1):
        frame_filename = f'frame_{str(frame_num).zfill(4)}.jpg'
        src_path = os.path.join(frame_dir, frame_filename)
        dest_path = os.path.join(output_dir, frame_filename)
        
        # ファイルが存在する場合にのみコピー
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
        else:
            print(f"Warning: {src_path} does not exist.")

def process_scenes():
    # ベースディレクトリの設定
    base_frame_dir = 'data/extracted_lip'
    base_output_dir = 'data/scenes_lip_pre'
    scenes_list_dir = 'data/scenes_list_pre'

    # シーンファイルを再帰的に検索
    scene_files = glob.glob(os.path.join(scenes_list_dir, '**', '*.txt'), recursive=True)
    total_files = len(scene_files)
    print(f"Found {total_files} scene files to process")

    for i, scene_file in enumerate(scene_files, 1):
        try:
            # シーンファイルのパスから対応する入力・出力ディレクトリを特定
            rel_path = os.path.relpath(scene_file, scenes_list_dir)
            parent_dir = os.path.dirname(rel_path)
            scene_name = os.path.basename(scene_file).replace('.txt', '')

            # `ROHAN4600_****` 形式から `LFROI_ROHAN4600_****` に変更
            frame_subdir = f"LFROI_{parent_dir}"
            
            # 対応するフレームディレクトリと出力ディレクトリのパスを構築
            frame_dir = os.path.join(base_frame_dir, frame_subdir)
            output_dir = os.path.join(base_output_dir, parent_dir, scene_name)
            
            print(f"\nProcessing {i}/{total_files}: {scene_file}")
            print(f"Frame directory: {frame_dir}")
            print(f"Output directory: {output_dir}")

            # シーン情報を読み込む
            scene_data = load_scene_info(scene_file)
            if scene_data:
                # シーンごとにフレーム画像を保存
                save_frames_by_scene(scene_data, frame_dir, output_dir)
                print(f"Successfully processed scene: {scene_name}")
                print(f"Frames {scene_data[0]} to {scene_data[1]} copied")
            else:
                print(f"Warning: No valid scene data found in {scene_file}")

        except Exception as e:
            print(f"Error processing {scene_file}: {str(e)}")
            continue

if __name__ == "__main__":
    process_scenes()
