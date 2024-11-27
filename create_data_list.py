import os
import random

def create_data_lists(output_lips_dir, train_ratio=0.8):
    """
    学習データとバリデーションデータのリストを作成する
    
    Args:
        output_lips_dir: 唇領域画像が保存されているディレクトリ
        train_ratio: 学習データの割合
    """
    # output_lipsディレクトリ内のすべての動画ディレクトリを取得
    all_videos = [d for d in os.listdir(output_lips_dir) 
                 if os.path.isdir(os.path.join(output_lips_dir, d))]
    
    # ランダムにシャッフル
    random.shuffle(all_videos)
    
    # 学習データとバリデーションデータに分割
    split_idx = int(len(all_videos) * train_ratio)
    train_videos = all_videos[:split_idx]
    val_videos = all_videos[split_idx:]
    
    # train_list.txtを作成
    with open('./unseen_train.txt', 'w') as f:
        for video in train_videos:
            f.write(f'{video}\n')
    
    # val_list.txtを作成
    with open('./unseen_val.txt', 'w') as f:
        for video in val_videos:
            f.write(f'{video}\n')
    
    print(f'Created train list with {len(train_videos)} videos')
    print(f'Created validation list with {len(val_videos)} videos')

if __name__ == '__main__':
    output_lips_dir = './extracted_lip'  # 唇領域画像が保存されているディレクトリ
    create_data_lists(output_lips_dir)
