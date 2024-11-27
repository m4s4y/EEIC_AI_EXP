import torch
import torch.nn.functional as F
from model_attention import LipNet
from dataset_for_pred import SimpleDataset  # 変更済みのデータセットをインポート
from torch.utils.data import DataLoader
from src.beam_search import beam_search_decoder
import os

# モデルの最終層のバイアスを加える（オプション）
def adjust_output_for_blank_token(output, blank_bias=-10):
    adjusted_output = output.clone()
    adjusted_output[:, :, 0] += blank_bias
    return adjusted_output

# ビームサーチでデコード
def ctc_decode_with_beam_search(y, beam_width=5):
    return beam_search_decoder(y, beam_width=beam_width)

# 学習済みモデルをロードして予測する関数
def predict(model_path, video_path):
    # モデルのロード
    model = LipNet()
    model = model.cuda()
    net = torch.nn.DataParallel(model).cuda()
    
    # 学習済みパラメータをロード
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 評価モードに切り替え

    # テスト用データセットの準備
    dataset = SimpleDataset(video_path, vid_pad=75)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    predictions = []

    with torch.no_grad():
        for input in loader:
            vid = input.get('vid').cuda()
            
            # モデルの出力を取得
            y = net(vid)
            
            # 出力に log_softmax を適用
            y = F.log_softmax(y, dim=-1)
            
            # ブランクトークンのスコアを調整（オプション）
            y = adjust_output_for_blank_token(y)
            
            # ビームサーチで予測
            pred_txt = ctc_decode_with_beam_search(y)
            
            # 予測結果を保存
            predictions.extend(pred_txt)

    return predictions

# メイン処理
if __name__ == '__main__':
    # 学習済みモデルのパス
    model_path = 'weights/LipNet_unseen_test8_only_train_10.pt'
    
    # テストデータの設定（適宜変更してください）
    video_path = 'data/scenes_lip_pre/ROHAN4600_0001/scene_1'
    
    # 予測実行
    predictions = predict(model_path, video_path)
    
    # 予測結果の表示
    print('Predictions:')
    for pred in predictions:
        print(f'Predicted: {" ".join(str(x) for x in pred)}')
    
    # 予測結果をファイルに保存する場合
    output_file = 'predictions.txt'
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(f'Predicted: {" ".join(str(x) for x in pred)}\n')
    
    print(f'Predictions saved to {output_file}')
