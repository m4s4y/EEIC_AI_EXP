# ビームサーチのプロトタイプ。あんまよくなかった

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
from torch.utils.data import DataLoader
# from model import LipNet
from model_attention import LipNet
from dataset import MyDataset
from tensorboardX import SummaryWriter

# 最小のresultxxx.txtを見つける関数
def find_available_result_file():
    for i in range(1000):
        filename = f"result{str(i).zfill(3)}.txt"
        if not os.path.exists(filename):
            return filename
    raise RuntimeError("No available resultxxx.txt slot found")

# ログファイルを決定
log_file = find_available_result_file()

# ログに書き込む関数
def log(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")

if(__name__ == '__main__'):
    log("hello1")
    opt = __import__('options')
    log("hello2")
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    log("hello3")    
    writer = SummaryWriter()
    log("hello4")

# 既存のカスタムクロスエントロピー損失を削除し、CTC損失に置き換える
criterion = nn.CTCLoss(blank=0)  # ここでの 'blank' は空白トークンのインデックス

def calculate_ctc_loss(y, txt, vid_len, txt_len):
    # CTCのため、入力を(batch_size, max_seq_len, num_classes) -> (max_seq_len, batch_size, num_classes)に転置
    y = y.permute(1, 0, 2)  # (max_seq_len, batch_size, num_classes)
    
    # CTC損失の計算
    loss = criterion(y, txt, vid_len, txt_len)
    
    return loss

def length_similarity(pred_len, true_len, max_len=200):
    """
    予測された音素列と正解の音素列の長さの一致度を計算する。
    :param pred_len: 予測されたシーケンスの長さ
    :param true_len: 正解のシーケンスの長さ
    :param max_len: 最大シーケンス長
    :return: 一致度（0から1の範囲で、1が完全一致）
    """
    length_diff = torch.abs(pred_len - true_len).float()
    reward = 1 - (length_diff / max_len)
    return reward

# CTC損失と長さの一致度を組み合わせる
def calculate_ctc_loss_with_length_penalty(y, txt, vid_len, txt_len, max_len=200):
    """
    CTC損失と長さの一致度に基づく報酬を加算する損失関数。
    :param y: モデルの出力 (max_seq_len, batch_size, num_classes)
    :param txt: 正解のラベル
    :param vid_len: ビデオシーケンスの長さ
    :param txt_len: 正解のテキストの長さ
    :param max_len: 最大シーケンス長
    :return: 結果の損失
    """
    criterion = nn.CTCLoss(blank=0)
    
    # CTC損失の計算
    y = y.permute(1, 0, 2)  # (max_seq_len, batch_size, num_classes)
    ctc_loss = criterion(y, txt, vid_len, txt_len)
    
    # 長さの一致度に基づく報酬を計算
    length_rewards = length_similarity(vid_len, txt_len, max_len)
    
    # 長さの一致度を損失に加える（逆の意味でペナルティ）
    loss_with_penalty = ctc_loss - torch.mean(length_rewards)
    
    return loss_with_penalty

def dataset2dataloader(dataset, num_workers=0, shuffle=True):
    return DataLoader(dataset,
        batch_size = opt.batch_size, 
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = False)

import heapq

# モデルの最終層のバイアスを加える
def adjust_output_for_blank_token(output, blank_bias=-10):
    # 出力テンソルをコピーして変更（非インプレース）
    adjusted_output = output.clone()
    adjusted_output[:, :, 0] += blank_bias
    return adjusted_output


# def beam_search_decoder(y, beam_width=5, blank_token=0):
#     """
#     ビームサーチアルゴリズムによるデコード
#     :param y: モデルの出力 (max_seq_len, batch_size, num_classes)
#     :param beam_width: ビームサーチの幅
#     :param blank_token: ブランクトークンのインデックス
#     :return: デコードされたシーケンスのリスト
#     """
#     seq_len, batch_size, num_classes = y.shape

#     # 最初のステップの初期化
#     beams = [(0, [])] * batch_size  # 各サンプルに対して(スコア, シーケンス)を格納
#     results = []

#     for t in range(seq_len):  # 時間ステップごとにループ
#         new_beams = []
        
#         for i in range(batch_size):  # バッチごとに処理
#             candidates = []
            
#             # 現在の時刻における出力確率
#             probs = y[t, i].cpu().numpy()  # 形状: (num_classes,)
            
#             # 各ビームに対して遷移を計算
#             for score, seq in beams[i]:
#                 for idx in range(num_classes):
#                     if idx == blank_token:  # ブランクトークンを無視
#                         new_score = score + probs[blank_token]
#                         new_seq = seq
#                     else:
#                         new_score = score + probs[idx]
#                         new_seq = seq + [idx]
                    
#                     candidates.append((new_score, new_seq))
            
#             # 上位beam_width個の候補を選択
#             new_beams.append(heapq.nlargest(beam_width, candidates, key=lambda x: x[0]))
        
#         beams = new_beams

#     # 最終的なビームサーチ結果を取得
#     for score, seq in beams[0]:
#         results.append(seq)

#     return results

def beam_search_decoder(y, beam_width=5, blank_token=0):
    """
    ビームサーチアルゴリズムによるデコード
    :param y: モデルの出力 (max_seq_len, batch_size, num_classes)
    :param beam_width: ビームサーチの幅
    :param blank_token: ブランクトークンのインデックス
    :return: デコードされたシーケンスのリスト
    """
    seq_len, batch_size, num_classes = y.shape

    # 初期化
    beams = [[(0, [])] for _ in range(batch_size)]  # 各バッチの初期ビーム [(スコア, シーケンス)]
    results = []

    for t in range(seq_len):  # 時間ステップごとにループ
        new_beams = [[] for _ in range(batch_size)]  # 各バッチごとの新しいビーム候補
        
        for i in range(batch_size):  # バッチごとに処理
            candidates = []
            
            # 現在の時刻における出力確率
            probs = y[t, i].cpu().detach().numpy()  # 形状: (num_classes,)
            
            # 各ビームに対して遷移を計算
            for score, seq in beams[i]:
                for idx in range(num_classes):
                    new_score = score + probs[idx]  # ブランクトークンも含めてスコアを計算
                    new_seq = seq if idx == blank_token else seq + [idx]
                    
                    candidates.append((new_score, new_seq))
            
            # 上位beam_width個の候補を選択
            new_beams[i] = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])
        
        # 新しいビームを次のステップに設定
        beams = new_beams

    # 最終的なビームサーチ結果を取得
    for i in range(batch_size):
        # 各バッチに対して最高スコアのシーケンスを取得
        best_beam = max(beams[i], key=lambda x: x[0])  # スコアが最大のシーケンスを選択
        results.append(best_beam[1])

    return [MyDataset.ctc_arr2txt(results[_], start=1) for _ in range(len(results))]


def ctc_decode_with_beam_search(y, beam_width=5):
    """
    ビームサーチを使用して、CTC出力をデコード
    :param y: モデルの出力 (max_seq_len, batch_size, num_classes)
    :param beam_width: ビームサーチの幅
    :return: デコードされたシーケンスのリスト
    """
    return beam_search_decoder(y, beam_width=beam_width)

def train(model, net):   # こいつだと、なんかエラー出る？音素列が文字列なんたらかんたら
    dataset = MyDataset(opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train')
        
    loader = dataset2dataloader(dataset) 
    optimizer = optim.Adam(model.parameters(),
                lr = opt.base_lr,
                weight_decay = 0.,
                amsgrad = True)
                
    log('num_train_data:{}'.format(len(dataset.data)))    
    tic = time.time()
    
    torch.cuda.empty_cache()
    train_cer = []
    for epoch in range(opt.max_epoch):
        for (i_iter, input) in enumerate(loader):
            # debug_batch(input)
            model.train()
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()
            
            optimizer.zero_grad()
            y = net(vid)
            
            # 出力にlog_softmaxを適用して対数確率に変換
            y = F.log_softmax(y, dim=-1)
            
            # ブランクトークンのスコアを低くする
            y = adjust_output_for_blank_token(y)
            
            # CTC損失と長さペナルティを組み合わせた損失を計算
            loss = calculate_ctc_loss_with_length_penalty(y, txt, vid_len, txt_len)
            
            # 損失にNaNやInfが含まれていないかチェック
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("Error: NaN or Inf detected in loss!")
                print(f"Loss: {loss}")
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 勾配クリッピング
            
            if(opt.is_optimize):
                optimizer.step()
            
            tot_iter = i_iter + epoch * len(loader)
            
            # ビームサーチでデコード
            pred_txt = ctc_decode_with_beam_search(y)
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_cer.extend(MyDataset.cer(pred_txt, truth_txt))
            
            # 予測と正解をわかりやすく表示
            if i_iter % opt.display == 0:
                log(''.join(101 * '-'))
                log('{:<50}|{:>50}'.format('Predicted Phonemes', 'Ground Truth Phonemes'))
                log(''.join(101 * '-'))
                
                for (predict, truth) in zip(pred_txt, truth_txt):
                    log('{:<50}|{:>50}'.format(' '.join(str(x) for x in predict), ' '.join(str(x) for x in truth)))
                    
                log(''.join(101 * '-'))
                log(f'Epoch {epoch}, Iter {i_iter}, Loss: {loss.item()}')

            writer.add_scalar('Loss/train', loss.item(), tot_iter)
            writer.add_scalar('CER/train', np.array(train_cer).mean(), tot_iter)

# def index_to_phoneme(indices):
#     """
#     インデックスのリストを音素に変換する関数
#     :param indices: 音素インデックスのリスト
#     :return: 音素のリスト
#     """
#     # MyDataset.letters に音素リストがあると仮定
#     phonemes = [MyDataset.letters[idx] for idx in indices if idx > 0]  # ブランクトークンを除外
#     return phonemes


# def train(model, net): # こいつは出力が数値列になってしまう。
#     dataset = MyDataset(opt.video_path,
#         opt.anno_path,
#         opt.train_list,
#         opt.vid_padding,
#         opt.txt_padding,
#         'train')
        
#     loader = dataset2dataloader(dataset) 
#     optimizer = optim.Adam(model.parameters(),
#                 lr = opt.base_lr,
#                 weight_decay = 0.,
#                 amsgrad = True)
                
#     log('num_train_data:{}'.format(len(dataset.data)))    
#     tic = time.time()
    
#     torch.cuda.empty_cache()
#     train_cer = []
#     for epoch in range(opt.max_epoch):
#         for (i_iter, input) in enumerate(loader):
#             # debug_batch(input)
#             model.train()
#             vid = input.get('vid').cuda()
#             txt = input.get('txt').cuda()
#             vid_len = input.get('vid_len').cuda()
#             txt_len = input.get('txt_len').cuda()
            
#             optimizer.zero_grad()
#             y = net(vid)
            
#             # 出力にlog_softmaxを適用して対数確率に変換
#             y = F.log_softmax(y, dim=-1)
            
#             # ブランクトークンのスコアを低くする
#             y = adjust_output_for_blank_token(y)
            
#             # CTC損失と長さペナルティを組み合わせた損失を計算
#             loss = calculate_ctc_loss_with_length_penalty(y, txt, vid_len, txt_len)
            
#             # 損失にNaNやInfが含まれていないかチェック
#             if torch.isnan(loss).any() or torch.isinf(loss).any():
#                 print("Error: NaN or Inf detected in loss!")
#                 print(f"Loss: {loss}")
                
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 勾配クリッピング
            
#             if(opt.is_optimize):
#                 optimizer.step()
            
#             tot_iter = i_iter + epoch * len(loader)
            
#             # ビームサーチでデコード
#             pred_indices = ctc_decode_with_beam_search(y)
            
#             # 数字列を音素列に変換
#             pred_txt = [index_to_phoneme(indices) for indices in pred_indices]
            
#             truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
#             train_cer.extend(MyDataset.cer(pred_txt, truth_txt))
            
#             # 予測と正解をわかりやすく表示
#             if i_iter % opt.display == 0:
#                 log(''.join(101 * '-'))
#                 log('{:<50}|{:>50}'.format('Predicted Phonemes', 'Ground Truth Phonemes'))
#                 log(''.join(101 * '-'))
                
#                 for (predict, truth) in zip(pred_txt, truth_txt):
#                     log('{:<50}|{:>50}'.format(' '.join(predict), ' '.join(truth)))
                    
#                 log(''.join(101 * '-'))
#                 log(f'Epoch {epoch}, Iter {i_iter}, Loss: {loss.item()}')

#             writer.add_scalar('Loss/train', loss.item(), tot_iter)
#             writer.add_scalar('CER/train', np.array(train_cer).mean(), tot_iter)


# main部分を変更し、ネットワークをトレーニング
if(__name__ == '__main__'):
    log("Loading options...")
    model = LipNet()
    model = model.cuda()
    net = nn.DataParallel(model).cuda()

    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        log('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        log('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    train(model, net)
