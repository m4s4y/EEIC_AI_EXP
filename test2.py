# いろいろ値変えたりしてみた

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

def ctc_decode(y):
    result = []
    y = y.argmax(-1)
    decoded_sequences = []
    for sequence in y:
        decoded = []
        prev = -1
        for token in sequence:
            if token != prev and token != 0:  # 0はブランクトークンとして使用
                decoded.append(token.item())
            prev = token
        decoded_sequences.append(decoded)
    
    return [MyDataset.ctc_arr2txt(decoded_sequences[_], start=1) for _ in range(len(decoded_sequences))]

def train(model, net):
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
            
            # デコードして予測と正解を取得
            pred_txt = ctc_decode(y)
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_cer.extend(MyDataset.cer(pred_txt, truth_txt))
            
            # 予測と正解をわかりやすく表示
            if i_iter % opt.display == 0:
                log(''.join(101 * '-'))
                log('{:<50}|{:>50}'.format('Predicted Phonemes', 'Ground Truth Phonemes'))
                log(''.join(101 * '-'))
                
                for (predict, truth) in zip(pred_txt, truth_txt):
                    log('{:<50}|{:>50}'.format(' '.join(predict), ' '.join(truth)))
                    
                log(''.join(101 * '-'))
                log(f'Epoch {epoch}, Iter {i_iter}, Loss: {loss.item()}')
            writer.add_scalar('Loss/train', loss.item(), tot_iter)
            writer.add_scalar('CER/train', np.array(train_cer).mean(), tot_iter)

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
