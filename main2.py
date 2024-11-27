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
    
    # デバッグ用ログ
    # log(f"Shape of y after permute: {y.shape}")
    # log(f"First few values of y: {y[:, 0, :5]}")
    # log(f"txt: {txt}")
    # log(f"vid_len: {vid_len}")
    # log(f"txt_len: {txt_len}")
    
    # CTC損失の計算
    loss = criterion(y, txt, vid_len, txt_len)
    
    return loss


def dataset2dataloader(dataset, num_workers=0, shuffle=True): #元々 num_workers=opt.num_workers,
    return DataLoader(dataset,
        batch_size = opt.batch_size, 
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = False)

def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()  

def ctc_decode(y):
    result = []
    y = y.argmax(-1)
    # CTCデコードを正しく行うため、ブランクトークンを無視する
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


def debug_batch(input):
    """
    データの不整合と異常を確認するデバッグ関数。
    """
    vid = input.get('vid')  # ビデオデータ
    txt = input.get('txt')  # テキストデータ
    vid_len = input.get('vid_len')  # ビデオシーケンスの長さ
    txt_len = input.get('txt_len')  # テキストシーケンスの長さ

    # vid_len >= txt_len を確認
    if not torch.all(vid_len >= txt_len):
        print("Error: vid_len < txt_len detected!")
        for i in range(len(vid_len)):
            if vid_len[i] < txt_len[i]:
                print(f"Batch {i}: vid_len = {vid_len[i].item()}, txt_len = {txt_len[i].item()}")

    # vid_len または txt_len が 0 以下になっていないか確認
    if torch.any(vid_len <= 0) or torch.any(txt_len <= 0):
        print("Error: vid_len or txt_len <= 0 detected!")
        print("vid_len <= 0:", vid_len[vid_len <= 0])
        print("txt_len <= 0:", txt_len[txt_len <= 0])

def test(model, net):

    with torch.no_grad():
        dataset = MyDataset(opt.video_path,
            opt.anno_path,
            opt.val_list,
            opt.vid_padding,
            opt.txt_padding,
            'test')
            
        log('num_test_data:{}'.format(len(dataset.data)))  
        model.eval()
        loader = dataset2dataloader(dataset, shuffle=False)
        loss_list = []
        wer = []
        cer = []
        crit = nn.CTCLoss()
        tic = time.time()
        for (i_iter, input) in enumerate(loader):            
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()
            
            y = net(vid)
            
            # クロスエントロピー損失を計算
            loss = calculate_ctc_loss(y, txt, vid_len, txt_len).detach().cpu().numpy()
            loss_list.append(loss)
            pred_txt = ctc_decode(y)
            
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            wer.extend(MyDataset.wer(pred_txt, truth_txt)) 
            cer.extend(MyDataset.cer(pred_txt, truth_txt))              
            if(i_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(i_iter+1)
                eta = v * (len(loader)-i_iter) / 3600.0
                
                log(''.join(101*'-'))                
                log('{:<50}|{:>50}'.format('predict', 'truth'))
                log(''.join(101*'-'))                
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
                    log('{:<50}|{:>50}'.format(predict, truth))                
                log(''.join(101 *'-'))
                log('test_iter={},eta={},wer={},cer={}'.format(i_iter,eta,np.array(wer).mean(),np.array(cer).mean()))                
                log(''.join(101 *'-'))
                
        return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean())
    
# def train(model, net):
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
#             debug_batch(input)
#             model.train()
#             vid = input.get('vid').cuda()
#             txt = input.get('txt').cuda()
#             vid_len = input.get('vid_len').cuda()
#             txt_len = input.get('txt_len').cuda()
            
#             optimizer.zero_grad()
#             y = net(vid)
            
#             # 出力にlog_softmaxを適用して対数確率に変換
#             y = F.log_softmax(y, dim=-1)
            
#             # CTC損失を計算
#             loss = calculate_ctc_loss(y, txt, vid_len, txt_len)
            
#             # 損失にNaNやInfが含まれていないかチェック
#             if torch.isnan(loss).any() or torch.isinf(loss).any():
#                 print("Error: NaN or Inf detected in loss!")
#                 print(f"Loss: {loss}")
                
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 勾配クリッピング
            
#             if(opt.is_optimize):
#                 optimizer.step()
            
#             tot_iter = i_iter + epoch*len(loader)
            
#             # デコードして予測と正解を取得
#             pred_txt = ctc_decode(y)
#             truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
#             train_cer.extend(MyDataset.cer(pred_txt, truth_txt))
            
#             # 予測と正解をわかりやすく表示
#             if i_iter % opt.display == 0:
#                 log(''.join(101 * '-'))
#                 log('{:<50}|{:>50}'.format('Predicted Phonemes', 'Ground Truth Phonemes'))
#                 log(''.join(101 * '-'))
                
#                 for (predict, truth) in zip(pred_txt, truth_txt):
#                     # 一つの行に予測と正解の音素列を表示
#                     log('{:<50}|{:>50}'.format(' '.join(predict), ' '.join(truth)))
                    
#                 log(''.join(101 * '-'))
#                 log(f'Epoch {epoch}, Iter {i_iter}, Loss: {loss.item()}')
#             writer.add_scalar('Loss/train', loss.item(), tot_iter)
#             writer.add_scalar('CER/train', np.array(train_cer).mean(), tot_iter)



# 長さの一致度を計算する関数
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

# 既存の訓練関数に変更を加える
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
            debug_batch(input)
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