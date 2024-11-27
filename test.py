import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
from torch.utils.data import DataLoader
from model_attention import LipNet
from dataset import MyDataset
from tensorboardX import SummaryWriter

# ブランクトークンの重み付きCTC損失を計算する関数
class WeightedCTCLoss(nn.Module):
    def __init__(self, blank_weight=0.5, blank_index=0):
        super(WeightedCTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_index)
        self.blank_weight = blank_weight
        self.blank_index = blank_index

    def forward(self, y, txt, vid_len, txt_len):
        # CTC損失の計算
        ctc_loss = self.ctc_loss(y, txt, vid_len, txt_len)
        
        # ブランクトークンに対する損失の重み調整
        blank_loss_mask = (txt != self.blank_index).float()
        weighted_loss = ctc_loss * (self.blank_weight + (1 - self.blank_weight) * blank_loss_mask.mean())
        
        return weighted_loss

criterion = WeightedCTCLoss(blank_weight=0.5)  # ここで損失関数を定義

# ビームサーチデコーディングを導入
def ctc_decode_with_beam_search(y, beam_width=3):
    batch_size = y.size(1)
    decoded_sequences = []

    for batch_idx in range(batch_size):
        sequence = y[:, batch_idx, :].cpu().detach().numpy()
        decoded = beam_search(sequence, beam_width=beam_width)
        decoded_sequences.append(decoded)
    
    return [MyDataset.ctc_arr2txt(decoded_sequences[_], start=1) for _ in range(len(decoded_sequences))]

def beam_search(log_probs, beam_width=3):
    max_seq_len, num_classes = log_probs.shape
    beams = [(0.0, [])]  # (累積スコア, シーケンス)のタプル

    for t in range(max_seq_len):
        all_candidates = []
        for score, seq in beams:
            for c in range(num_classes):
                new_score = score + log_probs[t, c]
                new_seq = seq + [c]
                all_candidates.append((new_score, new_seq))

        ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
        beams = ordered[:beam_width]
    
    return beams[0][1]

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


def calculate_ctc_loss(y, txt, vid_len, txt_len):
    # CTCのため、入力を(batch_size, max_seq_len, num_classes) -> (max_seq_len, batch_size, num_classes)に転置
    y = y.permute(1, 0, 2)  # (max_seq_len, batch_size, num_classes)
    
    # `vid_len` と `txt_len` の形状確認と修正
    if vid_len.dim() > 1:
        vid_len = vid_len.squeeze()
    if txt_len.dim() > 1:
        txt_len = txt_len.squeeze()
    
    # もしリスト形式ならテンソルに変換
    if isinstance(vid_len, list):
        vid_len = torch.tensor(vid_len).cuda()
    if isinstance(txt_len, list):
        txt_len = torch.tensor(txt_len).cuda()

    # CTC損失の計算
    loss = criterion(y, txt, vid_len, txt_len)
    
    return loss

def dataset2dataloader(dataset, num_workers=0, shuffle=True):
    return DataLoader(dataset,
        batch_size=opt.batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False)

def debug_batch(input):
    vid_len = input.get('vid_len')  # ビデオシーケンスの長さ
    txt_len = input.get('txt_len')  # テキストシーケンスの長さ

    # デバッグ情報の出力
    print(f"vid_len shape: {vid_len.shape}, expected: (batch_size,)")
    print(f"txt_len shape: {txt_len.shape}, expected: (batch_size,)")

    # vid_len >= txt_len を確認
    if not torch.all(vid_len >= txt_len):
        print("Error: vid_len < txt_len detected!")
        for i in range(len(vid_len)):
            if vid_len[i] < txt_len[i]:
                print(f"Batch {i}: vid_len = {vid_len[i].item()}, txt_len = {txt_len[i].item()}")

    if torch.any(vid_len <= 0) or torch.any(txt_len <= 0):
        print("Error: vid_len or txt_len <= 0 detected!")
        print("vid_len <= 0:", vid_len[vid_len <= 0])
        print("txt_len <= 0:", txt_len[txt_len <= 0])

# 訓練関数の修正
def train(model, net):
    dataset = MyDataset(opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train')
        
    loader = dataset2dataloader(dataset) 
    optimizer = optim.Adam(model.parameters(),
                lr=opt.base_lr,
                weight_decay=0.,
                amsgrad=True)
                
    log('num_train_data:{}'.format(len(dataset.data)))    
    tic = time.time()
    
    torch.cuda.empty_cache()
    train_cer = []
    criterion = WeightedCTCLoss(blank_weight=0.5)  # ブランクトークンの重みを0.5に設定
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
            
            # 重み付きCTC損失を計算
            loss = criterion(y, txt, vid_len, txt_len)
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("Error: NaN or Inf detected in loss!")
                print(f"Loss: {loss}")
                continue  # NaNやInfが検出された場合、ループをスキップ
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 勾配クリッピング
            
            if opt.is_optimize:
                optimizer.step()
            
            tot_iter = i_iter + epoch * len(loader)
            
            pred_txt = ctc_decode_with_beam_search(y)
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_cer.extend(MyDataset.cer(pred_txt, truth_txt))
            
            if i_iter % opt.display == 0:
                log(''.join(101 * '-'))
                log('{:<50}|{:>50}'.format('Predicted Phonemes', 'Ground Truth Phonemes'))
                log(''.join(101 * '-'))
                
                for (predict, truth) in zip(pred_txt, truth_txt):
                    log("{:<50}|{:>50}".format(predict, truth))
                    
                log('{}'.format(''.join(101 * '-')))
                log(f"[{epoch + 1}/{opt.max_epoch}] [{i_iter + 1}/{len(loader)}] Loss: {loss.item():.6f}, CER: {np.mean(train_cer):.6f}")

            # TensorBoardへの書き込み
            writer.add_scalar('Loss/train', loss.item(), tot_iter)
            writer.add_scalar('CER/train', np.mean(train_cer), tot_iter)

        # エポックごとにモデルの保存
        torch.save(model.state_dict(), os.path.join(opt.save_prefix, f"model_epoch_{epoch + 1}.pth"))

        log(f"Epoch {epoch + 1} completed in {time.time() - tic:.2f}s.")


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

# main部分を変更し、ネットワークをトレーニング
if(__name__ == '__main__'):
    log("Beam Search")
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