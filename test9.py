'''
実装方針：
test5.pyの場合、silに関する制限をかけることはできたが、同じ母音音素を連続して出力しちゃう。
同じ音素の連続に関して無視する。基本は"sil" の時と同じ。
加えて、日本語の音素配列上の制約を入れることも試みる。
'''

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
from src.phonotac import is_valid_phoneme_sequence
from src.ctc_loss_revise import calculate_ctc_loss_with_length_penalty
from src.beam_search import beam_search_decoder
from tensorboardX import SummaryWriter

import heapq


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
    log(f"実行したファイル名: {__file__}")
    log("hello1")
    opt = __import__('options')
    log("hello2")
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    log("hello3")    
    writer = SummaryWriter()
    log("hello4")


def dataset2dataloader(dataset, num_workers=0, shuffle=True):
    return DataLoader(dataset,
        batch_size = opt.batch_size, 
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = False)


# モデルの最終層のバイアスを加える
def adjust_output_for_blank_token(output, blank_bias=-10):
    # 出力テンソルをコピーして変更（非インプレース）
    adjusted_output = output.clone()
    adjusted_output[:, :, 0] += blank_bias
    return adjusted_output

def ctc_decode_with_beam_search(y, beam_width=5):
    """
    ビームサーチを使用して、CTC出力をデコード
    :param y: モデルの出力 (max_seq_len, batch_size, num_classes)
    :param beam_width: ビームサーチの幅
    :return: デコードされたシーケンスのリスト
    """
    return beam_search_decoder(y, beam_width=beam_width)

def test(model, net):
    model.eval()
    with torch.no_grad():
        # テスト用データセットを用意
        dataset = MyDataset(opt.video_path,
                            opt.anno_path,
                            opt.val_list,
                            opt.vid_padding,
                            opt.txt_padding,
                            'test')
        
        log('num_test_data:{}'.format(len(dataset.data)))
        model.eval()  # モデルを評価モードに切り替え
        
        # データローダーを作成
        loader = dataset2dataloader(dataset, shuffle=False)
        loss_list = []
        wer = []
        cer = []
        
        # CTC損失関数の初期化
        crit = nn.CTCLoss(blank=0)  # デフォルトで0番目のインデックスをブランクトークンと仮定
        
        tic = time.time()
        
        for (i_iter, input) in enumerate(loader):
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()
            
            # モデルの出力
            y = net(vid)
            
            # 出力にlog_softmaxを適用して対数確率に変換
            y = F.log_softmax(y, dim=-1)
            
            # CTC損失の計算
            loss = crit(y.transpose(0, 1), txt, vid_len, txt_len).cpu().item()
            loss_list.append(loss)
            
            # ビームサーチで予測
            pred_txt = ctc_decode_with_beam_search(y)
            
            # 正解音素列と予測音素列の取得
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            
            # WERおよびCERの計算
            wer.extend(MyDataset.wer(pred_txt, truth_txt))
            cer.extend(MyDataset.cer(pred_txt, truth_txt))
            
            # 結果のログ出力
            if i_iter % opt.display == 0:
                v = 1.0 * (time.time() - tic) / (i_iter + 1)
                eta = v * (len(loader) - i_iter) / 3600.0
                
                log(''.join(101 * '-'))
                log('{:<50}|{:>50}'.format('Predicted Phonemes', 'Ground Truth Phonemes'))
                log(''.join(101 * '-'))
                
                # 予測と正解を表示（上位10件）
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
                    log('{:<50}|{:>50}'.format(' '.join(str(x) for x in predict), ' '.join(str(x) for x in truth)))
                    
                log(''.join(101 * '-'))
                log(f'Test Iter {i_iter}, ETA: {eta:.2f} hours, WER: {np.array(wer).mean()}, CER: {np.array(cer).mean()}')
                log(''.join(101 * '-'))
        
        # 損失、WER、CERの平均を返す
        return (
            np.array(loss_list).mean(),  # 平均損失
            np.array(wer).mean(),        # 平均WER
            np.array(cer).mean()         # 平均CER
        )

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
                log(f'CER {np.array(train_cer).mean()}')

            if(tot_iter % opt.test_step == 0):                
                (loss, wer, cer) = test(model, net)
                log('i_iter={},lr={},loss={},wer={},cer={}'
                    .format(tot_iter,show_lr(optimizer),loss,wer,cer))
                writer.add_scalar('val loss', loss, tot_iter)                    
                writer.add_scalar('wer', wer, tot_iter)
                writer.add_scalar('cer', cer, tot_iter)
                savename = '{}_loss_{}_wer_{}_cer_{}.pt'.format(opt.save_prefix, loss, wer, cer)
                (path, name) = os.path.split(savename)
                if(not os.path.exists(path)): os.makedirs(path)
                torch.save(model.state_dict(), savename)
                if(not opt.is_optimize):
                    exit()

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