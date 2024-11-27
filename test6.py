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
from tensorboardX import SummaryWriter
import heapq

'''
----------------------------------------
音素についての制限を加える
'''
# 音素のリスト
letters = ['I', 'N', 'U', 'a', 'b', 'by', 'ch', 'cl', 
               'd', 'dy', 'e', 'f', 'fy', 'g', 'gw', 'gy', 
               'h', 'hy', 'i', 'j', 'k', 'kw', 'ky', 'm', 
               'my', 'n', 'ny', 'o', 'p', 'py', 'r', 
               'ry', 's', 'sh', 't', 'ts', 'ty', 'u', 
               'v', 'w', 'y', 'z', 'sil', 'pau'
]  # pauとsil取り除いた

# 音素制約を定義
vowels = [4, 19, 38, 11, 28] # 'a', 'i', 'u', 'e', 'o'
restricted_after_vowels = [1, 3] # 'I', 'U'
restricted_after_cl = [11, 19, 24, 26, 28, 38, 4, 5, 41] # 'e', 'i', 'm', 'n', 'o', 'u', 'a', 'b', 'y'
ch_sh_vowels = [4, 19, 38, 11, 28, 1, 3] # 'a', 'i', 'u', 'e', 'o', 'I', 'U'
fy_vowel = [38] # 'u'
ty_vowels = [4, 38, 28] # a, u, o
w_vowels = [4, 19, 11, 28] # a, i, e, o
youon_vowels = [4, 38, 11, 28] # a, u, e, o
# 特定の子音グループ
consonant_followed_by_vowel = [5, 9, 14, 15, 17, 20, 22, 24, 26, 29, 31, 35, 39, 42] # 'b', 'd', 'g', 'gw', 'h', 'j', 'k', 'kw', 'm', 'n', 'r', 't', 'v', 'z'
special_consonants = [6, 10, 16, 18, 23, 25, 27, 30, 32, 41] # 'by', 'dy', 'gy', 'hy', 'ky', 'my', 'ny', 'py', 'ry', 'y'
ch_sh = [7, 34, 12, 21, 29, 33, 36] # ch, sh, f, k, p, s, ts

# 音素制約チェック関数
def is_valid_phoneme_sequence(sequence):
    if sequence == None:
        return True
    last_arr = (len(sequence) - 2)
    current_phoneme = sequence[last_arr]
    next_phoneme = sequence[last_arr + 1]
    
    # ルール1: 子音の後には必ず母音
    if current_phoneme in consonant_followed_by_vowel and next_phoneme not in vowels:
        return False
    
    # ルール2: 特定の子音の後には特定の母音
    if current_phoneme in special_consonants and next_phoneme not in youon_vowels:
        return False
    
    # ルール3: ch, sh の後には特定の母音
    if current_phoneme in ch_sh and next_phoneme not in ch_sh_vowels:
        return False
    
    # ルール4: cl の後に制限された音素
    if current_phoneme == 8 and next_phoneme in restricted_after_cl:
        return False
    
    # ルール5: I, U の後には母音は来ない
    if current_phoneme in restricted_after_vowels and next_phoneme in vowels:
        return False
    
    # ルール6: fy の後は u のみ
    if current_phoneme == 13 and next_phoneme != 38:
        return False
    
    # ルール7: j の後には I, U は来ない
    if current_phoneme == 20 and next_phoneme in restricted_after_vowels:
        return False
    
    # ルール8: ty の後には特定の母音
    if current_phoneme == 37 and next_phoneme not in ty_vowels:
        return False
    
    # ルール9: w の後には特定の母音
    if current_phoneme == 40 and next_phoneme not in w_vowels:
        return False

    return True
'''
音素についての制限関数定義終了
-----------------------------------------------
'''
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


def beam_search_decoder(y, beam_width=5, blank_token=0, sil_token=43):
    """
    音素の制約を考慮したビームサーチアルゴリズム
    :param y: モデルの出力 (max_seq_len, batch_size, num_classes)
    :param beam_width: ビームサーチの幅
    :param blank_token: ブランクトークンのインデックス
    :param sil_token: 'sil'トークンのインデックス
    :return: デコードされたシーケンスのリスト
    """
    seq_len, batch_size, num_classes = y.shape

    # 初期化
    beams = [[(0, [], False)] for _ in range(batch_size)]  # 各バッチの初期ビーム [(スコア, シーケンス, sil使用済みフラグ)]
    results = []

    for t in range(seq_len):  # 時間ステップごとにループ
        new_beams = [[] for _ in range(batch_size)]  # 各バッチごとの新しいビーム候補
        
        for i in range(batch_size):  # バッチごとに処理
            candidates = []
            
            # 現在の時刻における出力確率
            probs = y[t, i].cpu().detach().numpy()  # 形状: (num_classes,)
            
            # 各ビームに対して遷移を計算
            for score, seq, sil_used in beams[i]:
                for idx in range(num_classes):
                    # 'sil'が既に使用済みの場合、再度選択しない
                    if idx == sil_token and sil_used:
                        continue
                    
                    # 同じ音素が連続するのを防ぐためのフィルタリング
                    if len(seq) > 0 and idx == seq[-1]:
                        continue
                    
                    # シーケンスを仮に更新して制約をチェック
                    tentative_seq = seq + [idx]
                    
                    # ブランクトークンは制約チェックをスキップ
                    if idx != blank_token and not is_valid_phoneme_sequence(tentative_seq):
                        continue
                    
                    # スコアを計算
                    new_score = score + probs[idx]
                    
                    # シーケンスを更新
                    if idx == blank_token:
                        new_seq = seq
                        new_sil_used = sil_used
                    else:
                        new_seq = tentative_seq
                        new_sil_used = sil_used or (idx == sil_token)
                    
                    # 新しい候補を追加
                    candidates.append((new_score, new_seq, new_sil_used))
            
            # 上位beam_width個の候補を選択
            if any(sil_used for _, _, sil_used in beams[i]):
                # 'sil'が使用済みの場合、次に高い確率のトークンを選ぶ
                filtered_candidates = [c for c in candidates if len(c[1]) > 0 and c[1][-1] != sil_token]
                if filtered_candidates:
                    new_beams[i] = heapq.nlargest(beam_width, filtered_candidates, key=lambda x: x[0])
                else:
                    # 他の選択肢がない場合は候補をそのまま使う
                    new_beams[i] = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])
            else:
                # 'sil'が未使用の場合、通常の候補選択
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

def test(model, net):
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
    count = 0
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
                savename = '{}_test6_only_train_{}.pt'.format(opt.save_prefix, count)
                log("save the model")
                count += 1
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

