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
    音素バランスを考慮した改良版ビームサーチアルゴリズム
    """
    seq_len, batch_size, num_classes = y.shape
    
    # 母音と子音の識別
    vowels = [letters.index(v) for v in ['a', 'i', 'u', 'e', 'o', 'I', 'U']]
    consonants = [i for i in range(num_classes) if i not in vowels + [blank_token, sil_token]]
    
    # 音素バランススコアの計算用パラメータ
    phoneme_weights = {
        'consonant': 1.2,  # 子音の重み
        'vowel': 1.0,     # 母音の重み
        'blank': 0.8,     # ブランクトークンの重み
        'sil': 0.9        # silトークンの重み
    }

    def calculate_phoneme_balance_score(sequence):
        """
        音素列のバランススコアを計算
        """
        if not sequence:
            return 1.0
            
        vowel_count = sum(1 for p in sequence if p in vowels)
        consonant_count = sum(1 for p in sequence if p in consonants)
        total_count = len(sequence)
        
        if total_count == 0:
            return 1.0
            
        # 理想的な比率からのずれをペナルティとして計算
        ideal_vowel_ratio = 0.4  # 理想的な母音の比率
        actual_vowel_ratio = vowel_count / total_count
        
        balance_score = 1.0 - abs(ideal_vowel_ratio - actual_vowel_ratio)
        return balance_score

    def get_token_weight(token_id):
        """
        トークンの種類に応じた重みを返す
        """
        if token_id in vowels:
            return phoneme_weights['vowel']
        elif token_id in consonants:
            return phoneme_weights['consonant']
        elif token_id == blank_token:
            return phoneme_weights['blank']
        elif token_id == sil_token:
            return phoneme_weights['sil']
        return 1.0

    # ビームの初期化
    beams = [[(0, [], False)] for _ in range(batch_size)]  # (スコア, シーケンス, sil使用済みフラグ)
    results = []

    for t in range(seq_len):
        new_beams = [[] for _ in range(batch_size)]
        
        for i in range(batch_size):
            candidates = []
            probs = y[t, i].cpu().detach().numpy()
            
            for score, seq, sil_used in beams[i]:
                for idx in range(num_classes):
                    # 既存の制約チェック
                    if idx == sil_token and sil_used:
                        continue
                    if len(seq) > 0 and idx == seq[-1]:
                        continue
                    
                    # 音素配列の制約チェック
                    tentative_seq = seq + [idx]
                    if idx != blank_token and not is_valid_phoneme_sequence(tentative_seq):
                        continue
                    
                    # スコアの計算
                    token_weight = get_token_weight(idx)
                    balance_score = calculate_phoneme_balance_score(tentative_seq)
                    
                    # 新しいスコアは元のスコア、トークンの重み、バランススコアを組み合わせる
                    new_score = score + probs[idx] * token_weight * balance_score
                    
                    # シーケンスの更新
                    if idx == blank_token:
                        new_seq = seq
                        new_sil_used = sil_used
                    else:
                        new_seq = tentative_seq
                        new_sil_used = sil_used or (idx == sil_token)
                    
                    candidates.append((new_score, new_seq, new_sil_used))
            
            # 文法的制約と音素バランスを考慮して上位beam_width個の候補を選択
            filtered_candidates = []
            for candidate in candidates:
                score, seq, sil_used = candidate
                
                # 最低限の子音を含むように制約
                if len(seq) >= 3:  # ある程度の長さになったら
                    consonant_count = sum(1 for p in seq if p in consonants)
                    if consonant_count == 0:  # 子音が全くない場合は候補から除外
                        continue
                
                filtered_candidates.append(candidate)
            
            # 候補がない場合は元の候補をそのまま使用
            if not filtered_candidates:
                filtered_candidates = candidates
            
            new_beams[i] = heapq.nlargest(beam_width, filtered_candidates, key=lambda x: x[0])
        
        beams = new_beams

    # 最終結果の生成
    for i in range(batch_size):
        best_beam = max(beams[i], key=lambda x: x[0])
        results.append(best_beam[1])

    return [MyDataset.ctc_arr2txt(results[_], start=1) for _ in range(len(results))]

def calculate_ctc_loss_with_phoneme_balance(y, txt, vid_len, txt_len):
    """
    音素バランスを考慮したCTC損失の計算
    """
    # 基本のCTC損失
    y = y.permute(1, 0, 2)
    ctc_loss = criterion(y, txt, vid_len, txt_len)
    
    # 母音と子音の分布を計算
    probs = torch.softmax(y, dim=-1)
    vowel_indices = torch.tensor([letters.index(v) for v in ['a', 'i', 'u', 'e', 'o', 'I', 'U']]).cuda()
    consonant_indices = torch.tensor([i for i in range(len(letters)) 
                                    if letters[i] not in ['a', 'i', 'u', 'e', 'o', 'I', 'U', 'sil', 'pau']]).cuda()
    
    # 母音と子音の確率の合計
    vowel_probs = torch.sum(probs[:, :, vowel_indices], dim=-1)
    consonant_probs = torch.sum(probs[:, :, consonant_indices], dim=-1)
    
    # 分布バランスのペナルティ
    balance_penalty = torch.mean(torch.abs(vowel_probs - consonant_probs))
    
    # 最終的な損失
    final_loss = ctc_loss + 0.1 * balance_penalty
    
    return final_loss
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
    
    # エポックごとのCER統計を保存するリスト
    epoch_cer_stats = []
    
    for epoch in range(opt.max_epoch):
        # エポックごとのCERを記録
        epoch_cer = []
        epoch_losses = []
        
        for (i_iter, input) in enumerate(loader):
            model.train()
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()
            
            optimizer.zero_grad()
            y = net(vid)
            y = F.log_softmax(y, dim=-1)
            
            # 損失計算
            loss = calculate_ctc_loss_with_phoneme_balance(y, txt, vid_len, txt_len)
            epoch_losses.append(loss.item())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if(opt.is_optimize):
                optimizer.step()
            
            # ビームサーチでデコード
            pred_txt = beam_search_decoder(y)
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            
            # バッチごとのCERを計算
            batch_cer = MyDataset.cer(pred_txt, truth_txt)
            epoch_cer.extend(batch_cer)
            
            tot_iter = i_iter + epoch * len(loader)
            
            # 定期的な出力（display間隔ごと）
            if i_iter % opt.display == 0:
                # 現在のバッチのCER統計
                current_batch_cer = np.mean(batch_cer)
                
                log(''.join(101 * '-'))
                log('{:<50}|{:>50}'.format('Predicted Phonemes', 'Ground Truth Phonemes'))
                log(''.join(101 * '-'))
                
                for (predict, truth, cer_value) in zip(pred_txt, truth_txt, batch_cer):
                    log('{:<50}|{:>50}'.format(
                        ' '.join(str(x) for x in predict), 
                        ' '.join(str(x) for x in truth)
                    ))
                    log(f'Character Error Rate: {cer_value:.2%}')
                    
                log(''.join(101 * '-'))
                log(f'Epoch {epoch}, Iter {i_iter}')
                log(f'Loss: {loss.item():.4f}')
                log(f'Batch CER: {current_batch_cer:.2%}')
                log(f'Running Epoch CER: {np.mean(epoch_cer):.2%}')
                
                # TensorBoard用のログ
                writer.add_scalar('Loss/train', loss.item(), tot_iter)
                writer.add_scalar('CER/batch', current_batch_cer, tot_iter)
                writer.add_scalar('CER/running', np.mean(epoch_cer), tot_iter)
        
        # エポック終了時の統計
        epoch_avg_cer = np.mean(epoch_cer)
        epoch_avg_loss = np.mean(epoch_losses)
        epoch_cer_stats.append({
            'epoch': epoch,
            'avg_cer': epoch_avg_cer,
            'avg_loss': epoch_avg_loss
        })
        
        # エポック終了時のログ
        log(''.join(50 * '='))
        log(f'Epoch {epoch} Complete')
        log(f'Average Loss: {epoch_avg_loss:.4f}')
        log(f'Average CER: {epoch_avg_cer:.2%}')
        log(''.join(50 * '='))
        
        # エポックごとのCER統計をファイルに保存
        cer_log_file = f'cer_statistics_epoch_{epoch}.txt'
        with open(cer_log_file, 'w') as f:
            f.write(f'Epoch {epoch} CER Statistics\n')
            f.write(f'Average CER: {epoch_avg_cer:.2%}\n')
            f.write(f'Average Loss: {epoch_avg_loss:.4f}\n')
            f.write('\nDetailed CER values:\n')
            for i, cer_value in enumerate(epoch_cer):
                f.write(f'Sample {i}: {cer_value:.2%}\n')
        
        # TensorBoardにエポックの統計を記録
        writer.add_scalar('Epoch/CER', epoch_avg_cer, epoch)
        writer.add_scalar('Epoch/Loss', epoch_avg_loss, epoch)
        
        # 最良のモデルを保存（CERが最小のもの）
        if epoch == 0 or epoch_avg_cer < min(stat['avg_cer'] for stat in epoch_cer_stats[:-1]):
            torch.save(model.state_dict(), f'best_model_epoch_{epoch}_cer_{epoch_avg_cer:.4f}.pth')
            log(f'Saved best model at epoch {epoch} with CER {epoch_avg_cer:.2%}')

    # 学習終了時の総合統計
    log('\nTraining Complete')
    log('Final CER Statistics:')
    for stat in epoch_cer_stats:
        log(f"Epoch {stat['epoch']}: CER = {stat['avg_cer']:.2%}, Loss = {stat['avg_loss']:.4f}")
    
    # 最終的なCER統計をファイルに保存
    with open('final_cer_statistics.txt', 'w') as f:
        f.write('Final Training Statistics\n')
        f.write('========================\n')
        for stat in epoch_cer_stats:
            f.write(f"Epoch {stat['epoch']}:\n")
            f.write(f"  CER: {stat['avg_cer']:.2%}\n")
            f.write(f"  Loss: {stat['avg_loss']:.4f}\n")
            f.write('------------------------\n')

    return model, epoch_cer_stats
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
    trained_model, cer_stats = train(model, net)

