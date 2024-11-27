# 音素のリスト
letters = ['I', 'N', 'U', 'a', 'b', 'by', 'ch', 'cl', 
               'd', 'dy', 'e', 'f', 'fy', 'g', 'gw', 'gy', 
               'h', 'hy', 'i', 'j', 'k', 'kw', 'ky', 'm', 
               'my', 'n', 'ny', 'o', 'p', 'py', 'r', 
               'ry', 's', 'sh', 't', 'ts', 'ty', 'u', 
               'v', 'w', 'y', 'z', 'sil', 'pau'
        ]  # pauとsil取り除いた

# 音素制約を定義
vowels = ['a', 'i', 'u', 'e', 'o']
restricted_after_vowels = ['I', 'U']
restricted_after_cl = ['e', 'i', 'm', 'n', 'o', 'u', 'a', 'b', 'y']
ch_sh_vowels = ['a', 'i', 'u', 'e', 'o', 'I', 'U']
fy_vowel = ['u']
ty_vowels = ['a', 'u', 'o']
w_vowels = ['a', 'i', 'e', 'o']
'''
f, k, p はch, shグループ
fy は u のみ
h は I, a, e, i, o

'''
# 特定の子音グループ
consonant_followed_by_vowel = ['b', 'd', 'f', 'g', 'gw', 'h', 'j', 'k', 'kw', 'm', 'n', 'r', 't', 'ts', 'v', 'z']
special_consonants = ['by', 'dy', 'gy', 'hy', 'ky', 'my', 'ny', 'py', 'ry', 'y']
ch_sh = ['ch', 'sh', 'f', 'k', 'p', 's', 'ts']
# 音素制約チェック関数
def is_valid_phoneme_sequence(sequence):
    for i in range(len(sequence) - 1):
        current_phoneme = sequence[i]
        next_phoneme = sequence[i + 1]
        
        # ルール1: 子音の後には必ず母音
        if current_phoneme in consonant_followed_by_vowel and next_phoneme not in vowels:
            return False
        
        # ルール2: 特定の子音の後には特定の母音
        if current_phoneme in special_consonants and next_phoneme not in ['a', 'u', 'e', 'o']:
            return False
        
        # ルール3: ch, sh の後には特定の母音
        if current_phoneme in ['ch', 'sh'] and next_phoneme not in ch_sh_vowels:
            return False
        
        # ルール4: cl の後に制限された音素
        if current_phoneme == 'cl' and next_phoneme in restricted_after_cl:
            return False
        
        # ルール5: I, U の後には母音は来ない
        if current_phoneme in restricted_after_vowels and next_phoneme in vowels:
            return False
        
        # ルール6: fy の後は u のみ
        if current_phoneme == 'fy' and next_phoneme != 'u':
            return False
        
        # ルール7: j の後には I, U は来ない
        if current_phoneme == 'j' and next_phoneme in ['I', 'U']:
            return False
        
        # ルール8: ty の後には特定の母音
        if current_phoneme == 'ty' and next_phoneme not in ty_vowels:
            return False
        
        # ルール9: w の後には特定の母音
        if current_phoneme == 'w' and next_phoneme not in w_vowels:
            return False

    return True

# テスト用の音素列
phoneme_sequence = ['by', 'a', 'cl', 'k', 'e', 'ty', 'a', 'w', 'a']
print(is_valid_phoneme_sequence(phoneme_sequence))  # 正しい列なら True, 違反があれば False