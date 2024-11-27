# 音素のリスト
letters = ['I', 'N', 'U', 'a', 'b', 'by', 'ch', 'cl', 
               'd', 'dy', 'e', 'f', 'fy', 'g', 'gw', 'gy', 
               'h', 'hy', 'i', 'j', 'k', 'kw', 'ky', 'm', 
               'my', 'n', 'ny', 'o', 'p', 'py', 'r', 
               'ry', 's', 'sh', 't', 'ts', 'ty', 'u', 
               'v', 'w', 'y', 'z', 'sil', 'pau'
]  # pauとsil取り除いた

# 音素からインデックスへの変換関数
def phonemes_to_indices(phonemes, letters):
    return [letters.index(p)+1 for p in phonemes if p in letters]

# 音素制約を定義
vowels = phonemes_to_indices(['a', 'i', 'u', 'e', 'o'], letters)
restricted_after_vowels = phonemes_to_indices(['I', 'U'], letters)
restricted_after_cl = phonemes_to_indices(['e', 'i', 'm', 'n', 'o', 'u', 'a', 'b', 'y'], letters)
ch_sh_vowels = phonemes_to_indices(['a', 'i', 'u', 'e', 'o', 'I', 'U'], letters)
fy_vowel = phonemes_to_indices(['u'], letters)
ty_vowels = phonemes_to_indices(['a', 'u', 'o'], letters)
w_vowels = phonemes_to_indices(['a', 'i', 'e', 'o'], letters)

# 特定の子音グループをインデックスに変換
consonant_followed_by_vowel = phonemes_to_indices(['b', 'd', 'f', 'g', 'gw', 'h', 'k', 'kw', 'm', 'n', 'p', 'r', 's', 't', 'ts', 'v', 'z'], letters)
special_consonants = phonemes_to_indices(['by', 'dy', 'fy', 'gy', 'hy', 'ky', 'my', 'ny', 'py', 'ry', 'y'], letters)

# 結果の表示
print("vowels:", vowels)
print("restricted_after_vowels:", restricted_after_vowels)
print("restricted_after_cl:", restricted_after_cl)
print("ch_sh_vowels:", ch_sh_vowels)
print("fy_vowel:", fy_vowel)
print("ty_vowels:", ty_vowels)
print("w_vowels:", w_vowels)
print("consonant_followed_by_vowel:", consonant_followed_by_vowel)
print("special_consonants:", special_consonants)
