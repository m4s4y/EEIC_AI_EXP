import editdistance
import numpy as np
def cer(predict, truth):      
    cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
    return cer

predicted_phonemes = [
    ['sil', 'a', 'a', 'k', 'y', 'a', 'h', 'y', 'r', 'e', 'u', 'pau', 'r', 'a', 'a', 'v', 'k', 'y', 't', 'n', 'y', 'm', 'y', 'e', 'o', 'a', 'o', 'u', 'k', 'y', 'o', 'pau', 'a', 'a', 'i', 'o', 'f', 'y', 'a'],
    ['sil', 'o', 'r', 'g', 'a', 'pau', 'a', 'h', 'y', 'e', 'N', 'u', 'u', 'h', 'y', 'k', 'y', 'a', 'o', 'u', 'f', 'y', 't', 'a', 'a', 'o', 'u', 'a', 'a', 'a', 'h', 'y', 't', 'r', 'h', 'y', 'h', 'y'],
    ['sil', 'o', 'a', 'a', 't', 't', 'f', 'y', 'u', 'g', 'w', 'k', 'w', 'a', 'u', 'v', 'I', 'a', 'U', 'h', 'pau', 'N', 'h', 'y', 'pau', 'h', 'y', 'r', 'k', 'y', 'h', 'y', 'f', 'y', 'a', 'u', 'r', 'y', 'o', 'a', 't'],
    ['sil', 'w', 'u', 'a', 'a', 'a', 'a', 'h', 'y', 'pau', 'a', 'v', 'f', 'y', 'u', 'k', 'w', 'r', 'u', 'f', 'y', 't', 'N', 'v', 't', 'a', 'a', 'u', 'a', 'c', 'h', 'f', 'y', 'r', 'u', 'pau'],
    ['sil', 'pau', 'u', 'o', 'u', 'a', 'a', 'pau', 'a', 'g', 'a', 'a', 'a', 'k', 'w', 'k', 'w', 'u', 'u', 'a', 'a', 'pau', 'a', 'u', 'a', 't', 'a', 'pau', 'pau', 'a', 'u', 'f', 'y'],
    ['sil', 'k', 'w', 'a', 'a', 'a', 'a', 'f', 'y', 'N', 'e', 't', 'a', 'a', 'pau', 'k', 'y', 'u', 'e', 'u', 'a', 'u', 'g', 'a', 'p', 'y', 'u', 'u', 'u', 't', 'a', 'h', 'y', 't', 'o', 'u', 'a'],
    ['sil', 'i', 'u', 'a', 'a', 'N', 'u', 'h', 'y', 'a', 'u', 'a', 'a', 'v', 'o', 't', 'u', 'a', 'u', 'g', 'U', 'e', 'c', 'h', 'o', 'pau', 'a', 'a', 'u', 'g', 'k', 'w', 'u', 'u'],
    ['sil', 'o', 'h', 'y', 'a', 'a', 'pau', 'a', 'a', 'u', 'f', 'y', 'a', 'a', 'a', 'k', 'y', 'a', 'u', 'u', 'u', 'u', 'p', 'y', 'a', 'o', 'o', 'u', 'u', 'a', 'u', 'u', 'a', 'pau', 'u', 't', 's'],
    ['sil', 'u', 'a', 'a', 'a', 'a', 'a', 'u', 'u', 'a', 'a', 'a', 'a', 'r', 'u', 'f', 'y', 'a', 'a', 'd', 'a', 'a', 'a', 'a', 'f', 'y', 'r', 'pau', 'o', 'v', 'o'],
    ['sil', 't', 'a', 'pau', 'a', 'N', 'f', 'y', 'u', 'e', 'u', 'a', 'a', 't', 'k', 'w', 'u', 't', 'u', 'a', 'u', 'f', 'y', 'a', 'p', 'y', 'a', 'a', 'a', 'a', 'o', 'u', 'o', 'i', 'e'],
]

predicted_phonemes = [
    ['sil', 'a', 'a', 'ky', 'a', 'hy', 'r', 'e', 'u', 'pau', 'r', 'a', 'a', 'v', 'ky', 't', 'ny', 'my', 'e', 'o', 'a', 'o', 'u', 'ky', 'o', 'pau', 'a', 'a', 'i', 'o', 'fy', 'a'],
    ['sil', 'o', 'r', 'g', 'a', 'pau', 'a', 'hy', 'e', 'N', 'u', 'u', 'hy', 'ky', 'a', 'o', 'u', 'fy', 't', 'a', 'a', 'o', 'u', 'a', 'a', 'a', 'hy', 't', 'r', 'hy', 'hy'],
    ['sil', 'o', 'a', 'a', 't', 't', 'fy', 'u', 'gw', 'kw', 'a', 'u', 'v', 'I', 'a', 'U', 'h', 'pau', 'N', 'hy', 'pau', 'hy', 'r', 'ky', 'hy', 'fy', 'a', 'u', 'ry', 'o', 'a', 't'],
    ['sil', 'w', 'u', 'a', 'a', 'a', 'a', 'hy', 'pau', 'a', 'v', 'fy', 'u', 'kw', 'r', 'u', 'fy', 't', 'N', 'v', 't', 'a', 'a', 'u', 'a', 'ch', 'fy', 'r', 'u', 'pau'],
    ['sil', 'pau', 'u', 'o', 'u', 'a', 'a', 'pau', 'a', 'g', 'a', 'a', 'a', 'kw', 'kw', 'u', 'u', 'a', 'a', 'pau', 'a', 'u', 'a', 't', 'a', 'pau', 'pau', 'a', 'u', 'fy'],
    ['sil', 'kw', 'a', 'a', 'a', 'a', 'fy', 'N', 'e', 't', 'a', 'a', 'pau', 'ky', 'u', 'e', 'u', 'a', 'u', 'g', 'a', 'py', 'u', 'u', 'u', 't', 'a', 'hy', 't', 'o', 'u', 'a'],
    ['sil', 'i', 'u', 'a', 'a', 'N', 'u', 'h', 'y', 'a', 'u', 'a', 'a', 'v', 'o', 't', 'u', 'a', 'u', 'g', 'U', 'e', 'ch', 'o', 'pau', 'a', 'a', 'u', 'g', 'kw', 'u', 'u'],
    ['sil', 'o', 'h', 'y', 'a', 'a', 'pau', 'a', 'a', 'u', 'fy', 'a', 'a', 'a', 'ky', 'a', 'u', 'u', 'u', 'u', 'py', 'a', 'o', 'o', 'u', 'u', 'a', 'u', 'u', 'a', 'pau', 'u', 't', 's'],
    ['sil', 'u', 'a', 'a', 'a', 'a', 'a', 'u', 'u', 'a', 'a', 'a', 'a', 'r', 'u', 'fy', 'a', 'a', 'd', 'a', 'a', 'a', 'a', 'fy', 'r', 'pau', 'o', 'v', 'o'],
    ['sil', 't', 'a', 'pau', 'a', 'N', 'fy', 'u', 'e', 'u', 'a', 'a', 't', 'kw', 'u', 't', 'u', 'a', 'u', 'fy', 'a', 'py', 'a', 'a', 'a', 'a', 'o', 'u', 'o', 'i', 'e'],
]

truth_phonemes = [
    ['sil', 't', 's', 'I', 'p', 'u', 'r', 'a', 'k', 'o', 'f', 'u', 'w', 'a', 'g', 'i'],
    ['e', 'N', 'p', 'u', 'u', 'k', 'i', 'o', 'm', 'i', 't', 'a', 'k', 'o', 't', 'o', 'n', 'a', 'i', 'z', 'o', 'sil'],
    ['a', 'sil'],
    ['t', 'u', 'u', 'g', 'u', 'c', 'l', 'd', 'o', 'n', 'i', 'j', 'o', 'r', 'y', 'o', 'k', 'u', 'o', 'a', 'o', 'g', 'u', 'n', 'o', 'j', 'a'],
    ['U', 'sil'],
    ['p', 'a', 'u', 'z', 'e', 'N', 'r', 'y', 'o', 'o', 'n', 'a', 'v', 'a', 's', 'u', 'd', 'e', 'v', 'a', 'N', 'n', 'o'],
    ['a', 'i', 'm', 'o', 'k', 'a', 'w', 'a', 'r', 'u', 't', 'o', 'p', 'a', 'u'],
    ['u', 'r', 'i', 'c', 'l', 't', 's', 'a', 'a', 'o', 'k', 'o', 'n', 'o', 'm', 'i', 'm', 'a', 's', 'U', 'n', 'a'],
    ['w', 'o', 'r', 'o', 'j', 'i', 'm', 'i', 'i', 'r', 'u', 'n', 'o', 'c', 'h', 'i', 'r', 'y', 'a', 'k', 'u', 'o', 'k', 'y', 'o', 'z', 'e', 't', 's'],
    ['s', 'h', 'I', 't', 'e', 'e', 'n', 'o', 'a', 'i', 'd', 'a', 'g', 'a', 'r', 'a', 'n', 'o', 'k', 'u', 'c', 'l', 't', 's', 'e', 'e', 'g', 'a']
]

predicted_phonemes_sec = [    
    ['u', 'o', 'U', 'sil', 'z', 'e', 'n', 'o', 'k', 'a', 'pau', 'u', 'b', 'o', 'g', 'u', 'pau', 'a', 'e', 'my', 'e', 'o', 'my', 'o', 'e', 'pau', 'e', 'U', 'pau', 'U', 'pau', 'u'],
    ['sil', 'gw', 'u', 'pau', 'gw', 'u', 'ts', 'o', 'd', 'e', 'u', 'a', 'u', 'e', 'j', 'e', 'b', 'e', 'j', 'u', 'o', 'a', 'pau', 'o', 'a', 'r', 'u', 'b', 'u', 'm', 'u', 'h'],
    ['sil', 'a', 'ky', 'o', 'u', 'I', 'fy', 'u', 'gw', 'a', 'by', 'a', 'b', 'u', 'n', 'u', 'e', 'o', 'ry', 'a', 'e', 'r', 'o', 'a', 'u', 'b', 'u', 'a', 'pau', 'u', 'U', 'pau'],
    ['sil', 'b', 'e', 'o', 'r', 'u', 'e', 'y', 'u', 'U', 'j', 'o', 'sh', 'u', 'b', 'e', 'u', 'U', 'v', 'e', 'u', 'b', 'o', 'a', 'b', 'a', 'b', 'o', 'b', 'e', 'o', 'a'],
    ['sil', 'u', 'by', 'e', 'b', 'o', 'e', 'u', 'o', 'b', 'u', 'e', 'o', 'u', 'i', 'e', 'a', 'j', 'e', 'gy', 'e', 'U', 'b', 'e', 'u', 'a', 'u', 'o', 'gy', 'e', 'U', 'sh'],
    ['sil', 'o', 'u', 'gy', 'e', 'v', 'u', 'e', 'k', 'e', 'u', 'a', 'n', 'u', 'a', 'm', 'u', 'i', 'a', 'u', 'e', 'p', 'a', 'o', 'r', 'e', 'a', 'r', 'o', 'm', 'e', 'pau'],
    ['sil', 'N', 'u', 'e', 'r', 'a', 'e', 'a', 'pau', 'e', 'j', 'e', 'gw', 'u', 'a', 'o', 'u', 'e', 'u', 'b', 'e', 'a', 'o', 'b', 'u', 'o', 'm', 'a', 'pau', 'e', 'U', 'r'],
    ['sil', 'u', 'ts', 'e', 'gy', 'e', 'g', 'u', 'pau', 'cl', 'r', 'e', 'u', 'e', 'm', 'e', 'u', 'y', 'o', 'b', 'e', 'f', 'u', 'by', 'u', 'o', 'e', 'u', 'b', 'a', 'o', 'j'],
    ['sil', 'b', 'u', 'a', 'e', 'v', 'e', 'u', 'm', 'e', 'j', 'e', 'm', 'u', 'n', 'u', 'pau', 'j', 'o', 'b', 'u', 'b', 'o', 'j', 'e', 'b', 'e', 'u', 'e', 'I', 'U', 'z'],
    ['sil', 'd', 'u', 'd', 'u', 'e', 'v', 'u', 'm', 'u', 'o', 'u', 'gw', 'u', 'm', 'e', 'b', 'a', 'u', 'b', 'e', 'a', 'm', 'a', 'e', 'm', 'a', 'b', 'e', 'i', 'd', 'a'],
]


c_error = cer(predicted_phonemes, truth_phonemes)
c_error_2 = cer(predicted_phonemes_sec, truth_phonemes)
print(np.mean(c_error))
print(np.mean(c_error_2))