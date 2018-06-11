import numpy as np
from collections import Counter
from scipy import spatial
"""
Input: cosine(list(words), list(words))
"""

def cosine(list1, list2):
    vocab = list(set(list1) | set(list2))
    list1_vec = np.zeros(len(vocab))
    list2_vec = np.zeros(len(vocab))
    c_list1 = Counter(list1)
    c_list2 = Counter(list2)
    for idx, w in enumerate(vocab):
        if w in c_list1.keys():
            list1_vec[idx] = c_list1[w]
        if w in c_list2.keys():
            list2_vec[idx] = c_list2[w]
    score = spatial.distance.cosine(list1_vec, list2_vec)

    return score