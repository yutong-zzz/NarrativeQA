# -*- coding:utf8 -*-
import re
import string
import numpy as np

def loadGloVe(filename, embed_size):
    vocab = []
    embeddings = []
    vocab.append('unk')
    embeddings.append([0]*embed_size)
    with open(filename,'r') as fin:
        for line in fin.readlines():
            row = line.strip().split(' ')
            vocab.append(row[0])
            embeddings.append(row[1:])
    print 'Loaded GloVe!'
    return vocab, embeddings


def change_to_ids(data_set, vocab):
    """
    convert dataset to ids
    :param data_set: [q,a,a,file_id]
    :return: list(list(tokens))
    """
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    questions = []
    answers = []
    for item in data_set:
        item[0] = regex.sub(' ', item[0])  # remove punctuations
        item[1] = regex.sub(' ', item[1])
        item[2] = regex.sub(' ', item[2])
        q_ids = vocab.convert_to_ids(item[0].split())  # convert tokens to ids
        a1_ids = vocab.convert_to_ids(item[1].split())
        a2_ids = vocab.convert_to_ids(item[2].split())
        questions += [q_ids, q_ids]
        answers += [a1_ids, a2_ids]
    return questions, answers


def pad_data(q, a, params):
    """
    pad q and a to max_len   <blank>:idx = 2
    :param q: list of questions
    :param a: list of answers
    :return:
    """
    max_q_len, max_a_len = params['max_q_len'], params['max_a_len']

    q_seq_mask = [min(len(s), max_q_len) for s in q]
    a_seq_mask = [min(len(s), max_a_len) for s in a]

    padded_q = [s[:max_q_len] if len(s)>=max_q_len else s + [2]*(max_q_len-len(s)) for s in q]
    padded_a = [s[:max_a_len] if len(s)>=max_a_len else s + [2]*(max_a_len-len(s)) for s in a]

    return padded_q, q_seq_mask, padded_a, a_seq_mask
