# -*- coding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('..')
import argparse
import logging
import os
import pickle
import torch.utils.data as Data
import torch

from dataset import Dataset
from vocab import Vocab
from config import *
from models.seq2seq import SeqModel
from utils import change_to_ids, pad_data
def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Machine Reading and Comprehension on the NarrativeQA')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='sgd',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=1,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['SEQ2SEQ', 'AS', 'SPAN'], default='SEQ2SEQ',
                                help='choose the algorithm to use')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--document_path', default='./data1',
                        help='document path')
    path_settings.add_argument('--vocab_dir', default='./data1/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='./saved_models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='./seq_baseline/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='./seq_baseline/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path', default='./seq_baseline/log.txt',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()


def run_seq(args):
    """
    Implements simple seq2seq baseline.
    return: all similarity scores
    """

    # logger
    logger = logging.getLogger("NarrativeQA")

    #params
    params = get_params('SEQ2SEQ')

    #build vocab
    if os.path.exists(os.path.join(args.vocab_dir, 'vocab.data')):
        logger.info('Loading vocab...')
        with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
            vocab = pickle.load(fin)
    else:
        logger.info('Building vocab...')
        vocab = Vocab(lower=True)

        logger.info('Assigning embeddings...')
        vocab.load_pretrained_embeddings(params['word2vec'])

        logger.info('Saving vocab...')
        with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
            pickle.dump(vocab, fout)

    # dataset
    data_set = Dataset(args.document_path, is_train=True)

    train_q,train_a = change_to_ids(data_set.train_set, vocab)
    valid_q,valid_a = change_to_ids(data_set.valid_set, vocab)
    test_q,test_a = change_to_ids(data_set.test_set, vocab)

    # padding
    train_q, train_q_lens, train_a, train_a_lens = pad_data(train_q, train_a, params)
    valid_q, valid_q_lens, valid_a, valid_a_lens = pad_data(valid_q, valid_a, params)
    test_q, test_q_lens, test_a, test_a_lens = pad_data(test_q, test_a, params)

    # transfrom data to torch-form
    valid_set = Data.TensorDataset(torch.cuda.LongTensor(valid_q), torch.cuda.LongTensor(valid_a))
    valid_loader = Data.DataLoader(dataset=valid_set, batch_size=args.batch_size,
                                  shuffle=True)
    test_set = Data.TensorDataset(torch.cuda.LongTensor(test_q), torch.cuda.LongTensor(test_a))
    test_loader = Data.DataLoader(dataset=test_set, batch_size=args.batch_size,
                                  shuffle=True)

    # init model
    model = SeqModel(vocab, params, args)

    # train
    if args.train:
        train_set = Data.TensorDataset(torch.cuda.LongTensor(train_q), torch.cuda.LongTensor(train_a))
        train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size,
                                       shuffle=True)
        model.train(train_loader, valid_loader, args.epochs, save_dir=args.model_dir,
                    save_prefix=args.algo)
        logger.info('Done with model training!')
    # evaluate
    _, valid_scores_map = model.evaluate(valid_loader)
    logger.info('Result on valid set: Bleu_1:{Bleu_1}, Bleu_4:{Bleu_4}, ROUGE_L:{ROUGE_L}, METEOR:{METEOR}'.format(
        **valid_scores_map))
    _, test_scores_map = model.evaluate(valid_loader)
    logger.info('Result on test set: Bleu_1:{Bleu_1}, Bleu_4:{Bleu_4}, ROUGE_L:{ROUGE_L}, METEOR:{METEOR}'.format(
        **test_scores_map))




def run_as(args, is_train=False):
    pass

def run_span(args, is_train=False):
    pass


def run():
    """
    Prepares and runs the whole system.
    """
    torch.cuda.manual_seed_all(22)
    args = parse_args()

    logger = logging.getLogger("NarrativeQA")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))


    if args.algo == 'SEQ2SEQ':
        run_seq(args)
    if args.algo == 'AS':
        run_as(args)
    if args.algo == 'SPAN':
        run_span(args)


if __name__ == '__main__':
    run()