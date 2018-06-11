#coding=utf-8
import csv
import os
import logging
import numpy as np
import nltk
from nltk.tokenize import WordPunctTokenizer
import pickle
import string
import glob
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import re

class Dataset(object):
    """
    APIs for NarrativeQA dataset.
    """
    def __init__(self, document_path, is_train=False):
        self.document_path = document_path
        self.train_set, self.valid_set, self.test_set = [], [], []
        self.logger = logging.getLogger("NarrativeQA")

        self._construct_dataset(is_train)

    def _construct_dataset(self, is_train):
        """
        Construct dataset lists: [q,a1,a2,file_id]
        Args:
            is_train: construct train_set when True
        """
        qa_reader = csv.reader(open('qaps.csv'))
        for lineidx, row in enumerate(qa_reader):
            if lineidx == 0:
                infos = row
                continue
            sample = dict(zip(infos, row))
            sample['question'] = sample['question']
            sample['answer1'] = sample['answer1']
            sample['answer2'] = sample['answer2']
            if is_train:
                if sample['set'] == 'train':
                    self.train_set += [[sample['question'], sample['answer1'], sample['answer2'], sample['document_id']]]
            if sample['set'] == 'valid':
                self.valid_set += [[sample['question'], sample['answer1'], sample['answer2'], sample['document_id']]]
            if sample['set'] == 'test':
                self.test_set += [[sample['question'], sample['answer1'], sample['answer2'], sample['document_id']]]

        self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))
        self.logger.info('Valid set size: {} questions.'.format(len(self.valid_set)))
        self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    def _select_paragraph(self, item, file_path):
        """
        Select relevant paragraphs from the document.
        :param item: current item. See _extract_gram
        :param file_path: document path
        :return: paragraphs:list[str]
        """
        def exists_in(nouns, line):
            for noun in nouns:
                if noun.lower() in line.lower():
                    return True
            return False

        with open(file_path, 'r') as fin:
            lines = fin.readlines()
        question = item[0]
        q_tokens = WordPunctTokenizer().tokenize(question)
        is_noun = lambda pos: pos[:2] == 'NN'
        nouns = [word for (word, pos) in nltk.pos_tag(q_tokens) if is_noun(pos)]
        paras = [line for line in lines if exists_in(nouns, line)]
        lines_len = sum([len(line) for line in lines])
        paras_len = sum([len(para) for para in paras])
        if paras_len == 0:
            print item[-1]
            print item[0]
            print nouns
        return paras

    def _extract_gram(self, cur_item, set_name):
        """
        Extract grams for IR model.
        Extract 8-gram;
                1 sentence;
                answer length
                spans from documents
        Args:
            cur_item: one data item: [q, a1, a2, file_id]
        Returns:
            [[gram formed document], q, a1, a2, file_id]
        """
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        file_id = cur_item[-1]
        data_path = os.path.join(self.document_path, set_name)
        # paras = self._select_paragraph(cur_item, os.path.join(data_path, file_id))
        with open(os.path.join(data_path, file_id), 'r') as fin:
            paras = fin.readlines()
        if self.candidate_type == 'gram-8':
            if os.path.exists(os.path.join(data_path, file_id + '.gram8')):
                try:
                    grams = pickle.load(open(os.path.join(data_path, file_id + '.gram8'), 'rb'))
                except EOFError:
                    print 'EOF: %s' %file_id
                    raise
            else:
                count = 0   # count of grams
                grams = []
                for line in paras:
                    p = re.compile(r'[^\x00-\x7f]')   # remove not ascii char
                    line = re.sub(p, '', line)
                    try:
                        sentences = tokenizer.tokenize(line.strip())
                    except UnicodeDecodeError:
                        print 'Unicode Error:%s'  % file_id
                        raise
                    for sent in sentences:
                        sent = regex.sub(' ', sent)  # remove punctuation
                        words = WordPunctTokenizer().tokenize(sent)
                        if len(words) < 8:   # omit sentence whose length < 8
                            continue
                        else:
                            for i in range(len(words)-7):
                                grams.append(words[i:i+8])
                                count += 1
                pickle.dump(grams, open(os.path.join(data_path, file_id + '.gram8'), 'wb'))
            print file_id
            print 'num of grams: %s' %len(grams)
            return [grams, cur_item[0], cur_item[1], cur_item[2], cur_item[3]]

        if self.candidate_type == 'sentence':
            if os.path.exists(os.path.join(data_path, file_id + '.sentence')):
                grams = pickle.load(open(os.path.join(data_path, file_id + '.sentence'), 'rb'))
            else:

                count = 0  # count of grams
                grams = []
                for line in paras:
                    p = re.compile(r'[^\x00-\x7f]')   # remove not ascii char
                    line = re.sub(p, '', line)
                    sentences = tokenizer.tokenize(line.strip())
                    for sent in sentences:
                        sent = regex.sub(' ', sent)   # remove punctuation
                        words = WordPunctTokenizer().tokenize(sent)
                        if len(words) > 0:  #not an empty sentence
                            grams.append(words)
                            count += 1
                pickle.dump(grams, open(os.path.join(data_path, file_id + '.sentence'), 'wb'))

            return [grams, cur_item[0], cur_item[1], cur_item[2], cur_item[3]]

        if self.candidate_type == 'answerlen':
            # TODO: for oracle IR Models
            if os.path.exists(os.path.join(data_path, file_id + '.answerlen')):
                pass

        else:
            raise Exception("Invalid candidate type!")

    def get_batches(self, set_name, batch_size, shuffle = True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches

        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'valid':
            data = self.valid_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield [self._extract_gram(data[i], set_name) for i in batch_indices]

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.valid_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'valid':
            data_set = self.valid_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if set_name is None:
            documents = glob.glob(os.path.join(self.document_path, 'train')) + \
                        glob.glob(os.path.join(self.document_path, 'valid')) + \
                        glob.glob(os.path.join(self.document_path, 'test'))
            for document in documents:
                with open(document) as fin:
                    lines = fin.readlines()
                for line in lines:
                    p = re.compile(r'[^\x00-\x7f]')  # remove not ascii char
                    line = re.sub(p, '', line)
                    words = WordPunctTokenizer().tokenize(line)
                    for token in words:
                        yield token
        else:
            if data_set is not None:
                for sample in data_set:
                    for token in sample[0].split(' '):
                        yield token
                    document_ids = [i[-1] for i in data_set]
                    for document_id in document_ids:
                        data_path = os.path.join(self.document_path, set_name)
                        # paras = self._select_paragraph(cur_item, os.path.join(data_path, file_id))
                        with open(os.path.join(data_path, document_id), 'r') as fin:
                            lines = fin.readlines()
                            for line in lines:
                                p = re.compile(r'[^\x00-\x7f]')  # remove not ascii char
                                line = re.sub(p, '', line)
                                words = WordPunctTokenizer().tokenize(line)
                                for token in words:
                                    yield token
