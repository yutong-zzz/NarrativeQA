import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
import time
import math
import logging
import os
from itertools import ifilter
from similarity.rouge_l import Rouge
from nltk.translate.bleu_score import bleu
from similarity.meteor import Meteor

use_cuda = torch.cuda.is_available()
SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 1

class SeqModel(object):
    def __init__(self, vocab, params, args):
        #path config
        self.model_dir = args.model_dir

        # logging
        self.logger = logging.getLogger("NarrativeQA")

        #hyper params
        self.learning_rate = args.learning_rate
        self.optim_type = args.optim
        self.batch_size = args.batch_size

        #params
        self.embed_size = params['embed_size']
        self.hidden_size = params['hidden_size']
        self.output_size = vocab.size()
        self.max_length = params['max_a_len']

        #embedding
        self.embeddings = torch.FloatTensor(vocab.embeddings).cuda()
        #network
        self.encoder = EncoderRNN(self.embeddings, self.embed_size, self.hidden_size).cuda()
        self.decoder = DecoderRNN(self.embeddings, self.embed_size, self.hidden_size, self.output_size).cuda()

        if not args.train:
            self.restore(save_dir=args.model_dir, save_prefix=args.algo)
        else:
            self.logger.info('Model inits from {}, with prefix {}'.format(args.model_dir, args.algo))

    def _train_batch(self, input_tensor, target_tensor, encoder_optimizer, decoder_optimizer, criterion):
        batch_size = input_tensor.size(0)
        encoder_hidden = self.encoder.initHidden(batch_size)

        input_tensor = Variable(input_tensor.transpose(0, 1))
        target_tensor = Variable(target_tensor.transpose(0, 1))

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        encoder_outputs = Variable(torch.zeros(input_length, batch_size, self.encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], batch_size, encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]

        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, batch_size, decoder_hidden)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, batch_size, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)
                decoder_input = Variable(topi.view(-1))
                # decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input

                loss += criterion(decoder_output, target_tensor[di])

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def train(self, train_loader, valid_loader, epochs, save_dir, save_prefix, print_every=100):
        start = time.time()
        max_rouge_l = 0
        print_loss_total = 0  # Reset every print_every
        encoder_parameters = ifilter(lambda p: p.requires_grad, self.encoder.parameters())
        decoder_parameters = ifilter(lambda p: p.requires_grad, self.decoder.parameters())
        if self.optim_type == 'adagrad':
            encoder_optimizer = optim.Adagrad(encoder_parameters, lr=self.learning_rate)
            decoder_optimizer = optim.Adagrad(decoder_parameters, lr=self.learning_rate)
        elif self.optim_type == 'adam':
            encoder_optimizer = optim.Adam(encoder_parameters, lr=self.learning_rate)
            decoder_optimizer = optim.Adam(decoder_parameters, lr=self.learning_rate)
        elif self.optim_type == 'rprop':
            encoder_optimizer = optim.Rprop(encoder_parameters, lr=self.learning_rate)
            decoder_optimizer = optim.Rprop(decoder_parameters, lr=self.learning_rate)
        elif self.optim_type == 'sgd':
            # for name, param in self.encoder.named_parameters():
            #     if not param.requires_grad:
            #         print name, param.data
            encoder_optimizer = optim.SGD(encoder_parameters, lr=self.learning_rate)
            decoder_optimizer = optim.SGD(decoder_parameters, lr=self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))

        criterion = nn.NLLLoss()

        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            for iter, (batch_x,batch_y) in enumerate(train_loader):
                # batch_x = batch_x.cuda()
                # batch_y = batch_y.cuda()
                train_loss = self._train_batch(batch_x, batch_y, encoder_optimizer, decoder_optimizer, criterion)
                print_loss_total += train_loss
                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print 'epoch:%d, step:%d, time:%s, train-loss:%.4f' % (epoch, iter, timeSince(start), print_loss_avg)
            self.logger.info('Evaluating the model after epoch {}'.format(epoch))
            eval_loss, valid_scores_map = self.evaluate(valid_loader)
            self.logger.info('Dev eval loss {}'.format(eval_loss))
            self.logger.info('Dev eval Rouge-L score: {}'.format(cur_rouge_l))

            if valid_scores_map['ROUGE-L'] > max_rouge_l:
                self.save(save_dir, save_prefix)
                max_rouge_l = valid_scores_map['ROUGE-L']
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, data_loader):
        loss = 0.0
        data_size = 0
        score = {'Bleu_1':0, 'Bleu_4':0, 'ROUGE_L':0, 'METEOR':0}
        r = Rouge()
        m = Meteor()
        criterion = nn.NLLLoss()
        for iter, (batch_x, batch_y) in enumerate(data_loader):
            batch_size = batch_x.size(0)
            encoder_hidden = self.encoder.initHidden(batch_size)

            batch_x = Variable(batch_x.transpose(0, 1))
            batch_y = Variable(batch_y.transpose(0, 1))

            input_length = batch_x.size(0)
            target_length = batch_y.size(0)

            data_size += batch_size

            output = torch.LongTensor(target_length, batch_size)

            encoder_outputs = torch.zeros(self.max_length, batch_size, self.encoder.hidden_size)
            encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(
                    batch_x[ei], batch_size, encoder_hidden)
                encoder_outputs[ei] = encoder_output[0]

            decoder_input = torch.LongTensor([SOS_token]* batch_size)
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            decoder_hidden = encoder_hidden

            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, batch_size, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)

                output[di] = topi.view(-1)
                decoder_input = topi.view(-1)
                loss += criterion(decoder_output, batch_y[di]).item()

            output = output.transpose(0, 1)  #(batch_Size, target_len)
            for di in range(output.size()[0]):
                ignore = [0, 1, 2]     # [SOS_token, EOS_token, PAD_token]
                sent = [str(word.item()) for word in output[di] if word not in ignore]
                y = [str(word.item()) for word in batch_y[di] if word not in ignore]
                score['ROUGE_L'] += r.calc_score([' '.join(sent)], [' '.join(y)])
                score['Bleu_1'] += bleu([y], sent, weights=[1.0])
                score['Bleu_4'] += bleu([y], sent, weights=[0.25,0.25,0.25,0.25])
                score['METEOR'] += m._score(" ".join(sent), [" ".join(y)])
        print 'data amount:%d' % data_size
        score['Bleu_1'] = score['Bleu_1'] / (target_length*data_size)
        score['Bleu_4'] = score['Bleu_4'] / (target_length*data_size)
        score['ROUGE_L'] = score['ROUGE_L'] / (target_length*data_size)
        score['METEOR'] = score['METEOR'] / (target_length*data_size)
        return loss / (target_length*data_size), score

    def save(self, save_dir, save_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        torch.save(self.encoder.state_dict(), os.path.join(save_dir, save_prefix + '_encoder.pt'))
        torch.save(self.decoder.state_dict(), os.path.join(save_dir, save_prefix + '_decoder.pt'))
        self.logger.info('Model saved in {}, with prefix {}.'.format(save_dir, save_prefix))

    def restore(self, save_dir, save_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.encoder.load_state_dict(torch.load(os.path.join(save_dir, save_prefix + '_encoder.pt')))
        self.decoder.load_state_dict(torch.load(os.path.join(save_dir, save_prefix + '_decoder.pt')))
        self.logger.info('Model restored from {}, with prefix {}'.format(save_dir, save_prefix))


class EncoderRNN(nn.Module):
    def __init__(self, embeddings, embed_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(embeddings).cuda()
        self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(embed_size, hidden_size)

    def forward(self, input, batch_size, hidden):
        embedded = self.embedding(input).view(1, batch_size, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class DecoderRNN(nn.Module):

    def __init__(self, embeddings, embed_size, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(embeddings).cuda()
        self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, batch_size, hidden):
        output = self.embedding(input).view(1, batch_size, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (asMinutes(s))
