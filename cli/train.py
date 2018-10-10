#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/1
from __future__ import absolute_import
import torch
import time
import datetime
import numpy as np
from models.model1 import BaseLineRNN
from datasets.binary import *
from utils.util import logger, accuracy
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

USE_CUDA = torch.cuda.is_available()


def train(args):
    train_dataloader, test_dataloader, model = init_from_scrach(args)
    best_acc = 0.0
    best_epoch = 0
    iter = 0
    logger('Begin training...')

    # FIXME : could modified for your model
    if args.log_dir:
        logger_path = '../logs/log-av%s-%s-model%s-emb%d-id%s' % (
            args.activation, args.dataset, model.__class__.__name__, args.embedding_dim, str(datetime.datetime.now()))
        logger('Save log to %s' % logger_path)
        writer = SummaryWriter(log_dir = logger_path)
    for i in range(args.num_epoches):
        loss_sum = 0
        acc_sum = 0.0
        samples_num = 0
        for j, a_data in enumerate(train_dataloader):
            iter += 1  # recorded for tensorboard

            # forward and loss
            model.optimizer.zero_grad()
            model.zero_grad()
            out, feature = model(*a_data)  # model should return the output not only predict result.
            loss = model.loss(out, a_data[-1])

            # backward
            loss.backward()

            # grad clip if args.grad_clipping != 0
            if args.grad_clipping != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping)

            # optimize
            model.optimizer.step()

            # record
            loss_sum += loss.item()
            samples_num += len(a_data[0])
            acc_sum += accuracy(out = out.data.cpu().numpy(), label = a_data[-1])

            if (j + 1) % args.print_every_n == 0:
                logging.info('train: Epoch = %d | iter = %d/%d | ' %
                             (i, j, len(train_dataloader)) + 'loss sum = %.2f | accuracy : %.4f' % (
                                 loss_sum * 1.0 / j, acc_sum / samples_num))

                # for tensorboard
                if args.log_dir:
                    writer.add_scalar('loss', loss_sum / (j + 1), iter)
                    writer.add_scalar('accuracy', acc_sum / samples_num, iter)

                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(name, param.clone().cpu().data.numpy(), j)
                            writer.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), j)
        # Test
        logging.info("Testing...... | Model : {0} | Task : {1}".format(model.__class__.__name__, train_dataloader.dataset.__class__.__name__))
        testacc, _ = evaluation(args, model, test_dataloader)
        best_acc, best_epoch = testacc, i if best_acc < testacc else best_acc, best_epoch
        logging.error('Test result acc1: %.4f | best acc: %.4f | best epoch : %d' % (testacc, best_acc, best_epoch))


def evaluation(args, model, data_loader):
    model.eval()
    samples_num = 0
    acc_sum = 0.0

    pred = list()

    for j, a_data in enumerate(data_loader):
        out, _ = model(*a_data)
        pred.extend(out.data.cpu().numpy().max(-1)[1].tolist())
        samples_num += len(a_data[0])
        acc_sum += accuracy(out = out.data.cpu().numpy(), label = a_data[-1])
    model.train()
    acc, pred = acc_sum / samples_num, pred
    save_pred(args, pred, data_loader.dataset)
    return acc, pred


def save_pred(args, pred, data):
    """
    if you want to save the prediction, just implement it
    :param args:
    :param pred:
    :param data:
    :return:
    """
    pass


def init_from_scrach(args):
    """
    init the model and load the datasets
    :param args:
    :return:
    """
    logger('No trained model provided. init model from scratch...')

    logger('Load the train dataset...')
    if args.dataset.lower() == 'cr':
        train_dataset = CR(args, filename = args.train_file)
        valid_dataset = CR(args, filename = args.valid_file)
    else:
        raise ("No dataset named {}, please check".format(args.dataset.lower()))

    train_dataloader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = False,
                                  collate_fn = train_dataset.__class__.batchfy_fn, pin_memory = True, drop_last = False)
    logger('Train data max length : %d' % train_dataset.max_len)

    logger('Load the test dataset...')
    valid_dataloader = DataLoader(dataset = valid_dataset, batch_size = args.batch_size, shuffle = False,
                                  collate_fn = valid_dataset.__class__.batchfy_fn, pin_memory = True, drop_last = False)
    logger('Valid data max length : %d' % valid_dataset.max_len)

    logger('Initiating the model...')
    model = BaseLineRNN(args = args, hidden_size = args.hidden_size, embedding_size = args.embedding_dim, vocabulary_size = len(train_dataset.word2id),
                        rnn_layers = args.num_layers,
                        bidirection = args.bidirectional, num_class = train_dataset.num_class)

    if USE_CUDA:
        model.cuda()
    model.init_optimizer()
    logger('Model {} initiate over...'.format(model.__class__.__name__))
    logger(model)
    return train_dataloader, valid_dataloader, model
