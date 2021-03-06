#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/9/28
from __future__ import absolute_import
import argparse
import torch
import sys
import os
from pprint import pprint

sys.path.append(os.getcwd() + '/..')

'''set the default gpu to use'''
# torch.cuda.set_device(0)

from cli.train import train
from cli.preprocess import preprocess


def main():
    parser = argparse.ArgumentParser(description = "Framework Code Base.")

    group1 = parser.add_argument_group("1.Basic options")
    # basis argument
    group1.add_argument("--mode", default = 1, type = int, help = "run mode, 0 for pre-process, 1 for train")

    group1.add_argument("--debug", default = False, type = bool, help = "is debug mode on or off")

    group1.add_argument("--train", default = True, type = bool, help = "train or not")

    group1.add_argument("--test", default = True, type = bool, help = "test or not")

    group1.add_argument("--ensemble", default = False, type = bool, help = "ensemble test or not")

    group1.add_argument("--log_file", default = None, type = str,
                        help = "which file to save the log, if None, use screen")

    group1.add_argument("--weight_path", default = "weights", help = "path to save all trained models")

    group1.add_argument("--args_file", default = None, type = str, help = "json file of current args, default is args.josn")

    group1.add_argument("--print_every_n", default = 10, type = int, help = "print performance every n steps")

    group1.add_argument("--save_val", default = False, type = bool, help = "whether save the validation prediction result.")

    group1.add_argument('--log_dir', default = '../logs/', type = str, help = 'tensorboard log dir, default is project_dir/logs/, If None. will not use it.')
    # data specific argument
    group2 = parser.add_argument_group("2.Data specific options")
    # noinspection PyUnresolvedReferences
    group2.add_argument("--embedding_file", default = "w2c.300d.txt", type = str, help = "pre-trained embedding file name")
    # group2.add_argument("--embedding_file", default = "../data/glove/glove.6B.100d.txt",
    #                     type = str, help = "pre-trained embedding file")

    group2.add_argument("--data_root", default = "../data/", help = "root path of the dataset")

    group2.add_argument("--tmp_dir", default = "../data/processed/", help = "dataset specific tmp folder, have been processed")

    group2.add_argument("--train_file", default = "train.txt", help = "train file name under data root")

    group2.add_argument("--valid_file", default = "dev.txt", help = "validation file name under data root")

    group2.add_argument("--test_file", default = "test.txt", help = "test file name under data root")

    group2.add_argument("--max_count", default = None, type = int, help = "read n lines of data file, if None, read all data")

    group2.add_argument("--word_file", default = "word2id.txt", type = str, help = "word2id file under data root")

    # train hyper-parameters
    group3 = parser.add_argument_group("3. train common hyper-parameters for model")

    group3.add_argument("--batch_size", default = 128, type = int, help = "batch size for train")

    group3.add_argument("--lr", default = 3e-3, type = float, help = "lr for model learning")

    group3.add_argument("--keep_prob", default = 0.5, type = float, help = "the keep prob")

    group3.add_argument("--optimizer", default = "ADAM", choices = ["SGD", "ADAM", "ADAD"], help = "optimize algorithms, SGD or Adam")

    group3.add_argument("--hidden_size", default = 100, type = int, help = "RNN hidden size")

    group3.add_argument("--embedding_dim", default = 200, type = int, help = "dimension of word embeddings")

    group3.add_argument("--grad_clipping", default = 0, type = int, help = "the threshold value of gradient clip")

    group3.add_argument("--num_epoches", default = 200, type = int, help = "max epoch iterations")

    group3.add_argument("--num_words", default = 20000, type = int, help = "max length of the sentences")

    group3.add_argument("--skip_top", default = 20, type = int, help = "max length of the sentences")

    # -----------------------------------------------------------------------------------------------------------
    group4 = parser.add_argument_group("4.model specific parameters")

    group4.add_argument('--dataset', default = 'imdb', type = str, help = 'dataset to choose.')

    args = parser.parse_args()

    pprint(vars(args), indent = 4)
    if args.mode == 1:
        train(args)
    elif args.mode == 0:
        preprocess(args)
    else:
        raise ValueError("no value of mode arg being : {}".format(str(args.mode)))


if __name__ == '__main__':
    main()
