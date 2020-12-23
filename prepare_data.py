# -*- coding:UTF-8 -*-
import random
import string
import jieba
import pickle
import os
from config import *


def read_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line[:-1]
            data.append(line)
    return data


def conbine_data(lang1_path, lang2_path, save_path):
    lang1_data = read_data(lang1_path)
    lang2_data = read_data(lang2_path)
    lang_conbine_data = zip(lang1_data, lang2_data)
    with open(save_path, "w", encoding="utf-8") as f:
        for j, k in lang_conbine_data:
            f.write("{}\t{}\n".format(j, k))


def filter_pair(p):
    return len(p[1].split(' ')) < EN_MAX_LENGTH and \
           len(p[0]) < CH_MAX_LENGTH


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def normalize(s):
    s = s.lower()
    s = s.rstrip()
    s = s.strip(string.punctuation)
    if len(s) != 0 and s[-1] in punc:
        s = s[:-1]
    s = s.replace(",", " ,")
    return s


def read_langs(data_path, reverse=False):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(data_path, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalize(s) for s in l.split('\t')] for l in lines]
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(target_language)
        output_lang = Lang(source_language)
    else:
        input_lang = Lang(source_language)
        output_lang = Lang(target_language)
    return input_lang, output_lang, pairs


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        if self.name == CH:
            words = list(jieba.cut(sentence))
        elif self.name == EN:
            words = sentence.split(' ')
        else:
            print("暂不支持该语言的分词！")
        for word in words:
            self.add_word(word)
        return words

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def process_data(data_path, dump_path, reverse=False):
    if os.path.exists(dump_path):
        pkl_file = open(dump_path, "rb")
        input_lang, output_lang, pairs = pickle.load(pkl_file)
    else:
        input_lang, output_lang, pairs = read_langs(data_path, reverse)
        print("Read %s sentence pairs" % len(pairs))
        pairs = filter_pairs(pairs)
        print("Counting words...")
        new_pairs = []
        for pair in pairs:
            input_words = input_lang.add_sentence(pair[0])
            output_words = output_lang.add_sentence(pair[1])
            new_pairs.append([input_words, output_words])
        pkl_file = open(dump_path, "wb")
        pickle.dump((input_lang, output_lang, new_pairs), pkl_file, protocol=3)
        pkl_file.close()
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


if __name__ == "__main__":
    # conbine_data(chinese_commentary_path, english_commentary_path, combine_commentary_path)
    input_lang, output_lang, pairs = process_data(combine_commentary_path, language_dump_path, False)
    print(random.choice(pairs))
