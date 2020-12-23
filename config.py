# -*- coding:UTF-8 -*-
import torch


chinese_commentary_path = "./data/train.zh"
english_commentary_path = "./data/train.en"
combine_commentary_path = "./data/train.txt"

language_dump_path = "./data/pairs.pkl"

fig_path = "./out/loss_function.png"
log_path = "./out/log_data.txt"

base_best_model_checkpoint = "./model/base.tar"
attention_best_model_checkpoint = "./model/attention.tar"

CH = "chinese"
EN = "english"
source_language = CH
target_language = EN
punc = ["。", "！", "？"]

CH_MAX_LENGTH = 10
EN_MAX_LENGTH = 10
MAX_LENGTH = max(CH_MAX_LENGTH, EN_MAX_LENGTH)
SOS_token = 0
EOS_token = 1
PAD_token = 2

train_sample = 20000
test_sample = 500
demo_sample = 10
hidden_size = 512
n_epoch = 30
batch_size = 100
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
