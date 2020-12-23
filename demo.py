import torch
import random
from base_model import EncoderRNN, DecoderRNN
from attention_model import AttnDecoderRNN
from prepare_data import process_data, Lang
from train import evaluate, indexes_from_pair, input_lang, output_lang, pairs
from config import *

if __name__ == '__main__':
    mode = "attention"
    # mode = "base"
    if mode == "base":
        checkpoint = base_best_model_checkpoint
        encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        decoder = DecoderRNN(hidden_size, output_lang.n_words).to(device)
    elif mode == "attention":
        checkpoint = attention_best_model_checkpoint
        encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        decoder = AttnDecoderRNN('general', hidden_size, output_lang.n_words).to(device)
    else:
        raise Exception("Error Mode")
    checkpoint = torch.load(checkpoint)
    encoder_state_dict = checkpoint['encoder_state_dict']
    decoder_state_dict = checkpoint['decoder_state_dict']
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)
    print('模型已加载完毕！')

    encoder.eval()
    decoder.eval()

    # 初始化module模型
    n_sample = 100
    n_start = train_sample + test_sample
    random.shuffle(pairs)
    demo_samples = pairs[n_start: n_start + n_sample]
    demo_pairs = [indexes_from_pair(pair) for pair in demo_samples]
    # 沿着从长到短排序，不然无法将句子和翻译一一对应
    list_of_source, list_of_reference, list_of_hypothesis = \
        evaluate(encoder, decoder, n_sample, demo_pairs, max_length=MAX_LENGTH, score=False)
    for i in range(n_sample):
        print('中文句子{}> {}'.format(i+1, " ".join(list_of_source[i])))
        print('标准答案{}> {}'.format(i+1, " ".join(list_of_reference[i][0])))
        print('英文句子{}> {}'.format(i+1, " ".join(list_of_hypothesis[i])))
