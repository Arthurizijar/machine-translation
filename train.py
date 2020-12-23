# -*- coding:UTF-8 -*-
import time
import torch
import random
from config import *
from torch import optim, nn
from prepare_data import process_data, Lang
from utils import time_since, show_plot, batch2data
from base_model import EncoderRNN, DecoderRNN
from attention_model import AttnDecoderRNN, Attn
from nltk.translate import bleu_score

random.seed(1)
input_lang, output_lang, pairs = process_data(combine_commentary_path, language_dump_path, False)
teacher_forcing_ratio = 0.5


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence] + [EOS_token]


def indexes_from_pair(pair):
    input_tensor = indexes_from_sentence(input_lang, pair[0])
    target_tensor = indexes_from_sentence(output_lang, pair[1])
    return input_tensor, target_tensor


def maskNLLLoss(inp, target, mask):
    n_total = mask.sum()
    cross_entropy = -torch.log(torch.gather(input=inp, dim=1, index=target.view(-1, 1)))
    loss = cross_entropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, n_total.item()


# def tensor_from_sentence(lang, sentence):
#     indexes = indexes_from_sentence(lang, sentence)
#     indexes.append(EOS_token)
#     return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


# def tensors_from_pair(pair):
#     input_tensor = tensor_from_sentence(input_lang, pair[0])
#     target_tensor = tensor_from_sentence(output_lang, pair[1])
#     return input_tensor, target_tensor


def train(train_data, encoder, decoder, encoder_optimizer, decoder_optimizer):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_tensors, lengths, target_tensors, mask, max_target_len = train_data

    # Set device options
    input_tensors = input_tensors.to(device)
    target_tensors = target_tensors.to(device)
    mask = mask.to(device)

    loss = 0
    print_losses = []
    n_totals = 0
    # 前向算法通过编码器
    encoder_outputs, encoder_hidden = encoder(input_tensors, lengths)
    # 初始化解码器的输入，每个句子的输入都是SOS
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)
    # 初始化解码器的隐状态，每个句子的隐状态都是编码器的输出
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: 将目标作为下一个输入
        for di in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = target_tensors[di].view(1, -1)
            mask_loss, n_total = maskNLLLoss(decoder_output, target_tensors[di], mask[di])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total
    else:
        # 不使用 teacher forcing: 使用自己的预测作为下一个输入
        for di in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            # detach from history as input
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).to(device)
            decoder_input = decoder_input.to(device)
            mask_loss, n_total = maskNLLLoss(decoder_output, target_tensors[di], mask[di])
            # print(decoder_output)
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return sum(print_losses) / n_totals


def evaluate_model(list_of_hypotheses, list_of_references):
    return bleu_score.corpus_bleu(list_of_references, list_of_hypotheses, weights=(1.0, 0),
                                  smoothing_function=bleu_score.SmoothingFunction().method1)


def evaluate(encoder, decoder, n_evaluate, testing_pairs, max_length=MAX_LENGTH, score=True):
    list_of_hypothesis = []
    list_of_reference = []
    list_of_source = []
    n_batch = n_evaluate // batch_size
    with torch.no_grad():
        for iter in range(1, n_batch+1):
            batch_target_words = []
            batch_pred_words = []
            batch_source_words = []
            for i in range(batch_size):
                batch_pred_words.append([])
            testing_batch = testing_pairs[(iter - 1) * batch_size: iter * batch_size]
            testing_batch.sort(key=lambda x: len(x[0]), reverse=True)
            testing_data = batch2data(testing_batch, True)
            input_tensors, lengths, target_tensors, mask, max_target_len = testing_data
            for test_item in testing_batch:
                batch_target_words.append([[output_lang.index2word[index] for index in test_item[1]][:-1]])
                batch_source_words.append([input_lang.index2word[index] for index in test_item[0]][:-1])
            # 将必要文件放在device上
            input_tensors = input_tensors.to(device)
            # 前向算法通过编码器
            encoder_outputs, encoder_hidden = encoder(input_tensors, lengths)
            # 初始化解码器的输入，每个句子的输入都是SOS
            decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # 初始化解码器的隐状态，每个句子的隐状态都是编码器的输出
            decoder_hidden = encoder_hidden[:decoder.n_layers]
            for di in range(max_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).to(device)
                for i in range(batch_size):
                    index = topi[i][0].item()
                    if index == EOS_token or index == PAD_token:
                        continue
                    word = output_lang.index2word[index]
                    batch_pred_words[i].append(word)
            # print(batch_target_words[0], batch_pred_words[0])
            list_of_hypothesis += batch_pred_words
            list_of_reference += batch_target_words
            list_of_source += batch_source_words
            # print(len(list_of_reference), len(list_of_hypothesis))
    if score:
        score = evaluate_model(list_of_hypothesis, list_of_reference)
        return score
    else:
        return list_of_source, list_of_reference, list_of_hypothesis


def train_iters(encoder, decoder, n_iters, n_evaluate, mode, print_every=1000, plot_every=100, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    n_batch = n_iters // batch_size
    best_bleu = 0

    save_data = []

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    random.shuffle(pairs)
    training_samples = pairs[:n_iters]
    training_pairs = [indexes_from_pair(pair) for pair in training_samples]
    testing_samples = pairs[n_iters: n_iters + n_evaluate]
    testing_pairs = [indexes_from_pair(pair) for pair in testing_samples]

    print("Start Training...")
    for epoch in range(n_epoch):
        for iter in range(1, n_batch + 1):
            training_batch = training_pairs[(iter - 1) * batch_size: iter * batch_size]
            training_data = batch2data(training_batch)
            loss = train(training_data, encoder, decoder, encoder_optimizer, decoder_optimizer, mode)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d  %d%%) %.4f' % (
                    time_since(start, iter / n_batch), iter, iter / n_batch * 100, print_loss_avg))
                score = evaluate(encoder, decoder, n_evaluate, testing_pairs)
                print('%.4f' % score)
                save_data.append("{}\t{}".format(print_loss_avg, score))
                if mode == "base":
                    model_path = base_best_model_checkpoint
                elif mode == "attention":
                    model_path = attention_best_model_checkpoint
                else:
                    raise Exception("Error Mode")
                if score > best_bleu:
                    best_bleu = score
                    torch.save({
                        'encoder_state_dict': encoder.state_dict(),
                        'decoder_state_dict': decoder.state_dict(),
                        'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                        'decoder_optimizerB_state_dict': decoder_optimizer.state_dict(),
                    }, model_path)

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    show_plot(plot_losses)
    with open(log_path, "w") as f:
        f.write("\n".join(save_data))


if __name__ == "__main__":
    # mode = "attention"
    mode = "base"
    print("the mode is {}".format(mode))
    if mode == "base":
        encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        decoder = DecoderRNN(hidden_size, output_lang.n_words).to(device)
    elif mode == "attention":
        encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        decoder = AttnDecoderRNN('general', hidden_size, output_lang.n_words).to(device)
    else:
        raise Exception("Error Mode")
    train_iters(encoder, decoder, train_sample, test_sample, mode, print_every=30)
