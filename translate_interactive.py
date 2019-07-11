#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser

import os
import re
import codecs
import nltk
import random
from tqdm import tqdm
from tools.apply_bpe import BPE

bpe_segmenter = BPE(codecs.open(os.path.join("available_models", "bpe-codes.txt"), encoding='utf-8'))

def nltk_tokenize(sentence):
    # nltk would turn " into ``, pretty weird behavior, spacy would not
    tokens = nltk.word_tokenize(sentence)
    return " ".join(tokens)

invalid_chars_re = re.compile(r'[^a-zA-Z ,\!\.\?0-9\'\$\%]')
repeat_punct_re = re.compile(r"(!|\.|,|\?|\$|\'){2,}")
punct_in_words = re.compile(r"([a-zA-Z])(!|\.|,|\?|\$)+([a-zA-Z])")
use_twitter_filter = True
def norm_text(text):
    words = text.split()
    cmds = []
    i = 0
    while i < len(words) and words[i].startswith("<#") and words[i].endswith("#>"):
        cmds.append(words[i])
        i += 1
    text = " ".join(words[i:])

    if use_twitter_filter:
        text = invalid_chars_re.sub(' ', text)
        #text = re.sub(r'`', ' ', text)
        text = repeat_punct_re.sub(r"\1", text)
        text = punct_in_words.sub(r"\1\2 \3", text)
    norm = text.lower().strip()
    norm = " ".join([word for word in norm.split() if word])
    norm = nltk_tokenize(norm)
    #norm = spacy_tokenize(norm)
    return cmds, norm

im_re = re.compile(r"(^|\b)i m(\b|$)")
ill_re = re.compile(r"(^|\b)i ll(\b|$)")
t_re = re.compile(r"(^|\b)(don|can) t(\b|$)")
its_re = re.compile(r"(^|\b)(it|what|he|she) s(\b|$)")
are_re = re.compile(r"(^|\b)(we|they|you) re(\b|$)")
#TODO: "i ve", "hasn t", "didn t", "i d"
def detokenize(input):
    output = input
    for item in ["'m", ".", "!", ",", "?", "'s", "'re", "n't", "'ve", "'ll", "'d"]:
        output = output.replace(" " + item, item)
    for pattern, repl in [(im_re, "i'm"), (t_re, r"\2't"), (its_re, r"\2's"), (ill_re, "i'll"), (are_re, r"\2're")]:
        output = pattern.sub(repl, output)
    output = output.replace("@@ ", "")
    return output

def detokenize_test():
    inputs = [
        "i can t catch a cold", "i m glad someone agrees with me", "i m not finna play with you",
        "i have a swim suit and it s super cute!", "lol i m serious tho", "i ll be seeing you in march then!",
        "they re nice", "what s that"
    ]
    for input in inputs:
        detok = detokenize(input)
        print("%s: %s" % (input, detok))

def prepare_model_input(message, use_cmd=True, min_len=1, max_len=20, question_prob=0.1):
    cmds, line = norm_text(message)
    line = bpe_segmenter.segment(line)
    if not use_cmd:
        return line.strip()
    if len(cmds) > 0:
        return " ".join(cmds) + " " + line.strip()
    
    len_cmd = "<#len{}#>".format(random.randint(min_len, max_len))
    output = len_cmd + " " + line
    if random.random() < question_prob:
        output = "<#question#> " + output
    return output

def run_model(opt, debug=False):
    translator = build_translator(opt, report_score=False, out_file=codecs.open(os.devnull, "w", "utf-8"))
    while True:
        message = input("User: ")
        model_input = prepare_model_input(message, max_len=15, question_prob=0)
        print("Model input: {}".format(model_input))
        texts_to_translate = [model_input]
        scores, predictions = translator.translate(
            texts_to_translate, 
            batch_size=opt.batch_size
        )
        assert len(scores) == 1
        assert len(predictions) == 1
        scores, predictions = scores[0], predictions[0]
        for i, (score, prediction) in enumerate(zip(scores, predictions)):
            output = "{}: [{:.4f}] \"{}\"".format(i+1, score.item(), detokenize(prediction))
            if debug:
                output += " (debug: \"{}\")".format(prediction)
            print(output)

def get_first_n_words(text, n):
    return " ".join(text.strip().split()[:n])

def run_model_batch(opt, debug=False):
    translator = build_translator(opt, report_score=False, out_file=codecs.open(os.devnull, "w", "utf-8"))
    with codecs.open('input.txt', 'r', 'utf-8') as reader, codecs.open('output.txt', 'w', 'utf-8') as writer:
        for line in reader:
            items = line.split('\t')
            if len(items) < 5:
                continue
            prev_q = items[2].strip()
            prev_a = items[3].strip()
            curr_q = items[4].strip()

            message = curr_q
            if len(message.split()) <= 1:
                message = (get_first_n_words(prev_q, 29) + " " + curr_q).strip()
            elif len(message.split()) > 30:
                message = get_first_n_words(message, 30)

            max_len = min(15, len(message.split()) * 5)
            model_input = prepare_model_input(message, max_len=max_len, question_prob=0)
            texts_to_translate = [model_input]
            scores, predictions = translator.translate(
                texts_to_translate, 
                batch_size=opt.batch_size
            )
            assert len(scores) == 1
            assert len(predictions) == 1
            scores, predictions = scores[0], predictions[0]
            prediction = detokenize(predictions[0])
            result = "%s\t%s\t%s\t%s\t%s\t%s\t%s" % (items[0], items[1], prev_q, prev_a, curr_q, prediction, model_input)
            print(result)
            writer.write(result + "\n")

def main(opt):
    ArgumentParser.validate_translate_opts(opt)
    #logger = init_logger(log_file=None)

    #run_model(opt)
    run_model_batch(opt)
    #detokenize_test()

def _get_parser():
    parser = ArgumentParser(description='translate_interactive.py')

    opts.config_opts(parser)
    opts.translate_opts(parser, interative=True)
    return parser

if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
