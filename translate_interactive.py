#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: set up an interactive test script

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

def nltk_tokenize(sentence):
    # nltk would turn " into ``, pretty weird behavior, spacy would not
    tokens = nltk.word_tokenize(sentence)
    return " ".join(tokens)

invalid_chars_re = re.compile(r'[^a-zA-Z ,\!\.\?0-9\'\$\%]')
repeat_punct_re = re.compile(r"(!|\.|,|\?|\$|\'){2,}")
punct_in_words = re.compile(r"([a-zA-Z])(!|\.|,|\?|\$)+([a-zA-Z])")
use_twitter_filter = True
def norm_text(text):
    if use_twitter_filter:
        text = invalid_chars_re.sub(' ', text)
        #text = re.sub(r'`', ' ', text)
        text = repeat_punct_re.sub(r"\1", text)
        text = punct_in_words.sub(r"\1\2 \3", text)
    norm = text.lower().strip()
    norm = " ".join([word for word in norm.split() if word])
    norm = nltk_tokenize(norm)
    #norm = spacy_tokenize(norm)
    return norm

def get_model_input(message, min_len=5, max_len=15, question_prob=0.1):
    #import pdb; pdb.set_trace()
    line = norm_text(message)
    input_words = line.strip().split()
    input_word_count = len(input_words)
    current_min_len = 1 #min_len
    current_max_len = 20 #min(5 * input_word_count, max_len)
    #if current_max_len < current_min_len:
    #    current_max_len = current_min_len
    len_cmd = "<#len{}#>".format(random.randint(current_min_len, current_max_len))
    output = len_cmd + " " + line
    if random.random() < question_prob:
        output = "<#question#> " + output
    return output

def a_very_stupid_first_implementation():
    tmp_src_file = 'tmp_src.txt'
    tmp_out_file = 'tmp_out.txt'

    if os.path.exists(tmp_out_file):
        os.remove(tmp_out_file)
    f_out_file = codecs.open(tmp_out_file, 'w', 'utf-8')
    translator = build_translator(opt, report_score=True, out_file=f_out_file)
    while True:
        message = input("User: ")
        model_input = get_model_input(message)
        print("model input: " + model_input)
        with open(tmp_src_file, 'w') as fw_src:
            fw_src.write(model_input)

        src_shards = split_corpus(tmp_src_file, opt.shard_size)
        tgt_shards = repeat(None)
        shard_pairs = zip(src_shards, tgt_shards)
        (src_shard, tgt_shard) = next(shard_pairs)
        translator.translate(
            src=src_shard,
            tgt=tgt_shard,
            src_dir="",
            batch_size=opt.batch_size,
            attn_debug=False
            )

        #with codecs.open(tmp_out_file, 'r', 'utf-8') as fr_out:
        #    all_out = fr_out.read()
        #    print(all_out)

        #os.remove(tmp_out_file)
    f_out_file.close()

def main(opt):
    ArgumentParser.validate_translate_opts(opt)
    #logger = init_logger(log_file=None)

    a_very_stupid_first_implementation()


def _get_parser():
    parser = ArgumentParser(description='translate_interactive.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
