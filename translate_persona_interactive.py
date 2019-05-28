#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: apply bpe

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
    words = text.split()
    cmds = []
    for i, word in enumerate(words):
        if word.startswith("<#") and word.endswith("#>"):
            cmds.append(word)
        else:
            break
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

def detokenize(input):
    output = input
    for item in ["'m", ".", "!", ",", "?", "'s", "'re", "n't", "'ve", "'ll", "'d"]:
        output = output.replace(" " + item, item)
    return output

def prepare_model_input(message, use_cmd=True, min_len=1, max_len=20, question_prob=0.1):
    cmds, line = norm_text(message)
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
        model_input = prepare_model_input(message)
        if opt.uid >= 0:
            model_input += "\t{}".format(opt.uid)
        print("Model input: {}".format(model_input))
        texts_to_translate = [model_input]
        scores, predictions = translator.translate(
            texts_to_translate, 
            batch_size=opt.batch_size,
            persona_model=True,
            persona_has_target=False
        )
        assert len(scores) == 1
        assert len(predictions) == 1
        scores, predictions = scores[0], predictions[0]
        for i, (score, prediction) in enumerate(zip(scores, predictions)):
            output = "{}: [{:.4f}] \"{}\"".format(i+1, score.item(), detokenize(prediction))
            if debug:
                output += " (debug: \"{}\")".format(prediction)
            print(output)

def main(opt):
    ArgumentParser.validate_translate_opts(opt)
    #logger = init_logger(log_file=None)

    run_model(opt)

def _get_parser():
    parser = ArgumentParser(description='translate_interactive.py')

    opts.config_opts(parser)
    opts.translate_persona_opts(parser, interactive=True)
    return parser

if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
