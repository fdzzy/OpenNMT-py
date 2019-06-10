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
from translate_interactive import nltk_tokenize, norm_text, detokenize, prepare_model_input

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
