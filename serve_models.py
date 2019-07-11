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
import sys
import copy
import re
import codecs
import nltk
import random
import traceback

from translate_interactive import detokenize
from tools.apply_bpe import BPE
import requests
import urllib.parse
import flask
from flask import Flask, request

twitter_retro_index_cmd_model = os.path.join("available_models", "twitter_tok_bpe_cmd_5_model_840k_to_848k_average.ppl.50.43.pt")
twitter_retro_index_no_cmd_model = os.path.join("available_models", "twitter_tok_bpe_transformer_big_step_average.pt")
convo_index_model = os.path.join("available_models", "index_bpe_cmd_transformer_big_step_average.pt")

def set_opt_params(opt, model_path):
    opt.models = [model_path]
    opt.replace_unk = True
    opt.report_time = False
    opt.n_best = 10
    opt.beam_size = 20
    opt.block_ngram_repeat = 1
    return opt

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

def prepare_model_input(message, use_cmd=True, min_len=1, max_len=20, question_prob=0.1):
    line = norm_text(message)
    line = bpe_segmenter.segment(line)

    if not line:
        return line

    if not use_cmd:
        return line.strip()
    
    len_cmd = "<#len{}#>".format(random.randint(min_len, max_len))
    output = len_cmd + " " + line
    if random.random() < question_prob:
        output = "<#question#> " + output
    return output

def run_model(opt, debug=False):
    translator = build_translator(opt, report_score=False, out_file=codecs.open(os.devnull, "w", "utf-8"))
    while True:
        message = input("User: ")
        model_input = prepare_model_input(message, question_prob=0)
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

def main(opt):
    ArgumentParser.validate_translate_opts(opt)
    run_model(opt)

def _get_parser():
    parser = ArgumentParser(description='translate_interactive.py')

    opts.config_opts(parser)
    #opts.translate_opts(parser, interative=True)
    opts.translate_opts(parser)
    return parser

app = Flask(__name__)
translator_twitter_retro_index_cmd = None
translator_twitter_retro_index_no_cmd = None
translator_convo_index_cmd = None
bpe_segmenter = None

def initialization():
    global translator_twitter_retro_index_cmd
    global translator_twitter_retro_index_no_cmd
    global translator_convo_index_cmd
    global bpe_segmenter

    print("Initializing BPE & models...")
    bpe_segmenter = BPE(codecs.open(os.path.join("available_models", "bpe-codes.txt"), encoding='utf-8'))

    parser = _get_parser()
    sys.argv += ['-model', 'dummy']
    sys.argv += ['-src', 'dummy']
    opt = parser.parse_args()

    opt1 = copy.deepcopy(opt)
    opt1 = set_opt_params(opt1, twitter_retro_index_cmd_model)
    translator_twitter_retro_index_cmd = build_translator(opt1, report_score=False, out_file=codecs.open(os.devnull, "w", "utf-8"))

    opt2 = copy.deepcopy(opt)
    opt2 = set_opt_params(opt2, twitter_retro_index_no_cmd_model)
    translator_twitter_retro_index_no_cmd = build_translator(opt2, report_score=False, out_file=codecs.open(os.devnull, "w", "utf-8"))

    opt3 = copy.deepcopy(opt)
    opt3 = set_opt_params(opt3, convo_index_model)
    translator_convo_index_cmd = build_translator(opt3, report_score=False, out_file=codecs.open(os.devnull, "w", "utf-8"))
    print("Initializing done!")

prod_generative_url = 'http://localhost:5001/gen?'

@app.route("/gen", methods=['GET'])
def generate_response():
    prev_q = request.args.get('prev_q', default='', type=str)
    prev_a = request.args.get('prev_a', default='', type=str)
    curr_q = request.args.get('q', type=str)
    data = {}

    try:
        print("prev_q: \"{}\", prev_a: \"{}\", curr_q: \"{}\"".format(prev_q, prev_a, curr_q))

        model_input_cmd = prepare_model_input(curr_q, use_cmd=True, question_prob=0)
        data["input4"] = model_input_cmd
        if not model_input_cmd:
            data["model4"] = "input is empty" 
        else:
            scores, predictions = translator_twitter_retro_index_cmd.translate([model_input_cmd], batch_size=1)
            if len(predictions) == 1 and len(predictions[0]) > 1 and predictions[0][0]:
                data["model4"] = detokenize(predictions[0][0])
        
        model_input_no_cmd = prepare_model_input(curr_q, use_cmd=False, question_prob=0)
        data["input3"] = model_input_no_cmd
        if not model_input_no_cmd:
            data["model3"] = "input is empty" 
        else:
            scores, predictions = translator_twitter_retro_index_no_cmd.translate([model_input_no_cmd], batch_size=1)
            if len(predictions) == 1 and len(predictions[0]) > 1 and predictions[0][0]:
                data["model3"] = detokenize(predictions[0][0])

        model_input_cmd = prepare_model_input(curr_q, use_cmd=True, question_prob=0)
        data["input1"] = model_input_cmd
        if not model_input_cmd:
            data["model1"] = "input is empty" 
        else:
            scores, predictions = translator_convo_index_cmd.translate([model_input_cmd], batch_size=1)
            if len(predictions) == 1 and len(predictions[0]) > 1 and predictions[0][0]:
                data["model1"] = detokenize(predictions[0][0])

        url = prod_generative_url + "q={}".format(urllib.parse.quote(curr_q))
        prev_q = prev_q.strip()
        prev_a = prev_a.strip()
        if prev_q and prev_a:
            url += "&prev_q={}&prev_a={}".format(urllib.parse.quote(prev_q), urllib.parse.quote(prev_a))
        print("sending request to: {}".format(url))
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        data["model2"] = result["response"]

        print("input1: \"{}\", input3: \"{}\", input4: \"{}\"".format(data["input1"], data["input3"], data["input4"]))

        data["success"] = True
    except Exception as e:
        data["excpetion"] = str(e)
        data["stack_trace"] = traceback.format_exc()
        data["success"] = False
    return flask.jsonify(data)

if __name__ == "__main__":
    initialization()
    app.run(host='0.0.0.0', port=5002, debug=True)
