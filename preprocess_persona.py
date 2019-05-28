#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""
import os
import codecs
import glob
import sys
import gc
import torch
from torchtext.data import Field
from torchtext.vocab import Vocab
from functools import partial
from collections import Counter

from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.inputters.text_dataset import text_sort_key, TextMultiField

def check_existing_pt_files(opt):
    """ Check if there are existing .pt files to avoid overwriting them """
    pattern = opt.save_data + '.{}*.pt'
    for t in ['train', 'valid', 'vocab']:
        path = pattern.format(t)
        if glob.glob(path):
            sys.stderr.write("Please backup existing pt files: %s, "
                             "to avoid overwriting them!\n" % path)
            sys.exit(1)

def build_save_dataset(corpus_type, fields, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        input_path = opt.train
    else:
        input_path = opt.valid

    logger.info("Reading data file: %s." % (input_path))

    data_shards = split_corpus(input_path, opt.shard_size) # joey: a generator that generates each time a list of at most shard_size lines of strings
    output_dataset_paths = []
    uid_set = set()
    for i, tsv_data_lines in enumerate(data_shards):
        logger.info("Building shard %d." % i)

        dataset = inputters.TabularDataset(tsv_data_lines, fields, separator='\t')
        input_path = "{:s}.{:s}.{:d}.pt".format(opt.save_data, corpus_type, i)
        output_dataset_paths.append(input_path)
        if not opt.no_uid:
            uid_set.update([int(example.uid) for example in dataset.examples])

        logger.info(" * saving %sth %s data shard to %s."
                    % (i, corpus_type, input_path))

        dataset.save(input_path)

        del dataset.examples
        gc.collect()
        del dataset
        gc.collect()

    if len(uid_set) > 0:
        logger.info("{} unique user ids, min: {}, max: {}".format(len(uid_set), min(uid_set), max(uid_set)))

    return output_dataset_paths

def build_save_vocab(train_dataset_files, fields, opt):
    # Vocab is shared between src and tgt
    counter = Counter()

    field_dict = dict(fields)
    src_field = field_dict["src"]
    tgt_field = field_dict["tgt"]
    for i, path in enumerate(train_dataset_files):
        dataset = torch.load(path)
        logger.info(" * reloading %s." % path)
        for example in dataset.examples:
            counter.update(example.src[0])
            counter.update(example.tgt[0])

        # Drop the none-using from memory but keep the last
        if i < len(train_dataset_files) - 1:
            dataset.examples = None
            gc.collect()
            del dataset.examples
            gc.collect()
            del dataset
            gc.collect()

    all_specials = [tgt_field.base_field.unk_token, tgt_field.base_field.pad_token,
                tgt_field.base_field.init_token, tgt_field.base_field.eos_token]
    specials = [tok for tok in all_specials if tok is not None]
    vocab = Vocab(counter, specials=specials, max_size=opt.max_vocab_size, min_freq=opt.words_min_frequency)
    # Add pad_vocab_to_multiple if necessary

    src_field.base_field.vocab = vocab
    tgt_field.base_field.vocab = vocab
    logger.info(" * vocab size: %d." % len(src_field.base_field.vocab))

    vocab_path = opt.save_data + '.vocab.pt'
    torch.save(dict(fields), vocab_path)

def get_fields(pad='<blank>', bos='<s>', eos='</s>', no_uid=False):
    # build fields for the tsv data set
    src_field = Field(include_lengths=True, pad_token=pad, init_token=None, eos_token=None)
    tgt_field = Field(include_lengths=False, pad_token=pad, init_token=bos, eos_token=eos) # set is_target?
    uid_field = Field(sequential=False, use_vocab=False, include_lengths=False, dtype=torch.long) # no need to set tokenizer for non-sequential field
    
    # Encapsulate the field into TextMultiField, to be compatible with OpenNMT-py
    src_multifield = TextMultiField('src', src_field, feats_fields=[])
    tgt_multifield = TextMultiField('tgt', tgt_field, feats_fields=[])

    #fields = [("src", src_field), ("tgt", tgt_field), ("uid", uid_field)]
    if no_uid:
        fields = [("src", src_multifield), ("tgt", tgt_multifield)]
    else:
        fields = [("src", src_multifield), ("tgt", tgt_multifield), ("uid", uid_field)]

    return fields

def validate_preprocess_presona_args(opt):
    assert os.path.isfile(opt.train), "Please check path of your train file!"
    assert not opt.valid or os.path.isfile(opt.valid), "Please check path of your valid file!"

def main(opt):
    validate_preprocess_presona_args(opt)
    torch.manual_seed(opt.seed)
    check_existing_pt_files(opt)

    init_logger(opt.log_file)
    fields = get_fields(no_uid=opt.no_uid)

    logger.info("Building & saving training data...")
    train_dataset_files = build_save_dataset('train', fields, opt)

    if opt.valid:
        logger.info("Building & saving validation data...")
        build_save_dataset('valid', fields, opt)

    logger.info("Building & saving vocabulary...")
    build_save_vocab(train_dataset_files, fields, opt)

def _get_parser():
    parser = ArgumentParser(description='preprocess_persona.py')

    opts.config_opts(parser)
    opts.preprocess_persona_opts(parser)
    return parser

if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
