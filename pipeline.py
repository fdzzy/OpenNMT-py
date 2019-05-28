import os
import codecs
import torch
import sentencepiece as spm
import pandas as pd
from ww import f

Encoding = "utf-8"
ONMT = "."
WORK_DIR = "./experiments"

class ExperimentSetting(object):
    def __init__(self, name, data_dir="", vocab_size=80000):
        self.name = name
        self.datda_dir = data_dir
        self.vocab_size = vocab_size

SETTINGS_LIST = [
    ExperimentSetting("debug", vocab_size=80000),
    ExperimentSetting("debug_persona", vocab_size=10000),
    ExperimentSetting("debug_persona_rnn", vocab_size=10000),
    ExperimentSetting("OpenSubTitles_tok_bpe", data_dir="/home/zhiyingz/work/generation/data/OpenNMT_OpenSubtitles/opensub_qa_en", vocab_size=32000),
    ExperimentSetting("twitter_tok_bpe", data_dir="/home/zhiyingz/work/generation/data/twitter", vocab_size=80000),
    ExperimentSetting("twitter_tok_bpe_cmd_transformer_big"),
    ExperimentSetting("index_bpe", data_dir="/home/zhiyingz/work/generation/data/index"),
    ExperimentSetting("index_bpe_cmd"),
    ExperimentSetting("index_bpe_cmd_transformer_big", data_dir="/home/zhiyingz/work/generation/data/index"),
    ExperimentSetting("twitter_tok_bpe_cmd_no_uid_trans_big"),
    ExperimentSetting("twitter_tok_bpe_cmd_uid_trans_big"),
    ExperimentSetting("twitter_tok_bpe_cmd_uid_trans_big_new_uid_mlp"),
    ExperimentSetting("twitter_tok_bpe_cmd_uid_trans_big_new_uid_mlp2"),
    ExperimentSetting("twitter_tok_bpe_cmd_uid_rnn_plus_emb"),
    ExperimentSetting("twitter_tok_bpe_cmd_uid_rnn_plus_emb2")
]

#EXP_NAME = "debug_persona"
#EXP_NAME = "debug_persona_rnn"
#EXP_NAME = "twitter_tok_bpe_cmd_transformer_big"
#EXP_NAME = "twitter_tok_bpe_cmd_no_uid_trans_big"
#EXP_NAME = "twitter_tok_bpe_cmd_uid_trans_big"
#EXP_NAME = "twitter_tok_bpe_cmd_uid_trans_big_new_uid_mlp"
#EXP_NAME = "twitter_tok_bpe_cmd_uid_trans_big_new_uid_mlp2"
#EXP_NAME = "index_bpe_cmd_transformer_big"
#EXP_NAME = "twitter_tok_bpe_cmd_uid_rnn_plus_emb"
EXP_NAME = "twitter_tok_bpe_cmd_uid_rnn_plus_emb2"


#====== EXPERIMENT BEGIN ======

settings_dict = {setting.name : setting for setting in SETTINGS_LIST}
CurrentSetting = settings_dict[EXP_NAME]
DATA_DIR = CurrentSetting.datda_dir
VOCAB_SIZE = CurrentSetting.vocab_size

OUT = f("experiments/{EXP_NAME}")

TRAIN_SRC = f("{DATA_DIR}/src-train-tokenized.txt")
TRAIN_TGT = f("{DATA_DIR}/tgt-train-tokenized.txt")
VALID_SRC = f("{DATA_DIR}/src-valid-tokenized.txt")
VALID_TGT = f("{DATA_DIR}/tgt-valid-tokenized.txt")

def run_cmd(cmd):
    print(f("Run command :: {cmd}"))
    retval = os.system(cmd)
    if retval != 0:
        raise Exception("Check exceptions!")

def remove_log_file(remove_tensorboard_folder=True):
    print("removing log file")
    log_file = f("{OUT}/log/log_file.txt")
    if os.path.exists(log_file):
        os.remove(log_file)

    if remove_tensorboard_folder:
        log_dir = f("{OUT}/log")
        for item in os.listdir(log_dir):
            dir_name = os.path.join(log_dir, item)
            if os.path.isdir(dir_name):
                run_cmd("rm -rf {}".format(dir_name))

def ensure_exist(path):
    if os.path.exists(path):
        return
    else:
        cmd = f("mkdir -p {path}")
        run_cmd(cmd)

def get_line_count(filename):
    with codecs.open(filename, 'r', Encoding) as reader:
        counter = 0
        for line in reader:
            line = line.strip()
            if not line:
                counter += 1
        return counter

def lines_check(left, right):
    print(f("Line count check of {left} and {right}..."))
    left_count = get_line_count(left)
    right_count = get_line_count(right)
    assert left_count == right_count, f("line count not the same: left: {left_count}, right {right_count}")

def s0_sanity_check():
    # Check if input exists
    for filename in [TRAIN_SRC, TRAIN_TGT, VALID_SRC, VALID_TGT]:
        print(f("Checking existence of {filename}..."))
        if not os.path.exists(filename):
            raise Exception(f("Input file {filename} does not exist. Please fix the paths."))

    lines_check(TRAIN_SRC, TRAIN_TGT)
    lines_check(VALID_SRC, VALID_TGT)

def s0a_ensure_dir_existence():
    print(f("Output dir = {OUT}"))
    ensure_exist(OUT)
    ensure_exist(os.path.join(OUT, "data"))
    ensure_exist(os.path.join(OUT, "models"))
    ensure_exist(os.path.join(OUT, "test"))
    ensure_exist(os.path.join(OUT, "log"))

def _apply_bpe(bpe_code_file, input, output):
    cmd = f("{ONMT}/tools/apply_bpe.py -c {bpe_code_file} <  {input} > {output}")
    run_cmd(cmd)

def s1a_preprocess_inputs_bpe_merge():
    print("Step 1a: Preprocess inputs bpe merge")
    run_cmd(f("cat {TRAIN_SRC} {TRAIN_TGT} > {OUT}/data/merged_data"))
    print("Learning BPE from merged data...")
    run_cmd(f("{ONMT}/tools/learn_bpe.py -s {VOCAB_SIZE} < {OUT}/data/merged_data > {OUT}/data/bpe-codes.txt"))
    run_cmd(f("rm {OUT}/data/merged_data"))
    
    inputs = [TRAIN_SRC, VALID_SRC, TRAIN_TGT, VALID_TGT]
    outputs = [f("{OUT}/data/train.src"), f("{OUT}/data/valid.src"), f("{OUT}/data/train.tgt"), f("{OUT}/data/valid.tgt")]
    for input, output in zip(inputs, outputs):
        print(f("Applying BPE to {input}"))
        _apply_bpe(f("{OUT}/data/bpe-codes.txt"), input, output)

    ## TODO: Check unk rate, maybe filter by unk rate

def _apply_spm(spm_processor, input, output):
    with codecs.open(input, 'r', Encoding) as reader, codecs.open(output, 'w', Encoding) as writer:
        for line in reader:
            line = line.strip()
            if line:
                pieces = spm_processor.EncodeAsPieces(line)
                pieces = [x for x in pieces if x]
                if len(pieces) > 0:
                    writer.write(" ".join(pieces) + "\n")

def s1a_preprocess_inputs_spm_merge():
    print("Step 1a: Preprocess inputs spm merge")
    merged_data_path = f("{OUT}/data/merged_data")
    run_cmd(f("{TRAIN_SRC} {TRAIN_TGT} > {merged_data_path}"))
    print("Learning spm model from merged data...")
    model_prefix = "convo_spm"
    cmd = f("--input={merged_data_path} --model_prefix={model_prefix} --vocab_size={VOCAB_SIZE}")
    spm.SentencePieceTrainer.Train(cmd)
    run_cmd(f("rm {merged_data_path}"))

    sp = spm.SentencePieceProcessor()
    spm_model_path = f("{model_prefix}.model")
    sp.Load(spm_model_path)

    inputs = [TRAIN_SRC, VALID_SRC, TRAIN_TGT, VALID_TGT]
    outputs = [f("{OUT}/data/train.src"), f("{OUT}/data/valid.src"), f("{OUT}/data/train.tgt"), f("{OUT}/data/valid.tgt")]
    for input, output in zip(inputs, outputs):
        print(f("Applying spm to {input}"))
        _apply_spm(sp, input, output)

def s1b_preprocess():
    print("Step 1b: Preprocess")
    cmd = f("python {ONMT}/preprocess.py " +
            "-train_src {OUT}/data/train.src " +
            "-train_tgt {OUT}/data/train.tgt " +
            "-valid_src {OUT}/data/valid.src " +
            "-valid_tgt {OUT}/data/valid.tgt " +
            "-save_data {OUT}/data/processed " +
            "--log_file {OUT}/log/preprocess.log "
            "--src_words_min_frequency 10 --tgt_words_min_frequency 10 " +
            "--src_vocab_size 100000 --tgt_vocab_size 100000 " +
            "--share_vocab"
        )
    run_cmd(cmd)

def preprocess_persona():
    cmd = f("python {ONMT}/preprocess_persona.py "
            "--train {OUT}/data/train.txt --valid {OUT}/data/valid.txt "
            "--save_data {OUT}/data/processed --log_file {OUT}/log/preprocess.log "
            "--words_min_frequency 10 --max_vocab_size 100000 "
            # uncomment the following line if test no_uid
            "--no_uid"
        )
    run_cmd(cmd)

def _get_gpu_params(visible_gpus):
    #TODO: check if it works with no GPUs
    gpu_devices_str = ','.join([str(x) for x in visible_gpus])
    world_size = len(visible_gpus)
    gpu_ranks = ' '.join([str(i) for i in range(world_size)])

    CUDA_VISIBLE_str = 'CUDA_VISIBLE_DEVICES="{}"'.format(gpu_devices_str)
    GPU_PARAMS_str = '-world_size {}'.format(world_size)
    if gpu_ranks:
        GPU_PARAMS_str += ' -gpu_ranks {}'.format(gpu_ranks)
    
    return CUDA_VISIBLE_str, GPU_PARAMS_str

def s2_train(train_from=-1, visible_gpus=[]):
    print("Step 2: Train")
    CUDA_VISIBLE_str, GPU_PARAMS_str = _get_gpu_params(visible_gpus)
    cmd = f("{CUDA_VISIBLE_str} python {ONMT}/train.py -data {OUT}/data/processed " +
        "-save_model {OUT}/models/{EXP_NAME} " +
        "--keep_checkpoint 5 --layers 6 --rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 " +
        "-encoder_type transformer -decoder_type transformer -position_encoding " +
        "-train_steps 10000000 -max_generator_batches 2 -dropout 0.1 " +
        "-batch_size 4096 -batch_type tokens -normalization tokens -accum_count 2 " +
        "-optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 " +
        "-max_grad_norm 0 -param_init 0 -param_init_glorot "
        "-label_smoothing 0.1 -valid_steps 5000 -save_checkpoint_steps 5000 " +
        "{GPU_PARAMS_str} " +
        "--tensorboard_log_dir {OUT}/log --log_file {OUT}/log/log_file.txt")
    if train_from > 0:
        cmd += f(" --train_from {OUT}/models/{EXP_NAME}_step_{train_from}.pt")
    run_cmd(cmd)

def s2_train_transformer_large(train_from=-1, visible_gpus=[]):
    print("Step 2: Train")
    CUDA_VISIBLE_str, GPU_PARAMS_str = _get_gpu_params(visible_gpus)
    cmd = f("{CUDA_VISIBLE_str} python {ONMT}/train.py -data {OUT}/data/processed " +
        "-save_model {OUT}/models/{EXP_NAME} --master_ip localhost --master_port 10012 " +
        "--keep_checkpoint 5 --layers 6 --rnn_size 1024 -word_vec_size 1024 " +
        "--share_decoder_embeddings --share_embeddings " +
        "-encoder_type transformer -decoder_type transformer -position_encoding -transformer_ff 4096 -heads 16 " +
        "-train_steps 10000000 -max_generator_batches 2 -dropout 0.3 " +
        "-batch_size 2000 -batch_type tokens -normalization tokens -accum_count 4 " +
        "-optim adam -adam_beta2 0.997 -decay_method noam -warmup_steps 8000 -learning_rate 2 " +
        "-max_grad_norm 0 -param_init 0 -param_init_glorot "
        "-label_smoothing 0.1 -valid_steps 2000 -save_checkpoint_steps 2000 " +
        "{GPU_PARAMS_str} " +
        "--tensorboard --tensorboard_log_dir {OUT}/log --log_file {OUT}/log/log_file.txt")
    if train_from > 0:
        cmd += f(" --train_from {OUT}/models/{EXP_NAME}_step_{train_from}.pt")
    run_cmd(cmd)

def s2_train_persona_rnn(train_from=-1, visible_gpus=[], uid_vocab_size=0, uid_emb_size=0):
    print("Step 2: Train")
    CUDA_VISIBLE_str, GPU_PARAMS_str = _get_gpu_params(visible_gpus)
    cmd = f("{CUDA_VISIBLE_str} python {ONMT}/train.py --data {OUT}/data/processed "
        "--save_model {OUT}/models/{EXP_NAME} "
        "--word_vec_size 1024 --encoder_type brnn --decoder_type rnn --layers 2 --rnn_size 1024 "
        "--dropout 0.3 --batch_size 150 --share_decoder_embeddings --share_embeddings "
        "--uid_vocab_size {uid_vocab_size} --uid_embedding_size {uid_emb_size} "
        "--train_steps 10000000 --start_decay_steps 10000000 --decay_steps 10000000 "
        "--optim sgd --learning_rate 1 --learning_rate_decay 0.5 --max_grad_norm 5 "
        "--valid_steps 5000 -save_checkpoint_steps 5000 --keep_checkpoint 5 "
        "{GPU_PARAMS_str} "
        "--tensorboard --tensorboard_log_dir {OUT}/log --log_file {OUT}/log/log_file.txt")
    if train_from > 0:
        cmd += f(" --train_from {OUT}/models/{EXP_NAME}_step_{train_from}.pt")
    run_cmd(cmd)

def s2_train_persona_transformer_large(train_from=-1, visible_gpus=[], uid_vocab_size=0, uid_emb_size=0):
    print("Step 2: Train")
    CUDA_VISIBLE_str, GPU_PARAMS_str = _get_gpu_params(visible_gpus)
    cmd = f("{CUDA_VISIBLE_str} python {ONMT}/train.py -data {OUT}/data/processed "
        "-save_model {OUT}/models/{EXP_NAME} --master_ip localhost --master_port 10012 "
        "--keep_checkpoint 5 --layers 6 --rnn_size 1024 -word_vec_size 1024 "
        "--share_decoder_embeddings --share_embeddings "
        "-encoder_type transformer -decoder_type transformer -position_encoding -transformer_ff 4096 -heads 16 "
        "-train_steps 10000000 -max_generator_batches 2 -dropout 0.3 "
        "-batch_size 1600 -batch_type tokens -normalization tokens -accum_count 4 "
        "-optim adam -adam_beta2 0.997 -decay_method noam -warmup_steps 8000 -learning_rate 2 "
        "-max_grad_norm 0 -param_init 0 -param_init_glorot "
        "-label_smoothing 0.1 -valid_steps 2000 -save_checkpoint_steps 2000 "
        "{GPU_PARAMS_str} "
        "--uid_vocab_size {uid_vocab_size} --uid_embedding_size {uid_emb_size} "
        "--tensorboard --tensorboard_log_dir {OUT}/log --log_file {OUT}/log/log_file.txt")
    if train_from > 0:
        cmd += f(" --train_from {OUT}/models/{EXP_NAME}_step_{train_from}.pt")
    run_cmd(cmd)

def s3_translate_valid(model_step):
    print("Step 3: Translate Valid")
    cmd = f("CUDA_VISIBLE_DEVICES=\"\" python {ONMT}/translate.py " + 
        "-model {OUT}/models/{EXP_NAME}_step_{model_step}.pt " +
        "-src {OUT}/data/valid.src.head -replace_unk -verbose " +
        "-n_best 10 --beam_size 20 --block_ngram_repeat 1 | tee {OUT}/test/valid_beam_10.log")
    run_cmd(cmd)

def s3_translate_persona_valid(model_step):
    cmd = f("CUDA_VISIBLE_DEVICES=\"\" python {ONMT}/translate_persona.py "
        "-model {OUT}/models/{EXP_NAME}_step_{model_step}.pt "
        "-data {OUT}/data/valid.head.txt --has_target -replace_unk -verbose "
        "-n_best 10 --beam_size 20 --block_ngram_repeat 1 | tee {OUT}/test/valid_beam_10.log")
    run_cmd(cmd)

def s3_translate_test(model_step):
    print("Step 3: Translate test")
    cmd = f("CUDA_VISIBLE_DEVICES=\"\" python {ONMT}/translate.py " + 
        "-model {OUT}/models/{EXP_NAME}_step_{model_step}.pt " +
        "-src {OUT}/data/test.src -replace_unk -verbose " +
        "-n_best 10 --beam_size 20 --block_ngram_repeat 1 | tee {OUT}/test/test_beam_10.log")
    run_cmd(cmd)

def s3_translate_persona_test(model_step):
    cmd = f("CUDA_VISIBLE_DEVICES=\"\" python {ONMT}/translate_persona.py "
        "-model {OUT}/models/{EXP_NAME}_step_{model_step}.pt "
        "-data {OUT}/data/test.uid.src -replace_unk -verbose "
        "-n_best 10 --beam_size 20 --block_ngram_repeat 1 | tee {OUT}/test/test_beam_10.log")
    run_cmd(cmd)

def s3_translate_test_interactive(model_step):
    cmd = f("CUDA_VISIBLE_DEVICES=\"\" python {ONMT}/translate_interactive.py "
        "-model {OUT}/models/{EXP_NAME}_step_{model_step}.pt "
        "-replace_unk --report_time "
        "-n_best 10 --beam_size 20 --block_ngram_repeat 1")
    run_cmd(cmd)

def s3_translate_persona_test_interactive(model_step, uid):
    cmd = f("CUDA_VISIBLE_DEVICES=\"\" python {ONMT}/translate_persona_interactive.py "
        "-model {OUT}/models/{EXP_NAME}_step_{model_step}.pt "
        "--uid {uid} -replace_unk --report_time "
        "-n_best 10 --beam_size 20 --block_ngram_repeat 1")
    run_cmd(cmd)

def average_models_old(model_start, model_step, model_count):
    models = []
    model_format = f("{OUT}/models/{EXP_NAME}_step_") + "%d.pt"
    for i in range(model_count):
        model_path = model_format % (model_start + i * model_step)
        if os.path.exists(model_path):
            models.append(model_path)
    if len(models) < 2:
        raise Exception("only %d valid model exist!" % len(models))
    cmd = f("CUDA_VISIBLE_DEVICES=\"\" python {ONMT}/tools/average_models.py -to_cpu " + 
            "-models " + " ".join(models) + " "
            "-output {OUT}/models/{EXP_NAME}_average.pt"
        )
    run_cmd(cmd)

def average_models(model_start, model_step, model_count, to_cpu=True):
    model_files = []
    model_format = f("{OUT}/models/{EXP_NAME}_step_%d.pt")
    for i in range(model_count):
        model_full_path = model_format % (model_start + i * model_step)
        if os.path.exists(model_full_path):
            model_files.append(model_full_path)
        else:
            print("WARNING: model file does not exist: %s" % model_full_path)
    if len(model_files) < 2:
        raise Exception("only %d valid model exist!" % len(model_files))

    vocab = None
    opt = None
    avg_model = None
    avg_generator = None

    for i, model_file in enumerate(model_files):
        print("loading file %s" % model_file)
        m = torch.load(model_file, map_location='cpu') if to_cpu else torch.load(model_file)
        model_weights = m['model']
        generator_weights = m['generator']

        if i == 0:
            vocab, opt = m['vocab'], m['opt']
            avg_model = model_weights
            avg_generator = generator_weights
        else:
            for (k, v) in avg_model.items():
                avg_model[k].mul_(i).add_(model_weights[k]).div_(i + 1)
            for (k, v) in avg_generator.items():
                avg_generator[k].mul_(i).add_(generator_weights[k]).div_(i + 1)
    
    final = {"vocab": vocab, "opt": opt, "optim": None, "generator": avg_generator, "model": avg_model}
    output_file = f("{OUT}/models/{EXP_NAME}_step_average.pt")
    print("Saving to %s" % output_file)
    torch.save(final, output_file)

def get_average_token_speed():
    src_speeds = []
    tgt_speeds = []
    log_filename = f("{OUT}/log/log_file.txt")
    print("Checking log file: %s" % log_filename)
    for line in open(log_filename, "r"):
        items = line.strip().split(";")
        if len(items) <= 1:
            continue
        for item in items:
            if item.endswith("tok/s"):
                tokens = item.strip().split()
                if len(tokens) != 2:
                    continue
                tokens = tokens[0].split("/")
                if len(tokens) != 2:
                    continue
                src_speed = int(tokens[0])
                tgt_speed = int(tokens[1])
                src_speeds.append(src_speed)
                tgt_speeds.append(tgt_speed)
    print("Source speeds: ")
    print(pd.Series(src_speeds).describe())
    print("\nTarget speeds: ")
    print(pd.Series(tgt_speeds).describe())

if __name__ == '__main__':
    visible_gpus = [0,1]#[2,3]
    #s0_sanity_check()
    #s0a_ensure_dir_existence()
    #s1a_preprocess_inputs_bpe_merge()
    #s1a_preprocess_inputs_spm_merge()
    #s1b_preprocess()
    #preprocess_persona()
    remove_log_file()
    s2_train_persona_rnn(visible_gpus=visible_gpus, uid_vocab_size=2789420, uid_emb_size=10)
    #s2_train_transformer_large(train_from=-1, visible_gpus=visible_gpus)
    #s2_train_persona_transformer_large(train_from=-1, visible_gpus=visible_gpus, uid_vocab_size=2789420, uid_emb_size=10)
    #s3_translate_test(model_step=5)
    #s3_translate_valid(model_step=160000)
    #s3_translate_persona_valid(model_step=38000)
    #s3_translate_test(model_step=160000)
    #s3_translate_persona_test(model_step=38000)
    #s3_translate_test_interactive(model_step='average')
    #s3_translate_persona_test_interactive(model_step='average', uid=247694)
    #average_models(model_start=820000, model_step=5000, model_count=5)

#===== EXPERIMENT END ======
