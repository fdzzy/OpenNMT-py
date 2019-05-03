import os
import codecs
import sentencepiece as spm
from ww import f

Encoding = "utf-8"
ONMT="."
WORK_DIR = "./experiments"

class ExperimentType:
    DEDUG = -1
    DEBUG_PERSONA = -2
    OPEN_SUB_TITLES_TOK_BPE = 0
    TWITTER_TOK_BPE = 1
    TWITTER_TOK_BPE_FILTER = 2
    TWITTER_TOK_BPE_FILTER_LEN_CMD = 3
    TWITTER_TOK_BPE_FILTER_LEN_CMD_TRANS = 4
    TWITTER_TOK_BPE_FILTER_LEN_CMD_TRANS_BIG = 5
    CHAT_INDEX = 6
    CHAT_INDEX_CMD = 7
    CHAT_INDEX_CMD_TRANS_BIG = 8

#CurrentExperimentType = ExperimentType.DEDUG
CurrentExperimentType = ExperimentType.DEBUG_PERSONA
#CurrentExperimentType = ExperimentType.CHAT_INDEX_CMD_TRANS_BIG

if CurrentExperimentType == ExperimentType.DEDUG:
    EXP_NAME = "debug"
    DATA_DIR = ""
    VOCAB_SIZE = 80000
elif CurrentExperimentType == ExperimentType.DEBUG_PERSONA:
    EXP_NAME = "debug_persona"
    DATA_DIR = ""
    VOCAB_SIZE = 10000
elif CurrentExperimentType == ExperimentType.OPEN_SUB_TITLES_TOK_BPE:
    EXP_NAME = "OpenSubTitles_tok_bpe"
    DATA_DIR = "/home/zhiyingz/work/generation/data/OpenNMT_OpenSubtitles/opensub_qa_en"
    VOCAB_SIZE = 32000
elif CurrentExperimentType == ExperimentType.TWITTER_TOK_BPE:
    EXP_NAME = "twitter_tok_bpe"
    DATA_DIR = "/home/zhiyingz/work/generation/data/twitter"
    VOCAB_SIZE = 80000
elif CurrentExperimentType == ExperimentType.CHAT_INDEX:
    EXP_NAME = "index_bpe"
    DATA_DIR = "/home/zhiyingz/work/generation/data/index"
    VOCAB_SIZE = -1
elif CurrentExperimentType == ExperimentType.CHAT_INDEX_CMD:
    EXP_NAME = "index_bpe_cmd"
    DATA_DIR = ""
    VOCAB_SIZE = 80000
elif CurrentExperimentType == ExperimentType.CHAT_INDEX_CMD_TRANS_BIG:
    EXP_NAME = "index_bpe_cmd_transformer_big"
    DATA_DIR = "/home/zhiyingz/work/generation/data/index"
    VOCAB_SIZE = 80000


#====== EXPERIMENT BEGIN ======

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

def remove_log_file():
    print("removing log file")
    log_file = f("{OUT}/log/log_file.txt")
    if os.path.exists(log_file):
        os.remove(log_file)

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
            "--words_min_frequency 10 --max_vocab_size 100000 --share_vocab"
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
        "-save_model {OUT}/models/{EXP_NAME} " +
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

def s3_translate_valid(model_step):
    print("Step 3: Translate Valid")
    cmd = f("CUDA_VISIBLE_DEVICES=\"\" python {ONMT}/translate.py " + 
        "-model {OUT}/models/{EXP_NAME}_step_{model_step}.pt " +
        "-src {OUT}/data/valid.src.head -replace_unk -verbose " +
        "-n_best 10 --beam_size 20 --block_ngram_repeat 1 | tee {OUT}/test/valid_beam_10.log")
    run_cmd(cmd)

def s3_translate_test(model_step):
    print("Step 3: Translate test")
    cmd = f("CUDA_VISIBLE_DEVICES=\"\" python {ONMT}/translate.py " + 
        "-model {OUT}/models/{EXP_NAME}_step_{model_step}.pt " +
        "-src {OUT}/data/test.src -replace_unk -verbose " +
        "-n_best 10 --beam_size 20 --block_ngram_repeat 1 | tee {OUT}/test/test_beam_10.log")
    run_cmd(cmd)

def s3_translate_test_interactive(model_step):
    print("Step 3: Translate test")
    cmd = f("CUDA_VISIBLE_DEVICES=\"\" python {ONMT}/translate_interactive.py " +
        "-model {OUT}/models/{EXP_NAME}_step_{model_step}.pt " +
        "-src {OUT}/data/test.src -replace_unk -verbose " +
        "-n_best 10 --beam_size 20 --block_ngram_repeat 1")
    run_cmd(cmd)

def average_models(model_start, model_step, model_count):
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

if __name__ == '__main__':
    visible_gpus = []#[2,3]
    #s0_sanity_check()
    #s0a_ensure_dir_existence()
    #s1a_preprocess_inputs_bpe_merge()
    #s1a_preprocess_inputs_spm_merge()
    #s1b_preprocess()
    preprocess_persona()
    #remove_log_file()
    #s2_train()
    #s2_train_transformer_large(train_from=-1, visible_gpus=visible_gpus)
    #s3_translate_test(model_step=28000)
    #s3_translate_test_interactive(model_step=28000)
    #s3_translate_valid(model_step=160000)
    #s3_translate_test(model_step=160000)
    #average_models(model_start=905000, model_step=5000, model_count=5)

#===== EXPERIMENT END ======
