#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=11
stop_stage=11

train_set="train"
valid_set="valid"
test_sets="devman devsge"

asr_config=conf/tuning/train_asr_conformer.yaml
lm_config=conf/tuning/train_lm_transformer.yaml
inference_config=conf/decode_asr.yaml

# pretrained_model="/Work21/2023/zhaojiahui/dual_encoder/pretrained/cn_new.pth /Work21/2023/zhaojiahui/dual_encoder/pretrained/en_new.pth"
pretrained_model="pretrained/cn.pth pretrained/en.pth"
asr_args=
#"--freeze_param encoder.embed encoder.encoders encoder.after_norm decoder ctc"

if [ ! -f "data/train/token.man.2" ]; then
    # must preprocess data first to get Mandarin character tokens
    if [ ${stage} -eq 1 ]; then
        ./asr.sh --stage 1 --stop_stage 1
        stage=2
    else
        echo "Error: data/train/token.man.2 does not exist! Run from stage=1 again."
        exit 1
    fi
fi

man_chars=2618
bpe_nlsyms=""

source data/train/token.man.2  # for bpe_nlsyms & man_chars
nbpe=$((3000 + man_chars + 4))  # 5626
# English BPE: 3000 / Mandarin: 2622 / other symbols: 4

CUDA_VISIBLE_DEVICES=3 ./asr.sh \
    --ngpu 1 \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --nbpe ${nbpe} \
    --bpe_nlsyms "${bpe_nlsyms}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --asr_args "${asr_args}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --pretrained_model "${pretrained_model}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text.eng.bpe" \
    --score_opts "-e utf-8 -c NOASCII" \
    "$@"
