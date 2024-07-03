# This is the script for pretrain / finetuned mms checkpoint to the right format for fairseq
# The modification is only for the idx in checkpoint['model'] (checkpoint is an dict type data)
# Powered by Hao SHI @ Kyoto University
# Updata data: 2023/10/25 @ group meeting
import torch



# path = '/Work21/2023/zhaojiahui/ssl_asr/fairseq/mms1b_all.pt'
path = "/Work21/2023/zhaojiahui/dual_encoder/pretrained/en.pth"


# This file can be downloaded as: https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt
# checkpoint_fairseq = torch.load('/Work21/2023/zhaojiahui/ssl_asr/fairseq/hubert_large_ll60k.pt')


# This is the only pre-trained model link: https://huggingface.co/facebook/mms-300m
# 
# This is another checkpoint which is finetuned by massive languages:
# (https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
# Model: MMS-1B-all
# Languages: 1162
# Dataset: MMS-lab + FLEURS + CV + VP + MLS
# The link is: https://dl.fbaipublicfiles.com/mms/asr/mms1b_all.pt
# 
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# The idx in finetuned checkpoint will be changed: 
# For example: 
# encoder.layers.23.self_attn.v_proj.weight 
# -> 
# w2v_encoder.w2v_model.encoder.layers.47.self_attn.k_proj.weight
# So, we modify the idxs
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
checkpoint_mms = torch.load(path)

new_checkpoint = {}

for idx, weight in checkpoint_mms.items():
    idx = idx.split('.', 1)
    if(idx[0]=='encoder'):
        tmp = idx[1].split('.', 1)
        if len(tmp) == 1:
            key = idx[0] + '.' + tmp[0] + "_en"
        else:
            key = idx[0] + '.' + tmp[0] + "_en." + tmp[1]
    else:
        key = idx[0] + '.' + idx[1]
    new_checkpoint[key] = weight

    # elif(idx == 'model'):
    #     model_weights = {}
    #     for idxs, weight in checkpoint_mms[idx].items():
    #         if("base" in path):
    #             model_weights[idxs] = weight
    #         else:
    #             model_weights[idxs.replace("w2v_encoder.w2v_model.", "")] = weight

    #     new_checkpoint[idx] = model_weights

    # else:
    #     new_checkpoint[idx] = checkpoint_fairseq[idx]


torch.save(new_checkpoint, './en.pth')