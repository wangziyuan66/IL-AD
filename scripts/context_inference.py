#!/usr/bin/env python3
from taiyaki import (
    chunk_selection, ctc, flipflopfings, helpers, layers,
    mapped_signal_files, maths, signal_mapping)
from itertools import islice
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from taiyaki.squiggle_match import squiggle_match_loss, embed_sequence
from taiyaki.cmdargs import (AutoBool, FileExists, NonNegative,
                             ParseToNamedTuple, Positive)

LOSS_DICT = {}




def modification_base(read,model,alphabet_info,canonical="ACGT",idx = [0,1,1,2,3],length = 100, right_len = 0, canonical_base = "C", mod_base = "m",thred = 0.6, mapping = True):
    m_position = np.where(np.array(list(canonical))[read.get_reference()]==canonical_base)[0]
    signals = []
    seqs = []
    seq_len = []
    potention_m = []
    quantile_loss = []
    for pos in m_position:
        try:
            if -1 in read.Ref_to_signal: return ""
            five_mer = "".join(np.array(list(canonical))[read.get_reference()[pos-2:pos+3]]) 
            if five_mer not in LOSS_DICT.keys(): continue
            quantile_loss.append(LOSS_DICT[five_mer])
            signal,seq = extract_signal(pos,read, length = length, right_len = right_len, mapping = mapping)
            seq = np.array(idx)[seq]
            signals.append(signal)
            chunk_seq = flipflopfings.flipflop_code(
                seq, alphabet_info.ncan_base)
            seqs.append(chunk_seq)
            seq_len.append(len(chunk_seq))
            potention_m.append(pos)
        except IndexError:
            print("Cannot basecall at "+str(pos))
            
    if len(signals) ==0: return ""
    # print(len(signals))
    potention_m = np.array(potention_m)
    stacked_current = np.vstack([
        np.array(chunk) for chunk in signals]).T
    indata = torch.tensor(stacked_current, device='cuda',
                            dtype=torch.float32).unsqueeze(2)
    seqs = torch.tensor(
        np.concatenate(seqs), dtype=torch.long, device='cpu')
    seqlens = torch.tensor(seq_len, dtype=torch.long, device='cpu')
    quantile_loss = torch.tensor(quantile_loss, dtype=torch.float, device='cpu')
    # loss = torch.mean(F.mse_loss(reconstrution,indata.to("cuda"),reduction = "none"),dim=0)
    with torch.set_grad_enabled(False):
        outputs = model(indata.cuda())
        nblk = float(outputs.shape[0])
        ntrans = outputs.shape[2]
        lossvector = ctc.crf_flipflop_loss(
            outputs[:, :, :40], seqs, seqlens, 1.0)

        lossvector += layers.flipflop_logpartition(
            outputs[:, :, :40]) / nblk
        basecall = np.array([canonical[i] for i in read.get_reference()])
        mls = torch.sum(lossvector.reshape(lossvector.size()[0],1).repeat(1,quantile_loss.shape[1]) 
        - quantile_loss.cuda() <0,1)
        basecall[potention_m[torch.where(mls>50)[0].cpu()]] = mod_base
        ml_tag = mls[torch.where(mls>200)[0].cpu()]
        # print(sum(basecall =="m")/(sum(basecall =="C")+sum(basecall =="m")))
    return "".join(basecall.tolist()),ml_tag.cpu().numpy().tolist()


def extract_signal(pos,read,length = 50,right_len = 0, mapping = True):
    winlen = int(length/2)
    if mapping:
        map_location = read.get_reference_locations((read.Ref_to_signal[pos]-winlen,read.Ref_to_signal[pos]+winlen+right_len))
        seq = read.get_reference()[map_location[0]:map_location[1]]
    else:
        seq = read.get_reference()[pos-2:pos+3]
    signal = read.get_current()[read.Ref_to_signal[pos]-winlen:read.Ref_to_signal[pos]+winlen]
    return np.pad(signal,(0,length-signal.size),"edge"),seq

def get_parser():
    parser = argparse.ArgumentParser(
        description="Basecall reads using a taiyaki model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "input_hdf5", action=FileExists,
        help="The hdf5 to input")
    parser.add_argument(
        "inference_model", action=FileExists,
        help="The inference model")
    parser.add_argument(
        "can", type=str,
        help="The inference model")
    parser.add_argument(
        "mod", type=str,
        help="The inference model")
    parser.add_argument(
        '--can_base_idx', default=[0,1,1,2,3], 
        help='Write output in fastq format (default is fasta)')
    parser.add_argument(
        "output", type=str,
        help="The fasta file to output")
    parser.add_argument(
        "--limit", type=Positive(int),default=None,
        help="The fasta file to output")
    parser.add_argument(
        "--length", type=int,default=100,
        help="The fasta file to output")
    parser.add_argument(
        "--type", type=str,default="dna",
        help="The fasta file to output")
    parser.add_argument(
        "thredhold", action=FileExists,
        help="The thredhold for kmers file to output")
    parser.add_argument(
        '--right_len', type=int,default=0,
        help='incremental_learning_parameters lambda')
    
    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument('--mapping', dest='mapping', action='store_true')
    flag_parser.add_argument('--no-mapping', dest='mapping', action='store_false')
    parser.set_defaults(mapping=True)

    return parser


# model = torch.load("/xdisk/hongxuding/ziyuan/meta-bonito/incremental_learning/context_abnormal/model_final.checkpoint",
#                             "cuda")
# model.eval()
# print("---------------C Percentage m%------------------")
# input = "/xdisk/hongxuding/dinglab/curlcake/analysis/error_signature/cmh/round3/dl_model/c.hdf5"
# with mapped_signal_files.MappedSignalReader(input) as msr:
#     alphabet_info = msr.get_alphabet_information()
#     # load list of signal_mapping.SignalMapping objects
#     read_data = list(islice(msr.reads(None), 50000))


# fh = open("/xdisk/hongxuding/ziyuan/meta-bonito/incremental_learning/context_abnormal/fasta/c/c.fasta","w")
# read = read_data[0]
# for read in read_data:
#     seq = modification_base(read, model,canonical = alphabet_info.collapse_alphabet,idx = [0,1,2,3])
#     fh.write("{}{}\n{}\n".format(
#     ">", read.read_id[2:-1],
#     seq))

# fh.close()

# print("---------------m Percentage m%------------------")
# input = "/xdisk/hongxuding/dinglab/curlcake/analysis/error_signature/cmh/round3/dl_model/m.hdf5"
# with mapped_signal_files.MappedSignalReader(input) as msr:
#     alphabet_info = msr.get_alphabet_information()
#     # load list of signal_mapping.SignalMapping objects
#     read_data = list(islice(msr.reads(None), 50000))


# fh = open("/xdisk/hongxuding/ziyuan/meta-bonito/incremental_learning/context_abnormal/fasta/m/m.fasta","w")
# # read = read_data[0]
# for read in read_data:
#     seq = modification_base(read, model,canonical = alphabet_info.collapse_alphabet)
#     fh.write("{}{}\n{}\n".format(
#     ">", read.read_id[2:-1],
#     seq))

# fh.close()
def generate_loss_dict(tred_path):
    with open(tred_path, 'r') as document:
        for line in document:
            line = line.split(",")
            if not line: # empty line?

                continue

            LOSS_DICT[line[0]] = [float(i) for i in list(line[1:])]

def main():
    args = get_parser().parse_args()
    # print(args.mapping)
    generate_loss_dict(args.thredhold)
    model = torch.load(args.inference_model)
    input = args.input_hdf5
    idx = [int(i) for i in list(args.can_base_idx)]
    with mapped_signal_files.MappedSignalReader(input) as msr:
        alphabet_info = msr.get_alphabet_information()
        read_data = list(islice(msr.reads(None), args.limit))

    print(len(read_data))
    fh = open(args.output,"w")
    ml = open("/".join(args.output.split("/")[:-1])+"/ml.tag","w")
    # read = read_data[0]
    for read in read_data:
        seq = modification_base(read, model,alphabet_info,canonical = alphabet_info.collapse_alphabet,
        idx = idx,length = args.length*2, right_len = args.right_len,
        canonical_base=args.can, mod_base=args.mod, thred = args.thredhold,mapping=args.mapping)
        if seq !="":
            if args.type=="rna":
                fh.write("{}{}\n{}\n".format(
                ">", read.read_id[2:-1],
                seq[::-1]))
            else:
                fh.write("{}{}\n{}\n".format(
                ">", read.read_id[2:-1],
                seq[0]))
                ml.write("{}\t{}\n".format(
                read.read_id[2:-1],
                seq[1]))
    fh.close()
    ml.close()

if __name__ == '__main__':
    main()