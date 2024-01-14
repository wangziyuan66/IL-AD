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





def modification_base(read,can_model, mod_model,alphabet_info,canonical="ACGT",
                        idx = [0,1,1,2,3],length = 100,scale=1,
                        right_len = 0, canonical_base = "C", mod_base = "m"):
    m_position = np.where(np.array(list(canonical))[idx][read.get_reference()]==canonical_base)[0]
    signals = []
    seqs = []
    seq_len = []
    potention_m = []
    quantile_loss = []
    for pos in m_position:
        if pos < len(read.get_reference())-25 and pos > 25:
            if -1 in read.Ref_to_signal: return ""
            signal,seq = extract_signal(pos,read, length = length, right_len = right_len)
            seq = np.array(idx)[seq]
            signals.append(signal)
            chunk_seq = flipflopfings.flipflop_code(
                seq, alphabet_info.ncan_base)
            seqs.append(chunk_seq)
            seq_len.append(len(chunk_seq))
            potention_m.append(pos)
            
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
    # loss = torch.mean(F.mse_loss(reconstrution,indata.to("cuda"),reduction = "none"),dim=0)
    with torch.set_grad_enabled(False):
        can_outputs,mod_outputs = can_model(indata.cuda()),mod_model(indata.cuda())
        nblk = float(can_outputs.shape[0])
        ntrans = can_outputs.shape[2]
        can_lossvector = ctc.crf_flipflop_loss(
            can_outputs[:, :, :40], seqs, seqlens, 1.0)

        can_lossvector += layers.flipflop_logpartition(
            can_outputs[:, :, :40]) / nblk

        mod_lossvector = ctc.crf_flipflop_loss(
            mod_outputs[:, :, :40], seqs, seqlens, 1.0)

        mod_lossvector += layers.flipflop_logpartition(
            mod_outputs[:, :, :40]) / nblk
        basecall = np.array([canonical[i] for i in read.get_reference()])
        mls = can_lossvector/mod_lossvector*1.0
        # print((can_lossvector,mod_lossvector))
        basecall[potention_m[torch.where(mls>scale/2)[0].cpu()]] = mod_base
        basecall[potention_m[torch.where(mls<scale/2)[0].cpu()]] = canonical_base
        mls = mls[torch.where(mls>scale/2)]
        ml_tag = torch.ceil((mls-1)*256/scale)
        ml_tag[torch.where(ml_tag>255)]=255
        # print(sum(basecall =="m")/(sum(basecall =="C")+sum(basecall =="m")))
    return "".join(basecall.tolist()),ml_tag.cpu().numpy().tolist()


def extract_signal(pos,read,length = 50,right_len = 0, mapping = True):
    winlen = int(length/2)
    if mapping:
        map_location = read.get_reference_locations((read.Ref_to_signal[pos]-winlen,read.Ref_to_signal[pos]+winlen+right_len))
        seq = read.get_reference()[map_location[0]:map_location[1]]
    else:
        seq = read.get_reference()[pos-2:pos+3]
    signal = read.get_current()[read.Ref_to_signal[pos]-winlen:read.Ref_to_signal[pos]+winlen+right_len]
    return np.pad(signal,(0,length+right_len-signal.size),"edge"),seq

def get_parser():
    parser = argparse.ArgumentParser(
        description="Basecall reads using a taiyaki model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "input_hdf5", action=FileExists,
        help="The hdf5 to input")
    parser.add_argument(
        "can_model", action=FileExists,
        help="The inference model")
    parser.add_argument(
        "mod_model", action=FileExists,
        help="The inference model")
    parser.add_argument(
        "can", type=str,
        help="The target base in canonical alphabet(ACGT)")
    parser.add_argument(
        "mod", type=str,
        help="The corresponding label for modification for the target base (y:m6A, m:5mC, h:5hmC)")
    parser.add_argument(
        '--can_base_idx', default=[0,1,1,2,3], 
        help='Write output in fastq format (default is fasta)')
    parser.add_argument(
        "output", type=str,
        help="The fasta file to output")
    parser.add_argument(
        "--limit", type=Positive(int),default=None,
        help="Then number of output reads")
    parser.add_argument(
        "--length", type=int,default=100,
        help="signal length parameter for AD")
    parser.add_argument(
        "--type", type=str,default="dna",
        help="specify the output type: dna/rna")
    parser.add_argument(
        '--right_len', type=int,default=0,
        help='length for one base')
    parser.add_argument(
        '--scale', type=float,default=3)
    

    return parser

def main():
    args = get_parser().parse_args()
    # print(args)
    can_model = torch.load(args.can_model)
    mod_model = torch.load(args.mod_model)
    input = args.input_hdf5
    idx = [int(i) for i in list(args.can_base_idx)]
    with mapped_signal_files.MappedSignalReader(input) as msr:
        alphabet_info = msr.get_alphabet_information()
        read_data = list(islice(msr.reads(None), args.limit))

    print(len(read_data))
    fh = open(args.output,"w")
    ml = open(args.output.split(".fa")[0]+".ml","w")
    # read = read_data[0]
    for read in read_data:
        seq = modification_base(read, can_model, mod_model,alphabet_info,canonical = alphabet_info.alphabet,
        idx = idx,length = args.length*2, right_len = args.right_len,
        canonical_base=args.can, mod_base=args.mod,scale = args.scale)
        # print(seq)
        if seq !="":
            if args.type=="rna":
                fh.write("{}{}\n{}\n".format(
                ">", read.read_id[2:-1],
                seq[0][::-1]))
                ml.write("{}\t{}\n".format(
                read.read_id[2:-1],
                [int(ml) for ml in seq[1]]))
            else:
                fh.write("{}{}\n{}\n".format(
                ">", read.read_id[2:-1],
                seq[0]))
                ml.write("{}\t{}\n".format(
                read.read_id[2:-1],
                [int(ml) for ml in seq[1]]))
    fh.close()
    ml.close()

if __name__ == '__main__':
    main()
