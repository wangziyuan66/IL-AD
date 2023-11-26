#!/usr/bin/env python3
from taiyaki import (
    chunk_selection, ctc, flipflopfings, helpers, layers,
    mapped_signal_files, maths, signal_mapping)
from itertools import islice
import numpy as np
from taiyaki.cmdargs import (AutoBool, FileExists, NonNegative,
                             ParseToNamedTuple, Positive)
import argparse                          
import torch
import torch.nn as nn
import torch.nn.functional as F
from taiyaki.squiggle_match import squiggle_match_loss, embed_sequence

KMER_LEN = None

def modification_base(read,model,length,alphabet_info,mapping,right_len = 0,canonical="ACGT",idx = [0,1,1,2,3], k = None, can = "C",scale = 1.0, mod_num = 1):
    m_position = np.where(np.array(list(alphabet_info.alphabet))[read.get_reference()]==can)[0]
    signals = []
    seqs = []
    seq_len = []
    potention_m = []
    kmers = []
    five_mers = []
    for pos in m_position:
        if pos > 20 and pos < len(read.get_reference())-20:
            if -1 in read.Ref_to_signal: return None
            signal,seq = extract_signal(pos,read,length,right_len,mapping=mapping)
            idx = list(idx)
            seq = np.array(idx,dtype = np.int64)[seq]
            kmer = "".join(np.array(list("ACGT"))[seq].tolist())
            five_mer = "".join(np.array(list(canonical))[read.get_reference()[pos-2:pos+3]])    
            chunk_seq = flipflopfings.flipflop_code(
                seq, alphabet_info.ncan_base)
            # print(potention_m)
            if (k == None or k == len(kmer)):
                if mod_num==None:
                    kmers.append(kmer)
                    signals.append(signal)
                    potention_m.append(pos)
                    seqs.append(chunk_seq)
                    five_mers.append(five_mer)
                    seq_len.append(len(chunk_seq))
                elif np.count_nonzero(read.get_reference()[pos-5:pos+6]-alphabet_info.alphabet.index(can)) == 11-mod_num:
                    kmers.append(kmer)
                    signals.append(signal)
                    potention_m.append(pos)
                    seqs.append(chunk_seq)
                    five_mers.append(five_mer)
                    seq_len.append(len(chunk_seq))
    if len(potention_m) ==0: return None
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
        outputs = model(indata.cuda())
        nblk = float(outputs.shape[0])
        ntrans = outputs.shape[2]
        lossvector = ctc.crf_flipflop_loss(
            outputs[:, :, :40], seqs, seqlens, 1.0)

        lossvector += layers.flipflop_logpartition(
            outputs[:, :, :40]) / nblk
    return lossvector.cpu().detach().numpy()*scale,kmers,five_mers

def encoder_kmer(read,pos, idx = [0,1,1,2,3]):
    ref_region = read.get_reference_locations((read.Ref_to_signal[pos]-50,read.Ref_to_signal[pos]+50))
    positions = read.Ref_to_signal[ref_region[0]:ref_region[1]]
    positions = np.insert(positions,len(positions),read.Ref_to_signal[pos]+50)
    positions[0] = read.Ref_to_signal[pos]-50
    kmers = []
    for pidx in range(-3,4):
        refs = read.get_reference()[(ref_region[0]+pidx):(ref_region[1]+pidx)]
        refs = np.array(idx)[refs]
        if (np.diff(positions).shape[0]>refs.shape[0]):
            print(pos)
        kmers.append(encode_onehot(np.concatenate([np.repeat(refs[i],j) for i,j in enumerate(np.diff(positions))]).reshape(1,-1)))
    kmers_encoded = np.concatenate(kmers,axis=1)
    return kmers_encoded

def encode_onehot(a):
    b = np.zeros((a.size, 4),dtype = np.int16)
    b[np.arange(a.size), a] = 1
    return b

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
        "model", action=FileExists,
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
        '--can', type=str,default="C",
        help='incremental_learning_parameters lambda')
    parser.add_argument(
        "--scale", type=Positive(float),default=1.0,
        help="The fasta file to output")
    parser.add_argument(
        "--mod_num", type=Positive(int),default=None,
        help="The fasta file to output")
    parser.add_argument(
        '--right_len', type=int,default=0,
        help='incremental_learning_parameters lambda')
    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument('--mapping', dest='mapping', action='store_true')
    flag_parser.add_argument('--no-mapping', dest='mapping', action='store_false')
    parser.set_defaults(mapping=True)

    return parser




def main():
    args = get_parser().parse_args()
    # print(args.mapping)
    model = torch.load(args.model,"cuda")
    model.eval()

    input = args.input_hdf5
    with mapped_signal_files.MappedSignalReader(input) as msr:
        alphabet_info = msr.get_alphabet_information()
        # load list of signal_mapping.SignalMapping objects
        read_data = list(islice(msr.reads(None), args.limit))


    fh = open(args.output,"w")
    # read = read_data[0]
    for read in read_data:
        loss = modification_base(read, model,args.length*2,alphabet_info,mapping = args.mapping,
        canonical = alphabet_info.alphabet,idx = args.can_base_idx, right_len = args.right_len,
        k = KMER_LEN,can=args.can,scale = args.scale,mod_num = args.mod_num)
        if loss!=None:
            losses,kmers,labels = loss
            for loss,kmer,label in zip(losses,kmers,labels):fh.write("{}\t{}\t{}\n".format(kmer,str(loss),label))

    fh.close()

if __name__ == '__main__':
    main()


