import pysam
import numpy as np
import pandas as pd
import array
from taiyaki.cmdargs import FileExists
import argparse
import ast

def get_parser():
    parser = argparse.ArgumentParser(
        description="Basecall reads using a taiyaki model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "bam_file", action=FileExists,
        help="Input BAM file containing reads to basecall")
    parser.add_argument(
        "fasta_files", action=FileExists,
        help="One or more FASTA files containing reference sequences")
    parser.add_argument(
        "ml_tag", action=FileExists,
        help="One or more FASTA files containing reference sequences")
    parser.add_argument(
        "output",
        help="Output directory to write basecalled sequences to")
    parser.add_argument(
        "--can",default = "C",
        help="The the label for target base in canonical alphebet(ACGT)")
    parser.add_argument(
        "--mod",default = "m",
        help="The the label for target base in modification alphebet(e.g y(m6A), x(m1A))")
    return parser



def add_mod_tage(bam_file, fasta_file,ml_file, mod_bam_file,can_base="C",mod_base="m"):
    bam = pysam.AlignmentFile(bam_file,"rb")
    mod_bam = pysam.AlignmentFile(mod_bam_file,\
        "wb", header = bam.header)
    fasta = pysam.FastaFile(fasta_file)
    ml = pd.read_table(ml_file,header =None,  index_col = 0)
    # can_base = "C"
    # mod_base = "m"
    mods = []
    num = 0
    reads = list(bam.fetch())
    qnames = [i.qname for i in reads]
    for ref in qnames:
        if ref not in qnames: continue
        mm_tag, ml_tag = "", array.array("B")
        try:
            qseq = fasta.fetch(ref)
        except (KeyError, FileNotFoundError, ValueError):
            continue

        # qseq = fasta.fetch(ref)
        mpos = np.where(np.array(list(qseq)) == mod_base)[0]
        if len(mpos)==0:  
            mod_bam.write(reads[qnames.index(ref)])
            continue
        can_base_mod_poss = np.cumsum([1 if b == can_base else 0 for b in qseq.replace(mod_base,can_base)])[mpos]-1
        # print(can_base_mod_poss)
        mod_gaps = ",".join(
            map(str, np.diff(np.insert(can_base_mod_poss, 0, -1)) -1)
        )
        # print(can_base_mod_poss)
        if qnames.count(ref) == 0: continue
        read = reads[qnames.index(ref)]
        if read.is_forward:
            mm_tag += f"{can_base}+{mod_base},{mod_gaps};"
        else:
            mm_tag += f"{can_base}-{mod_base},{mod_gaps};"
        try:
            mls = ast.literal_eval(ml.loc[ref][1])
        except KeyError:
            continue
        for ll in mls:
            ml_tag.extend([ll])
        # ml_tag.extend([0])
        if(len(ml_tag)/len(mpos)!=1): 
            mod_bam.write(read)
            continue
        # print(len(mpos))
        read.set_tag("ML", ml_tag)
        read.set_tag("MM",mm_tag)
        mod_bam.write(read)
        num+=1
    mod_bam.close()
    return mod_bam

def main():
    args = get_parser().parse_args()
    add_mod_tage(args.bam_file,args.fasta_files,args.ml_tag,args.output,args.can,args.mod)

if __name__ == '__main__':
    main()
