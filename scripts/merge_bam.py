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
        "output",
        help="Input BAM file containing reads to basecall")
    parser.add_argument(
        "bam_files",nargs = "+",
        help="Input BAM file containing reads to basecall")
    return parser
def merge(bam_files,output_bam):
    ref_bam = pysam.AlignmentFile(bam_files[0],"rb")
    ref_reads = list(ref_bam.fetch())
    bams = [list(pysam.AlignmentFile(i,"rb").fetch()) for i in bam_files]
    merge_bam = pysam.AlignmentFile(output_bam,\
        "wb", header = ref_bam.header)
    for i in range(len(bams[0])):
        print(len(bams[0]))
        print(len(bams[1]))
        print(len(bams[2]))
        print(len(bams[3]))
        # if i<46:
        #     ml_tags = [j[i].get_tag("ML") for j in bams if j[i].has_tag("ML") ]
        #     mm_tags = [j[i].get_tag("MM") for j in bams if j[i].has_tag("MM") ]
        # else:
        ml_tags = [j[i].get_tag("ML") for j in bams if j[i].has_tag("ML") ]
        mm_tags = [j[i].get_tag("MM") for j in bams if j[i].has_tag("MM") ]
        # print(bams[0][0].get_tag("ML").typecode)
        mm_tag, ml_tag = "", array.array(bams[0][0].get_tag("ML").typecode)
        for itm in mm_tags: mm_tag += itm
        for itm in ml_tags: 
            ml_tag += itm[:len(itm)]
        # ml_tag.extend([0])
        # ml_tag.extend([0])
        read = ref_reads[i]
        if len(ml_tag)==0: 
            merge_bam.write(read)
            continue
        read.set_tag("ML", ml_tag)
        read.set_tag("MM", mm_tag)
        merge_bam.write(read)
    merge_bam.close()

def main():
    args = get_parser().parse_args()
    merge(args.bam_files,args.output)

main()