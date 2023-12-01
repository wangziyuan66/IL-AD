#!/usr/bin/env python3
import argparse
import pysam
import sys

from taiyaki.bio import complement, fasta_file_to_dict, reverse_complement
from taiyaki.cmdargs import AutoBool, proportion, FileExists
from taiyaki.common_cmdargs import add_common_command_args
from taiyaki.fileio import readtsv
from taiyaki.helpers import open_file_or_stdout

# Base complements
_COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'X': 'X', 'N': 'N','@':'y','y':'T',
                'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'x': 'x', 'n': 'n',
                '-': '-'}

def get_parser():
    parser = argparse.ArgumentParser(
        description='Extract reference sequence for each read from a SAM ' +
        'alignment file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_common_command_args(parser, ["output"])

    parser.add_argument(
        '--complement', default=False, action=AutoBool,
        help='Complement all reference sequences')
    parser.add_argument(
        '--input_strand_list', default=None, action=FileExists,
        help='Strand summary file containing subset')
    parser.add_argument(
        '--min_coverage', metavar='proportion', default=0.6, type=proportion,
        help='Ignore reads with alignments shorter than min_coverage * ' +
        'read length')
    parser.add_argument(
        '--pad', type=int, default=0,
        help='Number of bases by which to pad reference sequence')
    parser.add_argument(
        '--reverse', default=False, action=AutoBool,
        help='Reverse all reference sequences (for RNA)')
    parser.add_argument(
        '--mod', default="y", 
        help='Reverse all reference sequences (for RNA)')
    parser.add_argument(
        '--mod_num', default=1, type=int,
        help='Reverse all reference sequences (for RNA)')
    parser.add_argument(
        'reference', action=FileExists,
        help="Genomic references that reads were aligned against")
    parser.add_argument(
        'input', metavar='input.sam', nargs='+',
        help="SAM or BAM file(s) containing read alignments to reference")

    return parser


def get_refs(sam, ref_seq_dict, min_coverage=0.6, pad=0, strand_list=None):
    """Read alignments from sam file and return accuracy metrics
    """
    with pysam.Samfile(sam, 'r') as sf:
        for read in sf:
            # print(read.flag)
            if read.flag != 0 and read.flag != 16:
                #  No match, or orientation is not 0 or 16
                continue
            if strand_list is not None and read.query_name not in strand_list:
                continue

            coverage = float(read.query_alignment_length) / read.query_length
            if coverage < min_coverage:
                #  Failed coverage criterian
                continue

            read_ref = ref_seq_dict.get(sf.references[read.reference_id], None)
            if read_ref is None:
                #  Reference mapped against does not exist in ref_seq_dict
                continue
            
            reference_positions = [i[1] for i in read.get_aligned_pairs(matches_only=True)]
            cigar_tuples = read.cigartuples
            reference_sequence = ''
            ref_position = read.reference_start

            for cigar_type, cigar_length in cigar_tuples:
                if cigar_type in [0, 2, 7, 8]:  # Matches, sequence, and equal (M, =, X in CIGAR)
                    reference_sequence += ''.join([read_ref[ref_pos:ref_pos + 1] for ref_pos in range(ref_position, ref_position + cigar_length)])
                    ref_position += cigar_length
                if cigar_type in [3]:  # Matches, sequence, and equal (M, =, X in CIGAR)
                    ref_position += cigar_length

            read_ref = reference_sequence.upper().replace("Y","y")

            if read.flag == 16:
                #  Mapped to reverse strand
                read_ref = reverse_complement(read_ref,compdict=_COMPLEMENT)

            yield read.qname, read_ref.replace("Y","y").replace("@","T"),read.query
            # yield read.qname, read.query


def main():
    args = get_parser().parse_args()

    sys.stderr.write(
        "* Loading references (this may take a while for large genomes)\n")
    references = fasta_file_to_dict(args.reference, filter_ambig=False,flatten_ambig = False)

    if args.input_strand_list is None:
        strand_list = None
    else:
        strand_list = readtsv(args.input_strand_list,
                              fields=['read_id'])['read_id']
        sys.stderr.write(
            '* Strand list contains {} reads\n'.format(len(strand_list)))

    sys.stderr.write("* Extracting read references using SAM alignment\n")
    with open_file_or_stdout(args.output) as fh:
        for samfile in args.input:
            for name, read_ref, _ in get_refs(
                    samfile, references, args.min_coverage, args.pad,
                    strand_list=strand_list):
                print(read_ref.count(args.mod))
                if args.reverse:
                    read_ref = read_ref[::-1]
                if args.complement:
                    read_ref = complement(read_ref)
                if read_ref.count(args.mod) >= args.mod_num:
                    fasta = ">{}\n{}\n".format(name, read_ref)
                    fh.write(fasta)


if __name__ == '__main__':
    main()