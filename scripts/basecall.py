#!/usr/bin/env python3
import argparse
import sys
import time
from multiprocessing import Pool

import torch
from remora.activations import swish
import torch.nn as nn
# from packaging import version as packaging_version
from ont_fast5_api import fast5_interface
from taiyaki import basecall_helpers, decodeutil, fast5utils, helpers, qscores
from taiyaki.cmdargs import (AutoBool, FileExists, NonNegative,
                             ParseToNamedTuple, Positive)
from taiyaki.common_cmdargs import add_common_command_args
from taiyaki.decode import flipflop_make_trans, flipflop_viterbi
from taiyaki.flipflopfings import nstate_flipflop, path_to_str
from taiyaki.helpers import (Progress, guess_model_stride, load_model,
                             open_file_or_stdout)
from taiyaki.maths import med_mad
from taiyaki.prepare_mapping_funcs import get_per_read_params_dict_from_tsv
from taiyaki.signal import Signal
from utils import  index_hanlder
import numpy as np
from OCNN_basecall import basecaller
# pyright: reportGeneralTypeIssues=false


R_TRED = 22.9130

class OCNN(torch.nn.Module):

    def __init__(self, size, num_out = 1, kmer_len=7):
        super(OCNN, self).__init__()

        self.sig_conv1 = nn.Conv1d(1, 4, 11, bias = False)
        self.sig_conv2 = nn.Conv1d(4, 16, 11, bias = False)
        self.sig_conv3 = nn.Conv1d(16, size, 9, 3, bias = False)

        self.seq_conv1 = nn.Conv1d(kmer_len * 4, 16, 11, bias = False)
        self.seq_conv2 = nn.Conv1d(16, 32, 11, bias = False)
        self.seq_conv3 = nn.Conv1d(32, size, 9, 3, bias = False)

        self.merge_conv1 = nn.Conv1d(size * 2, size, 5, bias = False)
        self.merge_conv2 = nn.Conv1d(size, size, 5, bias = False)

        self.merge_conv3 = nn.Conv1d(size, size, 3, stride=2, bias = False)
        self.merge_conv4 = nn.Conv1d(size, size, 3, stride=2, bias = False)

        self.fc = nn.Linear(size * 3, num_out, bias = False)

        self.sig_bn1 = nn.BatchNorm1d(4)
        self.sig_bn2 = nn.BatchNorm1d(16)
        self.sig_bn3 = nn.BatchNorm1d(size)

        self.seq_bn1 = nn.BatchNorm1d(16)
        self.seq_bn2 = nn.BatchNorm1d(32)
        self.seq_bn3 = nn.BatchNorm1d(size)

        self.merge_bn1 = nn.BatchNorm1d(size)
        self.merge_bn2 = nn.BatchNorm1d(size)
        self.merge_bn3 = nn.BatchNorm1d(size)
        self.merge_bn4 = nn.BatchNorm1d(size)
        

    def forward(self, sigs, seqs):
        sigs = sigs.permute(1, 2, 0)
        seqs = seqs.permute(0, 2, 1)
        sigs_x = swish(self.sig_bn1(self.sig_conv1(sigs)))
        sigs_x = swish(self.sig_bn2(self.sig_conv2(sigs_x)))
        sigs_x = swish(self.sig_bn3(self.sig_conv3(sigs_x)))

        seqs_x = swish(self.seq_bn1(self.seq_conv1(seqs)))
        seqs_x = swish(self.seq_bn2(self.seq_conv2(seqs_x)))
        seqs_x = swish(self.seq_bn3(self.seq_conv3(seqs_x)))
        z = torch.cat((sigs_x, seqs_x), 1)

        z = swish(self.merge_bn1(self.merge_conv1(z)))
        z = swish(self.merge_bn2(self.merge_conv2(z)))
        z = swish(self.merge_bn3(self.merge_conv3(z)))
        z = swish(self.merge_bn4(self.merge_conv4(z)))

        z = torch.flatten(z, start_dim=1)
        z = self.fc(z)

        return z

def get_parser():
    parser = argparse.ArgumentParser(
        description="Basecall reads using a taiyaki model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_common_command_args(
        parser, """alphabet device input_folder
        input_strand_list jobs limit output quiet
        recursive version""".split())

    parser.add_argument(
        '--beam', default=None, metavar=('width', 'guided'), nargs=2,
        type=(int, bool), action=ParseToNamedTuple,  # type: ignore
        help='Use beam search decoding')
    parser.add_argument(
        "--chunk_size", type=Positive(int), metavar="blocks",
        default=basecall_helpers._DEFAULT_CHUNK_SIZE,
        help="Size of signal chunks sent to GPU is chunk_size * model stride")
    parser.add_argument(
        '--fastq', default=False, action=AutoBool,
        help='Write output in fastq format (default is fasta)')
    parser.add_argument(
        "--max_concurrent_chunks", type=Positive(int), default=128,
        help="Maximum number of chunks to call at "
        "once. Lower values will consume less (GPU) RAM.")
    parser.add_argument(
        "--overlap", type=NonNegative(int), metavar="blocks",
        default=basecall_helpers._DEFAULT_OVERLAP,
        help="Overlap between signal chunks sent to GPU")
    parser.add_argument(
        '--posterior', default=True, action=AutoBool,
        help='Use posterior-viterbi decoding')
    parser.add_argument(
        "--qscore_offset", type=float, default=0.0,
        help="Offset to apply to q scores in fastq (after scale)")
    parser.add_argument(
        "--qscore_scale", type=float, default=1.0,
        help="Scaling factor to apply to q scores in fastq")
    parser.add_argument(
        '--reverse', default=False, action=AutoBool,
        help='Reverse sequences in output')
    parser.add_argument(
        '--scaling', action=FileExists, default=None,
        help='Path to TSV containing per-read scaling params')
    parser.add_argument(
        '--temperature', default=1.0, type=float,
        help='Scaling factor applied to network outputs before decoding')
    parser.add_argument(
        '--filter_prob', default=0.825, type=float,
        help='Scaling factor to filter low quality m')
    parser.add_argument(
        "can_model", action=FileExists,
        help="Model checkpoint file to use for basecalling")
    parser.add_argument(
        "mod_model", action=FileExists,
        help="Model checkpoint file to use for basecalling")
    parser.add_argument(
        "ae_model", action=FileExists,
        help="Model checkpoint file to mod inference for basecalling")

    return parser


def med_mad_norm(x, dtype='f4'):
    """ Normalise a numpy array using median and MAD

    Args:
        x (:class:`ndarray`): 1D array containing values to be normalised.
        dtype (str or :class:`dtype`): dtype of returned array.

    Returns:
        :class:`ndarray`:  Array of same shape as `x` and dtype `dtype`
            contained normalised values.
    """
    med, mad = med_mad(x)
    normed_x = (x - med) / mad
    return normed_x.astype(dtype)


def get_signal(read_filename, read_id):
    """ Get raw signal from read tuple

    Args:
        read_filename (str): Name of file from which to read signal.
        read_id (str): ID of signal to read from `read_filename`

    Returns:
        class:`ndarray`: 1D array containing signal.

        If unable to read signal from file, `None` is returned.
    """
    try:
        with fast5_interface.get_fast5_file(read_filename, 'r') as f5file:
            read = f5file.get_read(read_id)
            sig = Signal(read)
            return sig.current

    except Exception as e:
        sys.stderr.write(
            'Unable to obtain signal for {} from {}.\n{}\n'.format(
                read_id, read_filename, repr(e)))
        return None


def worker_init(device, canmodelname, modmodelname, chunk_size, overlap,
                read_params, alphabet, max_concurrent_chunks,
                fastq, qscore_scale, qscore_offset, beam, posterior,
                temperature,filter_prob,autoencoder_cpt):
    global all_read_params
    global process_read_partial

    all_read_params = read_params
    device = helpers.set_torch_device(device)
    can_model,mod_model = load_model(canmodelname).to(device),load_model(modmodelname).to(device)
    stride = guess_model_stride(can_model)
    chunk_size = chunk_size * stride
    overlap = overlap * stride

    n_can_base = 4
    n_can_state = nstate_flipflop(n_can_base)

    def process_read_partial(read_filename, read_id, read_params):
        res = process_read(read_filename, read_id,
                           (can_model,mod_model,autoencoder_cpt), chunk_size, overlap, read_params,
                           n_can_state, stride, alphabet,
                           max_concurrent_chunks, fastq, qscore_scale,
                           qscore_offset, beam, posterior, temperature,filter_prob)
        return (read_id, *res)


def worker(args):
    read_filename, read_id = args
    read_params = all_read_params[
        read_id] if read_id in all_read_params else None
    return process_read_partial(read_filename, read_id, read_params)


def process_read(
        read_filename, read_id, model, chunk_size, overlap, read_params,
        n_can_state, stride, alphabet, max_concurrent_chunks,
        fastq=False, qscore_scale=1.0, qscore_offset=0.0, beam=None,
        posterior=True, temperature=1.0, filter_prob = 0.825):
    """Basecall a read, dividing the samples into chunks before applying the
    basecalling network and then stitching them back together.

    Args:
        read_filename (str): filename to load data from.
        read_id (str): id used in comment line in fasta or fastq output.
        model (:class:`nn.Module`): Taiyaki network.
        chunk_size (int): chunk size, measured in samples.
        overlap (int): overlap between chunks, measured in samples.
        read_params (dict str -> T): reads specific scaling parameters,
            including 'shift' and 'scale'.
        n_can_state (int): number of canonical flip-flop transitions (40 for
            ACGT).
        stride (int): stride of basecalling network (measured in samples)
        alphabet (str): Alphabet (e.g. 'ACGT').
        max_concurrent_chunks (int): max number of chunks to basecall at same
            time (having this limit prevents running out of memory for long
            reads).
        fastq (bool): generate fastq file with q scores if this is True,
            otherwise generate fasta.
        qscore_scale (float): Scaling factor for Q score calibration.
        qscore_offset (float): Offset for Q score calibration.
        beam (None or NamedTuple): Use beam search decoding
        posterior (bool): Decode using posterior probability of transitions
        temperature (float): Multiplier for network output

    Returns:
        tuple of str and str and int: strings containing the called bases and
            their associated Phred-encoded quality scores, and the number of
            samples in the read (before chunking).

        When `fastq` is False, `None` is returned instead of a quality string.
    """
    signal = get_signal(read_filename, read_id)
    can_model, mod_model, autoencoder_cpt = model
    if signal is None:
        return None, None, 0
    if can_model.metadata['reverse']:
        signal = signal[::-1]

    if read_params is None:
        normed_signal = med_mad_norm(signal)
    else:
        normed_signal = (signal - read_params['shift']) / read_params['scale']

    chunks, chunk_starts, chunk_ends = basecall_helpers.chunk_read(
        normed_signal, chunk_size, overlap)

    qstring = None
    with torch.no_grad():
        device = next(can_model.parameters()).device
        chunks = torch.tensor(chunks, device=device)
        trans = []
        mods = []
        for some_chunks in torch.split(chunks, max_concurrent_chunks, 1):
            mod_outputs = mod_model(some_chunks)
            outputs = mod_outputs
            trans.append(outputs[:, :, :n_can_state])
            mods.append(outputs[:, :, n_can_state:])
        trans = torch.cat(trans, 1) * temperature

        if posterior:
            trans = (flipflop_make_trans(trans) + 1e-8).log()

        if beam is not None:
            trans = basecall_helpers.stitch_chunks(trans, chunk_starts,
                                                   chunk_ends, stride)
            best_path, score = decodeutil.beamsearch(trans.cpu().numpy(),
                                                     beam_width=beam.width,
                                                     guided=beam.guided)
        else:
            _, _, chunk_best_paths = flipflop_viterbi(trans)
            best_path = basecall_helpers.stitch_chunks(
                chunk_best_paths, chunk_starts, chunk_ends,
                stride).cpu().numpy()

        if fastq:
            chunk_errprobs = qscores.errprobs_from_trans(trans,
                                                         chunk_best_paths) # type: ignore
            errprobs = basecall_helpers.stitch_chunks(
                chunk_errprobs, chunk_starts, chunk_ends, stride)
            qstring = qscores.path_errprobs_to_qstring(errprobs, best_path,
                                                       qscore_scale,
                                                       qscore_offset)

    # This makes our basecalls agree with Guppy's, and removes the
    # problem that there is no entry transition for the first path
    # element, so we don't know what the q score is.
    chunk_errprobs = qscores.errprobs_from_trans(trans,
                                                chunk_best_paths) # type: ignore
    errprobs = basecall_helpers.stitch_chunks(
                chunk_errprobs, chunk_starts, chunk_ends, stride)
    errprobs = errprobs[1:][best_path[1:] != best_path[:-1]]
    basecall = list(path_to_str(best_path, alphabet=alphabet,
                           include_first_source=False))
    ref2sig = np.where(best_path[1:] != best_path[:-1])[0]*stride-1
    ocnn = torch.load(autoencoder_cpt,"cuda")
    basecall = basecaller(np.array(basecall), normed_signal, ref2sig, ocnn)
    return basecall, qstring, len(signal)



def main():
    args = get_parser().parse_args()

    # TODO convert to logging

    sys.stderr.write("* Initializing reads file search.\n")
    fast5_reads = fast5utils.iterate_fast5_reads(
        args.input_folder, limit=args.limit,
        strand_list=args.input_strand_list, recursive=args.recursive)

    if args.scaling is not None:
        sys.stderr.write(
            "* Loading read scaling parameters from {}.\n".format(
                args.scaling))
        all_read_params = get_per_read_params_dict_from_tsv(args.scaling)
        input_read_ids = frozenset(rec[1] for rec in fast5_reads)
        scaling_read_ids = frozenset(all_read_params.keys()) # type: ignore
        sys.stderr.write("* {} / {} reads have scaling information.\n".format(
            len(input_read_ids & scaling_read_ids), len(input_read_ids)))
        fast5_reads = [rec for rec in fast5_reads if rec[
            1] in scaling_read_ids]
    else:
        all_read_params = {}

    sys.stderr.write("* Calling reads.\n")
    nbase, ncalled, nread, nsample = 0, 0, 0, 0
    t0 = time.time()
    progress = Progress(quiet=args.quiet)
    startcharacter = '@' if args.fastq else '>'
    initargs = [args.device, args.can_model, args.mod_model, args.chunk_size, args.overlap,
                all_read_params, args.alphabet,
                args.max_concurrent_chunks, args.fastq, args.qscore_scale,
                args.qscore_offset, args.beam, args.posterior,
                args.temperature, args.filter_prob,args.ae_model]
    pool = Pool(args.jobs, initializer=worker_init, initargs=initargs)
    with open_file_or_stdout(args.output) as fh:
        for read_id, basecall, qstring, read_nsample in \
                pool.imap_unordered(worker, fast5_reads):
            if basecall is not None and len(basecall) > 0:
                fh.write("{}{}\n{}\n".format(
                    startcharacter, read_id,
                    basecall[::-1] if args.reverse else basecall))
                nbase += len(basecall)
                ncalled += 1
                if args.fastq:
                    fh.write("+\n{}\n".format(
                        qstring[::-1] if args.reverse else qstring)) # type:ignore

            nread += 1
            nsample += read_nsample
            progress.step()
    total_time = time.time() - t0

    sys.stderr.write(
        "* Called {} reads in {:.2f}s\n".format(nread, int(total_time)))
    sys.stderr.write(
        "* {:7.2f} kbase / s\n".format(nbase / total_time / 1000.0))
    sys.stderr.write(
        "* {:7.2f} ksample / s\n".format(nsample / total_time / 1000.0))
    sys.stderr.write("* {} reads failed.\n".format(nread - ncalled))
    return


if __name__ == '__main__':
    main()
