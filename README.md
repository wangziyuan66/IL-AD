# IL-AD

We leverage machine learning approaches to adapt nanopore sequencing basecallers for nucleotide modification detection. We first apply the incremental learning technique to improve the basecalling of modification-rich sequences, which are usually of high biological interests. With sequence backbones resolved, we further run anomaly detection on individual nucleotides to determine their modification status. By this means, our pipeline promises the single-molecule, single-nucleotide and sequence context-free detection of modifications. 

## Pre-request

samtools: https://github.com/samtools/samtools

taiyaki: https://github.com/nanoporetech/taiyaki/tree/master/taiyaki

## Usage

### Incremental Learning

Training Process
```sh
python train.py model_template.py pretained_model.checkpoint input.hdf5 --device cuda:0 --outdir path/to/output \
--save_every epochs --niteration niterations --lr_max lr_max --lambda lambda --min_sub_batch_size batchsize
```

Basecall: 

You should then be able to export your checkpoint to json (using bin/dump_json.py in [taiyaki](https://github.com/nanoporetech/taiyaki/tree/master)) that can be used to basecall with Guppy.

See Guppy documentation for more information on how to do this.

Key options include selecting the Guppy config file to be appropriate for your application, and passing the complete path of your .json file.

For example:

```sh
guppy_basecaller --input_path /path/to/input_reads --save_path /path/to/save_dir --config dna_r9.4.1_450bps_flipflop.cfg --model path/to/model.json --device cuda:1
```

### Anomaly Detection Training

```py
python context_abnormal.py --device cuda:0 model_template.py initial_checkpoint.checkpoint \
input.hdf5 --outdir path/to/output --save_every save_every --niteration niteration  --sig_win_len n --min_sub_batch_size BATCHSIZE --right_len m --can BASE
```

**sig_win_len** and **right_len** are $n$ and $m$ we mentioned in the manuscript.

### Modification Inference

### RNA splicing

### tRNA specific processing


## Usage

## Example

![curlcake](images/rna.jpeg)

<p align='center'><b>Accuracy on RNA  synthesized oligo (canonical, fully modified 5mC, 6mA, 1mA)</b></p>

![curlcake](images/trna.jpeg)

<p align='center'><b>Incremental learning benefit the reads mappability of tRNA</b></p>

## Data Code

The yeast native tRNA nanopore sequencing data was downloaded from European National Archive (ENA) under accession number PRJEB55684. Corresponding reference genome and modification annotation were downloaded from https://github.com/novoalab/Nano-tRNAseq/tree/main/ref. 

The CpG and GpC methylated, and the unmodified E.coli genomic DNA nanopore sequencing datasets were downloaded from https://sra-pub-src-2.s3.amazonaws.com/SRR11953238/ecoli_CpGGpC.fast5.tgz.2 and https://sra-pub-src-2.s3.amazonaws.com/SRR11953241/ecoli_Unmethylated.fast5.tgz.1, respectively.

Corresponding reference genome was downloaded from https://www.ncbi.nlm.nih.gov/nuccore/U00096. The human HEK293 cell line native mRNA nanopore sequencing data was downloaded from ENA under accession number PRJEB40872. 

Corresponding m1A and m6A ground-truth annotations were downloaded from the Supplementary Table 2 of 35 and the Supplementary Data 4 of 32, respectively. 

The mouse ESC native mRNA nanopore sequencing data was downloaded from NCBI Sequence Read Archive (SRA) under the accession number SRP166020. Corresponding m6A ground-truth annotations were downloaded from NCBI Gene Expression Omnibus (GEO) under the accession number GSM2300431. DNA and RNA oligo datasets were deposited at SRA under the accession number. 

ONT DNA and RNA kmer models were downloaded from https://github.com/nanoporetech/kmer_models. 

Original DNA and RNA basecalling models were downloaded from https://github.com/nanoporetech/taiyaki/blob/master/models/mLstm_flipflop_model_r941_DNA.checkpoint and https://s3-eu-west-1.amazonaws.com/ont-research/taiyaki_modbase.tar.gz, respectively. 

DNA and RNA taiyaki model templates were downloaded from https://github.com/nanoporetech/taiyaki/tree/master/models.

## Citation & Contact

Ziyuan Wang PhD student in University of Arizona, R. Ken Coit College of Pharmcay, email:princezwang@arizona.edu
