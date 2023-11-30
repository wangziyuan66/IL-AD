# IL-AD

We leverage machine learning approaches to adapt nanopore sequencing basecallers for nucleotide modification detection. We first apply the incremental learning technique to improve the basecalling of modification-rich sequences, which are usually of high biological interests. With sequence backbones resolved, we further run anomaly detection on individual nucleotides to determine their modification status. By this means, our pipeline promises the single-molecule, single-nucleotide and sequence context-free detection of modifications. 

## Pre-request

samtools
taiyaki

## Installation

## Usage

## Example

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
