# IL-AD

We leverage machine learning approaches to adapt nanopore sequencing basecallers for nucleotide modification detection. We first apply the incremental learning technique to improve the basecalling of modification-rich sequences, which are usually of high biological interests. With sequence backbones resolved, we further run anomaly detection on individual nucleotides to determine their modification status. By this means, our pipeline promises the single-molecule, single-nucleotide and sequence context-free detection of modifications. 


<!--  <p align='center'><img src="https://www.pharmacy.arizona.edu/sites/default/files/styles/az_medium/public/2023-05/HD3.png?itok=EBqnN-7q" width = "140" height = "200" alt="图片名称" align=center /></p> -->

## Dependencies

samtools: https://github.com/samtools/samtools

taiyaki: https://github.com/nanoporetech/taiyaki/tree/master/taiyaki

## Usage

### Incremental Learning

Training Process
```sh
python train.py model_template.py pretained_model.checkpoint input.hdf5 --device cuda:0 --outdir path/to/output \
--save_every epochs --niteration niterations --lr_max lr_max --lambda lambda --min_sub_batch_size batchsize
```

### Basecalling 

You should then be able to export your checkpoint to json (using bin/dump_json.py in [taiyaki](https://github.com/nanoporetech/taiyaki/tree/master)) that can be used to basecall with Guppy.

See Guppy documentation for more information on how to do this.

Key options include selecting the Guppy config file to be appropriate for your application, and passing the complete path of your .json file.

For example:

```sh
guppy_basecaller --input_path /path/to/input_reads --save_path /path/to/save_dir --config dna_r9.4.1_450bps_flipflop.cfg --model path/to/model.json --device cuda:1
```

### Anomaly Detection

```sh
python context_abnormal.py --device cuda:0 model_template.py initial_checkpoint.checkpoint \
input.hdf5 --outdir path/to/output --save_every save_every --niteration niteration  --sig_win_len n --min_sub_batch_size BATCHSIZE --right_len m --can BASE
```

**sig_win_len** and **right_len** are $n$ and $m$ we mentioned in the manuscript.

### Modification Inference

```sh
python modification_inference.py mapped_reads.hdf5 can.checkpoint mod.checkpoint CANBASE MODBASE path/to/output/fasta --can_base_idx can_base_idx --type rna/dna --length n --right_len m
```

can_base_idx means the mapping from alphabet labels in the hdf5 files to canonical labels(**default** 0123 ATGC). For example, for a hdf5 whose labels are ACm(5mC)h(5hmC)GT, the can_base_idx should be 01123.

After generating fasta file containing mod base we can use `modbase_tag.py` to write ML/MM tag into the bam file. We can merge ML/MM tags from different bams containing different kinds of modifications using `merge_bam.py`.

After these process, we can visualize the per site results from the bam file using `samtools mpileup`.

## Miscellanies

### RNA splicing

If you are dealing with mRNA data and your reference is the genome reference fasta file, you need to pay more attention on RNA splicing.

If you want to create training hdf5 files or mapped reads for modification inference according to https://github.com/nanoporetech/taiyaki/tree/master#steps-from-fast5-files-to-basecalling, you should replace the `bin/get_ref_from_sam.py` in taiyaki with `scripts/rna_process/get_ref_from_sam_rna.py` in our project.

### tRNA specific processing

If you want to implement iterative labeling, you need to train model with `bin/train_flip_flop.py` in taiyaki. However, if you need to handle with tRNA data, please replace it with `scripts/trna/train_flip_flop.py` in our project.


## Results

![curlcake](images/rna.jpeg)

<p align='center'><b>Accuracy on RNA  synthesized oligo (canonical, fully modified 5mC, 6mA, 1mA)</b></p>

![curlcake](images/trna.jpeg)

<p align='center'><b>Incremental learning benefit the reads mappability of tRNA</b></p>

## Pretrained model

For DNA/RNA curlcake, the model after incremental learning and the model for modification inference are available at: .

## Citation & Contact

Ziyuan Wang PhD student in University of Arizona, R. Ken Coit College of Pharmcay, email:princezwang@arizona.edu
