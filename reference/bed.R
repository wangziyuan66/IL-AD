fa <- read.table("./dna.fa", quote="\"", comment.char="", stringsAsFactors=FALSE)$V1

bed <- NULL
for (i in seq(1, length(fa), 2)){
  chr <- gsub(">", "", fa[i])
  pos <- which(unlist(strsplit(fa[i+1], split = "")) == "A")
  bed <- rbind(bed, cbind(rep(chr, length(pos)), pos))}
dimnames(bed) <- NULL
write.table(bed, file = "./dna_a.bed", quote = F, row.names = F, col.names = F)

bed <- NULL
for (i in seq(1, length(fa), 2)){
  chr <- gsub(">", "", fa[i])
  pos <- which(unlist(strsplit(fa[i+1], split = "")) == "T")
  bed <- rbind(bed, cbind(rep(chr, length(pos)), pos))}
dimnames(bed) <- NULL
write.table(bed, file = "./dna_t.bed", quote = F, row.names = F, col.names = F)

bed <- NULL
for (i in seq(1, length(fa), 2)){
  chr <- gsub(">", "", fa[i])
  pos <- which(unlist(strsplit(fa[i+1], split = "")) == "G")
  bed <- rbind(bed, cbind(rep(chr, length(pos)), pos))}
dimnames(bed) <- NULL
write.table(bed, file = "./dna_g.bed", quote = F, row.names = F, col.names = F)

bed <- NULL
for (i in seq(1, length(fa), 2)){
  chr <- gsub(">", "", fa[i])
  pos <- which(unlist(strsplit(fa[i+1], split = "")) == "C")
  bed <- rbind(bed, cbind(rep(chr, length(pos)), pos))}
dimnames(bed) <- NULL
write.table(bed, file = "./dna_c.bed", quote = F, row.names = F, col.names = F)

fa <- read.table("./rna.fa", quote="\"", comment.char="", stringsAsFactors=FALSE)$V1

bed <- NULL
for (i in seq(1, length(fa), 2)){
  chr <- gsub(">", "", fa[i])
  pos <- which(unlist(strsplit(fa[i+1], split = "")) == "A")
  bed <- rbind(bed, cbind(rep(chr, length(pos)), pos))}
dimnames(bed) <- NULL
write.table(bed, file = "./rna_a.bed", quote = F, row.names = F, col.names = F)

bed <- NULL
for (i in seq(1, length(fa), 2)){
  chr <- gsub(">", "", fa[i])
  pos <- which(unlist(strsplit(fa[i+1], split = "")) == "T")
  bed <- rbind(bed, cbind(rep(chr, length(pos)), pos))}
dimnames(bed) <- NULL
write.table(bed, file = "./rna_t.bed", quote = F, row.names = F, col.names = F)

bed <- NULL
for (i in seq(1, length(fa), 2)){
  chr <- gsub(">", "", fa[i])
  pos <- which(unlist(strsplit(fa[i+1], split = "")) == "G")
  bed <- rbind(bed, cbind(rep(chr, length(pos)), pos))}
dimnames(bed) <- NULL
write.table(bed, file = "./rna_g.bed", quote = F, row.names = F, col.names = F)

bed <- NULL
for (i in seq(1, length(fa), 2)){
  chr <- gsub(">", "", fa[i])
  pos <- which(unlist(strsplit(fa[i+1], split = "")) == "C")
  bed <- rbind(bed, cbind(rep(chr, length(pos)), pos))}
dimnames(bed) <- NULL
write.table(bed, file = "./rna_c.bed", quote = F, row.names = F, col.names = F)
