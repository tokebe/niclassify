#!/usr/bin/env Rscript
# SCRIPT EXPECTS 2 ARGUMENTS: FNAME_IN, FNAME_OUT
# Script does not modify original file (unless both arguments are same)

library(ape)


# read in command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Load given FASTA formatted sequence file
alignment <- read.FASTA(args[1], type = "DNA")

# Loop through codon start offsets and find the first one that works
# Then trim to be a multiple of 3 length from there

for (offset in 1:4) {
    alignmentAA <- trans(alignment, codonstart = offset)
    # check for stop codons
    if (!(TRUE %in% grepl("*", alignmentAA[[names(alignmentAA)[1]]], fixed = TRUE))) {
        break
    }
    # check if we've surpassed the potential 3 codon offset (meaning file is somehow broken)
    if (offset == 4) {
        writeLines("ERROR", file(args[2]))
    }
}

# calculate end to get sequence length multiple of 3
seq_len <- length(alignment[[names(alignment)[1]]])
offset_0 <- offset - 1

end <- seq_len - ((seq_len - offset_0) %% 3)

# save with trimming to (potentially) new offset
write.dna(
  as.matrix(alignment)[, offset:end],
  file = args[2],
  format="fasta",
  nbcol=-1,
  colsep="")
