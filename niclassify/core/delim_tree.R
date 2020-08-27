# SCRIPT EXPECTS 3 ARGUMENTS: FNAME_IN, FNAME_TREE_OUT, FNAME_OUT
# I have yet to figure out error handling, so please be gentle.

library(ape)
library(adephylo)
library(phytools)
library(phangorn)
library(splits)


# read in command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Load given FASTA formatted sequence file
infasta <- read.FASTA(args[1], type = "DNA")

# debug
# str(na.omit(infasta))

# Make a distance matrix from the alignment
Dist <- dist.hamming(infasta)

# make a UPGMA Tree
UPGMA <- upgma(Dist)

# Write tree to file
write.tree(UPGMA, file = args[2])

if (length(args) > 2) {
    # Run splits to delimit species
    GMYC <- gmyc(UPGMA)
    # Save results to given file
    write_delim(spec.list(GMYC), args[3], delim = "\t")
}
