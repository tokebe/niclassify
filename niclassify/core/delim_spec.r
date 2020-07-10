# SCRIPT EXPECTS 2 ARGUMENTS: FNAME_IN, FNAME_OUT
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
Dist <- dist.dna(infasta)

# make a UPGMA Tree
UPGMA <- upgma(Dist)

# Run splits to delimit species
GMYC <- gmyc(UPGMA)

# Save results to given file
write.csv(spec.list(GMYC), args[2], row.names = FALSE)
