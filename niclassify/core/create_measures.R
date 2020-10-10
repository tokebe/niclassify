# SCRIPT EXPECTS 3 ARGUMENTS: FNAME_IN, FNAME_SPECIES, FNAME_OUT
# I have yet to figure out error handling, so please be gentle.

library(ape)
library(adephylo)
library(phytools)
library(phangorn)
library(splits)
library(seqinr)
library(stringr) # string manipulation utilities
library(reshape2) # data frame manipulation utilities
library(tidyverse)

### Debug output redirection ###
fs = 0
# TODO actually implement debug so it can work on any system
while (file.exists(paste("C:/Users/J C/Documents/Github/niclassify/output/r-debug/log", fs, ".txt", sep=""))) {
  fs = fs + 1
}
logfile <- file(paste("C:/Users/J C/Documents/Github/niclassify/output/r-debug/log", fs, ".txt", sep=""))
sink(logfile, append=TRUE)
sink(logfile, append=TRUE, type="message")


# read in command-line arguments
args <- commandArgs(trailingOnly = TRUE)

### Load FASTA formated sequence file ###
seq_alignment<-read.FASTA(args[1], type="DNA")

## Pull in Species Assignments from FNAME_SPECIES ##
speciesNames <- read_tsv(args[2])

# Make DNA distance matrix
seqDNA_Dist<-dist.hamming(seq_alignment)

### Translate DNA sequence to Amino Acid Sequence ###
### Number should be updated based on the study organisms following https://www.ncbi.nlm.nih.gov/Taxonomy/taxonomyhome.html/index.cgi?chapter=cgencodes ####
seqAA <- trans(seq_alignment,5)

### Make a distance matrix from the AA alignment ###
seqAA_Dist <- dist.hamming(seqAA,ratio=TRUE)

### KaKs matrix in seqinr ###
dnafile <- read.alignment(file=args[1], format="fasta")

seqKaKs <- kaks(dnafile, verbose = FALSE, debug = FALSE, forceUpperCase = TRUE, rmgap = TRUE)

#### Matrix Analyses Below #################
### DNA Distance ###
dnaDist <- as.matrix(seqDNA_Dist)
dnaDist <- melt(dnaDist)

### Add species groups to distance table ###
dnaDist <- dnaDist %>%
  left_join(speciesNames, by = c("Var1" = "sample_name")) %>%
  rename(Var1_group = GMYC_spec) %>%
  left_join(speciesNames, by = c("Var2" = "sample_name")) %>%
  rename(Var2_group = GMYC_spec)

### Create summary statistics ###
dnaDistAggregated <- dnaDist %>%
  group_by(Var1_group) %>%
  summarise(
    dnaDist_mean = mean(value[Var1_group != Var2_group]),  # only where species group isn't the same
    dnaDist_med = median(value[Var1_group != Var2_group]),
    dnaDist_std = sd(value[Var1_group != Var2_group]),
    dnaDist_min = min(value[Var1_group != Var2_group]),
    dnaDist_max = max(value[Var1_group != Var2_group]),
    dnaSim_mean = mean(value[Var1_group == Var2_group]), # only where species group is the same
    dnaSim_med = median(value[Var1_group == Var2_group]),
    dnaSim_std = sd(value[Var1_group == Var2_group]),
    dnaSim_min = min(value[Var1_group == Var2_group]),
    dnaSim_max = max(value[Var1_group == Var2_group]),
  )

### Join with sample names in preparation for output ###
groups_metrics <- dnaDistAggregated %>%
  right_join(speciesNames, by = c("Var1_group" = "GMYC_spec")) %>%
  rename(species_group = Var1_group)


### AA Distance ###
aaDist <- as.matrix(seqAA_Dist)
aaDist <- melt(aaDist)

### Add species groups to distance table ###
aaDist <- aaDist %>%
  left_join(speciesNames, by = c("Var1" = "sample_name")) %>%
  rename(Var1_group = GMYC_spec) %>%
  left_join(speciesNames, by = c("Var2" = "sample_name")) %>%
  rename(Var2_group = GMYC_spec)

### Create summary statistics ###
aaDistAggregated <- aaDist %>%
  group_by(Var1_group) %>%
  summarise(
    aaDist_mean = mean(value[Var1_group != Var2_group]),
    aaDist_med = median(value[Var1_group != Var2_group]),
    aaDist_std = sd(value[Var1_group != Var2_group]),
    aaDist_min = min(value[Var1_group != Var2_group]),
    aaDist_max = max(value[Var1_group != Var2_group]),
    aaSim_mean = mean(value[Var1_group == Var2_group]),
    aaSim_med = median(value[Var1_group == Var2_group]),
    aaSim_std = sd(value[Var1_group == Var2_group]),
    aaSim_min = min(value[Var1_group == Var2_group]),
    aaSim_max = max(value[Var1_group == Var2_group]),
  )

### Join with sample names in preparation for output ###
groups_metrics <- aaDistAggregated %>%
  right_join(groups_metrics, by = c("Var1_group" = "species_group")) %>%
  rename(species_group = Var1_group)

### Ka Distance ###
if (!is.atomic(seqKaKs)) {
  kaDist <- as.matrix(seqKaKs$ka)
  kaDist <- melt(kaDist)

  ### Add species groups to distance table ###
  kaDist <- kaDist %>%
    left_join(speciesNames, by = c("Var1" = "sample_name")) %>%
    rename(Var1_group = GMYC_spec) %>%
    left_join(speciesNames, by = c("Var2" = "sample_name")) %>%
    rename(Var2_group = GMYC_spec)

  ### Create summary statistics ###
  kaDistAggregated <- kaDist %>%
    group_by(Var1_group) %>%
    summarise(
      kaDist_mean = mean(value[Var1_group != Var2_group]),
      kaDist_med = median(value[Var1_group != Var2_group]),
      kaDist_std = sd(value[Var1_group != Var2_group]),
      kaDist_min = min(value[Var1_group != Var2_group]),
      kaDist_max = max(value[Var1_group != Var2_group]),
      kaSim_mean = mean(value[Var1_group == Var2_group]),
      kaSim_med = median(value[Var1_group == Var2_group]),
      kaSim_std = sd(value[Var1_group == Var2_group]),
      kaSim_min = min(value[Var1_group == Var2_group]),
      kaSim_max = max(value[Var1_group == Var2_group]),
    )

  ### Join with sample names in preparation for output ###
  groups_metrics <- kaDistAggregated %>%
    right_join(groups_metrics, by = c("Var1_group" = "species_group")) %>%
    rename(species_group = Var1_group)

  ### Ks Distance ###
  ksDist <- as.matrix(seqKaKs$ks)
  ksDist <- melt(ksDist)

  ### Add species groups to distance table ###
  ksDist <- ksDist %>%
    left_join(speciesNames, by = c("Var1" = "sample_name")) %>%
    rename(Var1_group = GMYC_spec) %>%
    left_join(speciesNames, by = c("Var2" = "sample_name")) %>%
    rename(Var2_group = GMYC_spec)

  ### Create summary statistics ###
  ksDistAggregated <- ksDist %>%
    group_by(Var1_group) %>%
    summarise(
      ksDist_mean = mean(value[Var1_group != Var2_group]),
      ksDist_med = median(value[Var1_group != Var2_group]),
      ksDist_std = sd(value[Var1_group != Var2_group]),
      ksDist_min = min(value[Var1_group != Var2_group]),
      ksDist_max = max(value[Var1_group != Var2_group]),
      ksSim_mean = mean(value[Var1_group == Var2_group]),
      ksSim_med = median(value[Var1_group == Var2_group]),
      ksSim_std = sd(value[Var1_group == Var2_group]),
      ksSim_min = min(value[Var1_group == Var2_group]),
      ksSim_max = max(value[Var1_group == Var2_group]),
    )

  ### Join with sample names in preparation for output ###
  groups_metrics <- ksDistAggregated %>%
    right_join(groups_metrics, by = c("Var1_group" = "species_group")) %>%
    rename(species_group = Var1_group)
} else {
  print("KaKs failure (reading frame incorrect), skipping...")
}

### prepare and output data ###
print(groups_metrics$species_group[1])
print(nrow(groups_metrics))
print(names(groups_metrics))
print(groups_metrics)

groups_metrics %>%
  select(sample_name, species_group, everything()) %>%  # reorder data
  rename(UPID = sample_name) %>%
  select_if(~ length(unique(na.omit(.))) > 1 | length(.) < 2) %>%  # drop columns with all same values
  write_delim(args[3], delim = "\t")  # save one combined file

sink()
sink(type="message")
