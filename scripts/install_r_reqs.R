#!/usr/bin/env Rscript
# attempt to install required packages, skipping those already installed
if (!require(ape)) { install.packages("ape", repos="https://cloud.r-project.org/") }
if (!require(adephylo)) { install.packages("adephylo", repos="https://cloud.r-project.org/") }
if (!require(phytools)) { install.packages("phytools", repos="https://cloud.r-project.org/") }
if (!require(phangorn)) { install.packages("phangorn", repos="https://cloud.r-project.org/") }
if (!require(paran)) { install.packages("paran", repos="https://cloud.r-project.org/") }
if (!require(splits)) { install.packages("splits", repos="http://R-Forge.R-project.org") }
if (!require(seqinr)) { install.packages("seqinr", repos="https://cloud.r-project.org/") }
if (!require(stringr)) { install.packages("stringr", repos="https://cloud.r-project.org/") }
if (!require(reshape2)) { install.packages("reshape2", repos="https://cloud.r-project.org/") }
if (!require(tidyverse)) { install.packages("tidyverse", repos="https://cloud.r-project.org/") }
