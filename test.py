
ro.r("infasta <- read.FASTA('{}')".format(filename))
ro.r("dist <- dist.dna(infasta)")
ro.r("UPGMA <- upgma(dist)")
ro.r("plot(UPGMA)")
