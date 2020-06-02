require(readxl)
require(tidyverse)

dat <- read_excel(
  "data/Copy of HymenopteraSequenceDataNE.xlsx",
  sheet = "Results",
  range = "A1:F160"
)

dat <- dat %>% drop_na(4)

clusters <- kmeans(dat[, 3:4], 2)

dat$Cluster <- as.factor(clusters$cluster)

ggplot(dat) +
  aes(x = Distance, y = Similarity, color = Cluster) +
  geom_point() +
  geom_text(mapping = aes(label = Group), hjust = 0, vjust = 0)

ggplot(dat) +
  aes(x = Cluster, y = Status, fill = Status) +
  geom_bar(stat = "identity")
