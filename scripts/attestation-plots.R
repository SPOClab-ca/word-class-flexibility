# Visualizations of attestation dates and corpus frequency
library(tidyverse)
library(ggplot2)
library(ggrepel)

attestation_dates <- read_csv('attestation_dates.csv')
sim_annotations <- read_csv('myself_plus_mturk.csv')

attestation_dates$gap <- attestation_dates$verb_date - attestation_dates$noun_date
attestation_dates$gap <- pmax(attestation_dates$gap, -500)
attestation_dates$gap <- pmin(attestation_dates$gap, 500)

ggplot(attestation_dates, aes(x=gap)) +
  geom_histogram(binwidth=50, col="white") +
  scale_x_continuous(breaks=seq(-500, 500, 50)) +
  theme_bw()

smaller_attestation_dates <- attestation_dates %>%
  merge(sim_annotations, by.x="word", by.y="lemma") %>%
  arrange(- noun_count - verb_count)
smaller_attestation_dates$noun_ratio <-
  smaller_attestation_dates$noun_count / (smaller_attestation_dates$noun_count + smaller_attestation_dates$verb_count)

# Too many labels otherwise
smaller_attestation_dates <- head(smaller_attestation_dates, 90)

ggplot(smaller_attestation_dates, aes(x=noun_date, y=verb_date)) +
  theme_bw() +
  geom_point() +
  geom_label_repel(label=smaller_attestation_dates$word,
                  size=3,
                  aes(color=smaller_attestation_dates$noun_ratio)) +
  scale_color_gradient(name="Noun Ratio", low="red", high="darkgreen") +
  geom_abline(slope=1) +
  xlab("Noun Date") +
  ylab("Verb Date")
