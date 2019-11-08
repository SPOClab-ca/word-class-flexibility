library(tidyverse)
library(ggplot2)

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

ggplot(smaller_attestation_dates, aes(x=noun_date, y=verb_date)) +
  theme_bw() +
  geom_text(label=smaller_attestation_dates$word,
            check_overlap=T, size=3.5) +
  geom_abline(slope=1)
