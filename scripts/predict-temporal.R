# Compare attestation date, corpus frequency, and semantic variation
library(tidyverse)
library(ggplot2)

attestation_dates <- read_csv('attestation_dates.csv')
data <- read_csv('nv_variation.csv') %>%
  merge(attestation_dates, by.x='lemma', by.y='word')

# Filter attestation dates not same
data <- data %>% filter(noun_date != verb_date)

# Binarize variables
data$is_freq_noun <- data$noun_count > data$verb_count
data$is_attestation_noun <- data$noun_date < data$verb_date
data$is_semantic_noun <- data$n_variation > data$v_variation

# Cross-tables
table(data$is_attestation_noun, data$is_freq_noun)
table(data$is_attestation_noun, data$is_semantic_noun)
table(data$is_freq_noun, data$is_semantic_noun)

sum((data$is_attestation_noun == data$is_freq_noun) & (data$is_freq_noun == data$is_semantic_noun))
