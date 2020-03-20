library(tidyverse)

df <- read.csv("lemmas.csv")
theme_set(theme_bw())

# Compare distribution of responses
ggplot(df, aes(human_score1)) +
  geom_bar()
ggplot(df, aes(human_score2)) +
  geom_bar()

# Combine categories 0 and 1
df$human_score1[df$human_score1 == 1] <- 0
df$human_score2[df$human_score2 == 1] <- 0

# Confusion matrix
table(df$human_score1, df$human_score2)
cor(df$human_score1, df$human_score2)


# Compare human1 and ELMo
cor.test(df$human_score1, df$prototype_sim)
cor.test(df$human_score1, df$median_sim)
cor.test(df$human_score1, df$mean_sim)

# Compare human2 and ELMo
cor.test(df$human_score2, df$prototype_sim)
cor.test(df$human_score2, df$median_sim)
cor.test(df$human_score2, df$mean_sim)

# Combine humans
df$human_combined <- pmin(df$human_score1, df$human_score2)
cor.test(df$human_combined, df$prototype_sim)
cor.test(df$human_combined, df$median_sim)
cor.test(df$human_combined, df$mean_sim)


# Best so far: median_sim with minimum of two humans
ggplot(df, aes(x=as.factor(human_combined), y=median_sim)) +
  xlab("Human score (combined)") +
  ylab("ELMo score") +
  geom_boxplot()
