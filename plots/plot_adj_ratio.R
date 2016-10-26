rm(list = ls())

options(scipen=999)

library(tm)
library(ggplot2)
library(ggrepel)
library(scales)
library(grid)
library(gridExtra)

setwd("/Users/katharina/Dropbox/Yelp Challenge/Data")
df <- read.csv("term_count_adj.csv")

setwd("/Users/katharina/Dropbox/Yelp Challenge/Output")

# Don't include very rare terms
df$rare_term <- 0
df$rare_term[df$Mexican < 5 & df$Italian < 5] <- 1
table(df$rare_term)
df <- subset(df, rare_term == 0)

# Replace 0 with 1 to avoid infinity problem
df$Mexican[df$Mexican == 0] <- 1
df$Italian[df$Italian == 0] <- 1

df$Italian_pct <- (df$Italian / sum(df$Italian, na.rm = T)) * 100
df$Mexican_pct <- (df$Mexican / sum(df$Mexican, na.rm = T)) * 100
df$American_pct <- (df$American / sum(df$American, na.rm = T)) * 100

# Calculate ratio of Mexican to Italian term percentage
df$MexIt_pct_ratio <- df$Mexican_pct / df$Italian_pct
df$MexIt_pct_ratio[df$Mexican_pct < df$Italian_pct] <-
  (-1 / df$MexIt_pct_ratio[df$Mexican_pct < df$Italian_pct])

# sum(abs(df$MexIt_pct_ratio) > 200)

plot_data <- subset(df, abs(MexIt_pct_ratio) > 200)

# Barplot with most distinctive adjectives
p <- ggplot(plot_data, aes(x = reorder(term, MexIt_pct_ratio), y = MexIt_pct_ratio)) +
  geom_bar(stat = "identity") + theme_bw() +
  theme(axis.text.x = element_text(size = 10, angle = 45, vjust = 0.5),
        plot.title = element_text(size = 18),
        axis.title = element_text(size = 16)) +
  xlab("Adjective") + ylab("More Mexican than Italian (Ratio of Percentages)") +
  ggtitle("Adjectives Differentiating Mexican and Italian Restaurants")

grid.newpage()
footnote <- "Note: Only terms occurring at least 5 times in either Mexican or Italian restaurants are included.  0 occurences were replace with 1."
g <- arrangeGrob(p, bottom = textGrob(footnote, x = 0, hjust = -0.03, vjust=0.1, gp = gpar(fontface = "italic", fontsize = 10)))

ggsave('it_mex_adj_ratio.png', g, dpi = 300, width = 12, height = 10)
