rm(list = ls())

library(tm)
library(ggplot2)
library(ggrepel)
library(scales)
library(grid)
library(gridExtra)

setwd("/Users/katharina/Dropbox/Yelp Challenge/Data")
df <- read.csv("term_count.csv")

setwd("/Users/katharina/Dropbox/Yelp Challenge/Output")

df$Italian_pct <- (df$Italian / sum(df$Italian, na.rm = T)) * 100
df$Mexican_pct <- (df$Mexican / sum(df$Mexican, na.rm = T)) * 100
df$American_pct <- (df$American / sum(df$American, na.rm = T)) * 100

# Calculate ratio of Mexican to Italian term percentage
# Only include terms present at least ten times in each type
df$MexIt_pct_ratio <- df$Mexican_pct / df$Italian_pct
df$MexIt_pct_ratio[df$Mexican_pct < df$Italian_pct] <-
  (-1 / df$MexIt_pct_ratio[df$Mexican_pct < df$Italian_pct])

df$rare_term <- 0
df$rare_term[df$Mexican < 5 | df$Italian < 5] <- 1
table(df$rare_term)
df <- subset(df, rare_term == 0)

# Create measure for terms that are distinctly Italian or Mexican
df$big_ItMex_pct <- 0
df$big_ItMex_pct[abs(df$MexIt_pct_ratio) > 100] <- 1
table(df$big_ItMex_pct)

# Scatter plot with a log scale b/c a few words are very frequent
p <- ggplot(df, aes(x = Italian_pct, y = Mexican_pct)) +
  geom_point(aes(color = factor(big_ItMex_pct))) +
  scale_color_manual(values = c("grey", "red")) + guides(color = F) +
  theme_bw(base_size = 16) + ggtitle("Term Frequency by Cuisine") +
  xlab("Italian Term Percentage") + ylab("Mexican Term Percentage") +
  coord_fixed() + geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  geom_text_repel(data = subset(df, big_ItMex_pct == 1), aes(label = term), size = 5,
                  box.padding = unit(0.05, "lines"), point.padding = unit(0.05, "lines")) +
  scale_y_log10() + scale_x_log10() # log scales include 0

grid.newpage()
footnote <- "Note: Only terms occurring at least 5 times in both cuisines are included."
g <- arrangeGrob(p, bottom = textGrob(footnote, x = 0, hjust = -0.05, vjust=0.1, gp = gpar(fontface = "italic", fontsize = 10)))
grid.draw(g)

ggsave('it_mex_terms.png', g, dpi = 300, width = 12, height = 10)
