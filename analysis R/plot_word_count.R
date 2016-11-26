library(ggplot2)
library(plyr)
library(gtools)
library(data.table)
library(grid)
library(gridExtra)
library(scales)

setwd("/Users/katharina/Dropbox/Projects/Yelp Challenge/Output")

# Word count 10-word buckets, centered
df$word_count_cat <- (cut(df$word_count, seq(0, 1050, 10), labels = F) * 10) - 5
plot_data <- ddply(df, .(word_count_cat), summarise,
	stars_review = mean(stars_review, na.rm=T),
	food = mean(food, na.rm=T)  * 100,
	service = mean(service, na.rm=T)  * 100,
	money = mean(money, na.rm=T) * 100)
plot_data <- subset(plot_data, !is.na(word_count_cat))

p <- ggplot(plot_data, aes(x = word_count_cat, y = stars_review)) + geom_point() +
  theme_bw() + xlab("Word Count") + ylab("Stars") + ggtitle("Review Stars by Word Count") +
  theme(axis.text = element_text(size = 18), axis.title = element_text(size = 22),
        plot.title = element_text(size = 26)) + scale_x_continuous(labels = comma)

g <- arrangeGrob(p, sub = textGrob("Note: Data are from Yelp Academic Challenge, Round 8.",
                                   x = unit(0.02, "npc"), just = "left",
                                   gp = gpar(fontsize = 16)), nrow = 2, heights = c(20, 1))

ggsave("word_count_stars.png", g, width=12, height=10)


p <- ggplot(plot_data, aes(x = word_count_cat, y = service)) + geom_point() +
  theme_bw() + xlab("Word Count") + ylab("% Mention Service") + ggtitle("Service Mentions by Word Count") +
  theme(axis.text = element_text(size = 14), axis.title = element_text(size = 18),
        plot.title = element_text(size = 22)) + scale_x_continuous(labels = comma)

g <- arrangeGrob(p, sub = textGrob("Note: Data are from Yelp Academic Challenge, Round 8.",
	x = unit(0.02, "npc"), just = "left", gp = gpar(fontsize = 14)), nrow = 2, heights = c(20, 1))

ggsave("word_count_service.png", g, width=12, height=10)
