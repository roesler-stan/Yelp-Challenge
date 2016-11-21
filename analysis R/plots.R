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
plot_data <- ddply(df, .(word_count_cat), summarise, stars_review = mean(stars_review, na.rm = T))
plot_data <- subset(plot_data, !is.na(word_count_cat))

p <- ggplot(plot_data, aes(x = word_count_cat, y = stars_review)) + geom_point() +
  theme_bw() + xlab("Word Count") + ylab("Stars") + ggtitle("Review Stars by Word Count") +
  theme(axis.text = element_text(size = 14), axis.title = element_text(size = 18),
        plot.title = element_text(size = 22)) + scale_x_continuous(labels = comma)

g <- arrangeGrob(p, sub = textGrob("Note: Data are from Yelp Academic Challenge, Round 8.",
                                   x = unit(0.02, "npc"), just = "left",
                                   gp = gpar(fontsize = 10)), nrow = 2, heights = c(20, 1))

ggsave("word_count_stars.png", g, width=12, height=10)


plot_data <- data.frame()
groups <- c("food_present", "service_present", "money_present")
for (group in groups) {
  N = tapply(df$stars_review, df[, group],
             FUN = function(x) length(which(!is.na(x))))
  star_means = tapply(df$stars_review, df[, group], FUN = mean, na.rm=TRUE)
  star_sd = tapply(df$stars_review, df[, group], FUN = sd, na.rm=TRUE)
  polarity_means = tapply(df$polarity, df[, group], FUN = mean, na.rm=TRUE)
  polarity_sd = tapply(df$polarity, df[, group], FUN = sd, na.rm=TRUE)
  row1 <- c("group" = group, "group_value" = 0, "N" = N[[1]],
            "stars_review_mean" = star_means[[1]], "stars_review_sd" = star_sd[[1]],
            "polarity_mean" = polarity_means[[1]], "polarity_sd" = polarity_sd[[1]])
  row2 <- c("group" = group, "group_value" = 1,  "N" = N[[2]],
            "stars_review_mean" = star_means[[2]], "stars_review_sd" = star_sd[[2]],
            "polarity_mean" = polarity_means[[2]], "polarity_sd" = polarity_sd[[2]])
  if (!nrow(plot_data)) {
    plot_data <<- row1
    plot_data <<- smartbind(plot_data, row2)
  }
  else {
    plot_data <<- smartbind(plot_data, row1)
    plot_data <<- smartbind(plot_data, row2)    
  }
}
row.names(plot_data) <- NULL

plot_data$group_name <- paste0(plot_data$group, plot_data$group_value)
cols <- c("stars_review_mean", "stars_review_sd", "polarity_mean", "polarity_sd",
          "N")
for (col in cols) {
  plot_data[, col] <- as.numeric(as.character(plot_data[, col]))  
}

# SE = SD / sqrt(N)
plot_data$stars_review_se <- plot_data$stars_review_sd / sqrt(plot_data$N)


p <- ggplot(plot_data, aes(x=group, y=stars_review_mean, group=group_value,
                           fill = group_value)) +
  geom_bar(stat = "identity", position = "dodge") + theme_bw() +
  xlab("Topic") + ylab("Average Review Stars") +
  ggtitle("Review Stars by Topic Mentioned") +
  scale_fill_brewer("Topic Present", labels = c("No", "Yes"), palette = "Accent") +
  geom_errorbar(aes(ymax = stars_review_mean + stars_review_se,
                    ymin = stars_review_mean - stars_review_se), width=0.8,
                position = position_dodge(width=0.9)) +
  scale_x_discrete(labels = c("Food", "Service", "Money")) +
  coord_cartesian(ylim = c(1, 5)) +
  theme(axis.text = element_text(size = 14),
        axis.title = element_text(size = 18),
        plot.title = element_text(size = 22),
        legend.title = element_text(size = 18), legend.text = element_text(size=14))

g <- arrangeGrob(p, sub = textGrob("Note: Data are from Yelp Academic Challenge, Round 8.  Black lines represent standard errors.",
                                   x = unit(0.02, "npc"), just = "left",
                                   gp = gpar(fontsize = 10)), nrow = 2, heights = c(20, 1))

ggsave("stars_mean_by_topic.png", g, width = 12, height = 10)


df = subset(df, stars_review > 0)
table(df$food_present, df$stars_review)

plot_data = ddply(df, .(stars_review), summarise,
      food_present=mean(food_present, na.rm=T) * 100,
      service_present=mean(service_present, na.rm=T) * 100,
      money_present=mean(money_present, na.rm=T) * 100)

plot_data = melt(plot_data, id.vars="stars_review", variable.name="Topic", value.name="percent")

p <- ggplot(plot_data, aes(x=stars_review, y=percent, fill=Topic)) +
  geom_bar(stat="identity", position="dodge") +
  scale_fill_brewer(palette="Accent", labels=c("Food", "Service", "Money")) +
  xlab('Review Stars') + ylab("Percent Mentionoing Topic") +
  ggtitle("Review Stars by Topic") + theme_bw() +
  theme(axis.text = element_text(size = 14), axis.title = element_text(size = 18),
        plot.title = element_text(size = 22),
        legend.title = element_text(size = 18), legend.text = element_text(size=14))

g <- arrangeGrob(p, sub = textGrob("Note: Data are from Yelp Academic Challenge, Round 8.  Black lines represent standard errors.",
                                   x = unit(0.02, "npc"), just = "left",
                                   gp = gpar(fontsize = 10)), nrow = 2, heights = c(20, 1))

ggsave("stars_by_topic.png", g, width = 12, height = 10)
