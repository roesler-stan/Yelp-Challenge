library(ggplot2)
library(plyr)
library(gtools)
library(data.table)
library(grid)
library(gridExtra)
library(scales)

setwd("/Users/katharina/Dropbox/Projects/Yelp Challenge/Output")

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
  theme(axis.text = element_text(size = 16), axis.title = element_text(size = 20),
        plot.title = element_text(size = 24),
        legend.title = element_text(size = 20), legend.text = element_text(size=16))

g <- arrangeGrob(p, sub = textGrob("Note: Data are from Yelp Academic Challenge, Round 8.  Black lines represent standard errors.",
                                   x = unit(0.02, "npc"), just = "left",
                                   gp = gpar(fontsize = 14)), nrow = 2, heights = c(20, 1))

ggsave("stars_by_topic.png", g, width = 12, height = 10)
