library(stm)

ROWS <- 1000
df_small <- df[1: ROWS, ]

setwd("/Users/katharina/Dropbox/Projects/Yelp Challenge/Output")

# Only include words occuring at least MIN_WORDS times
MIN_WORDS = 1
NUM_TOPICS = 10

cols <- names(df_small)[names(df_small) != "text"]
processed <- textProcessor(df_small$text, meta = df_small[, cols], striphtml = T)
out <- prepDocuments(processed$documents, processed$vocab, processed$meta,
                     lower.thresh = MIN_WORDS)

prevFit <- stm(out$documents, out$vocab, K = NUM_TOPICS, prevalence =~ stars_review,
               max.em.its = 75, data = out$meta, init.type = "Spectral")

# Top words for each topic
labelTopics(prevFit)

# Word clouds
for (i in 1:NUM_TOPICS) {
  jpeg(paste0("cloud", i, ".jpeg"), quality = 100, pointsize = 18)
  cloud(prevFit, topic = i, scale = c(3, 0.25))
  dev.off()
}


# Topic proportions by review stars and associated words
plot.STM(prevFit, text.cex = 0.6, n = 4, xlim = c(0, 0.7))
plot.STM(prevFit, text.cex = 0.6, n = 4, xlim = c(0, 0.7), labeltype = "score")


# Consider the words for each stars_review level - takes a long time (EM iteration)
content <- stm(out$documents, out$vocab, K = NUM_TOPICS,
               prevalence =~ stars_review, content =~ stars_review,
               max.em.its = 75, data = out$meta, init.type = "Spectral")

for (i in 1:NUM_TOPICS) {
  filename = paste0("topic", i, "_stars.png")
  png(filename)
  plot.STM(content, type = "perspectives", topics = i, n = 50, covarlevels = c(3, 5),
           plabels = c("3 Stars", "5 Stars"))
  plot_title = paste0("Topic ", i)
  title(plot_title)
  dev.off()
}

image <- "/Users/katharina/Dropbox/Projects/Yelp Challenge/Data/classified.Rdata"
save.image(image)
