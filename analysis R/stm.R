library(stm)

setwd("/Users/katharina/Dropbox/Projects/Yelp Challenge/Output")

# Only include words occuring at least MIN_WORDS times
MIN_WORDS = 1
NUM_TOPICS = 5

cols <- names(df)[names(df) != "text"]
processed <- textProcessor(df$text, meta = df[, cols], striphtml = T)
out <- prepDocuments(processed$documents, processed$vocab, processed$meta,
                     lower.thresh = MIN_WORDS)

prevFit <- stm(out$documents, out$vocab, K = NUM_TOPICS, prevalence =~ stars_review,
               max.em.its = 75, data = out$meta, init.type = "Spectral")

# Top words for each topic
labelTopics(prevFit)

# Word clouds
for (i in 1:5) {
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

png("topic5_stars.png")

plot.STM(content, type="perspectives", topics = 2, covarlevels = c(3, 5))
plot.STM(content, type="perspectives", topics = 2, covarlevels = c(4, 5))

plot.STM(content, type = "perspectives", topics = 5, custom.labels = c(seq(1, NUM_TOPICS)), n = 50)
title("Topic 5 by Review Stars")
dev.off()

image <- "/Users/katharina/Dropbox/Projects/Yelp Challenge/Data/classified.Rdata"
save.image(image)
