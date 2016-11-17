rm(list = ls())

setwd("/Users/katharina/Dropbox/Projects/Yelp Challenge/Data")

image <- "classified.Rdata"

df <- read.csv("reviews_classified.csv")
save.image(image)

load(image)