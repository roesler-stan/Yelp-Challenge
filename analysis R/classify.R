library(rms)
library(texreg)
library(robustbase)

setwd("/Users/katharina/Dropbox/Projects/Yelp Challenge/Output")

## Train model on 70% of data, then calculate R^2 for remaining 30% predictions
df_std <- df
df_std <- subset(df_std, !is.na(food))
df_std <- subset(df_std, !is.na(service))
df_std <- subset(df_std, !is.na(money))

controls = c("word_count", "food_present", "service_present", "money_present")

# Standardize independent variables
for (var in controls) {
  m <- mean(df_std[, var], na.rm = T)
  s <- sd(df_std[, var], na.rm = T)
  df_std[, var] <- (df_std[, var] - m) / s
}

vars1 <- c(controls, "food")
vars2 <- c(controls, "service")
vars3 <- c(controls, "money")

model1a <- lrm(food ~ word_count, df_std[, vars1], x = T, y = T)
model1b <- lrm(food ~ ., df_std[, vars1], x = T, y = T)
model2a <- lrm(service ~ word_count, df_std[, vars2], x = T, y = T)
model2b <- lrm(service ~ ., df_std[, vars2], x = T, y = T)
model3a <- lrm(money ~ word_count, df_std[, vars3], x = T, y = T)
model3b <- lrm(money ~ ., df_std[, vars3], x = T, y = T)


# model1 <- glmrob(food ~ ., df_std[, vars1], x = T, y = T,
#                  family = binomial(link = "logit"))
# model2 <- glmrob(service ~ ., df_std[, vars2], x = T, y = T,
#                  family = binomial(link = "logit"))
# model3 <- glmrob(money ~ ., df_std[, vars3], x = T, y = T,
#                  family = binomial(link = "logit"))
# summary(model1)


outfile <- "models_classify.tex"
texreg(list(model1b, model2b, model3b),
        center = F, file = outfile, digits = 2,
        custom.model.names = c("Food", "Service", "Money"),
        caption = paste0('Logistic Regression Predicting Topic Mentioned'),
        caption.above = T, inline.css = F, longtable = F)


model1b <- glmrob(food ~ ., df_std[, vars1], x = T, y = T,
                family = binomial(link = "logit"))
model2b <- glmrob(service ~ ., df_std[, vars2], x = T, y = T,
                  family = binomial(link = "logit"))
model3b <- glmrob(money ~ ., df_std[, vars3], x = T, y = T,
                  family = binomial(link = "logit"))

predictions_food <- as.numeric(model1b$fitted.values > 0.5)
predictions_service <- as.numeric(model2b$fitted.values > 0.5)
predictions_money <- as.numeric(model3b$fitted.values > 0.5)

actual_food <- df_std[, "food"]
actual_service <- df_std[, "service"]
actual_money <- df_std[, "money"]

# Compare true values to predictions
mean(actual_food == predictions_food)
mean(actual_service == predictions_service)
mean(actual_money == predictions_money)

mean(actual_food > predictions_food)
mean(actual_food < predictions_food)

mean(actual_food)
mean(actual_service)
mean(actual_money)

mean(df$food_present)
mean(df$service_present)
mean(df$money_present)
