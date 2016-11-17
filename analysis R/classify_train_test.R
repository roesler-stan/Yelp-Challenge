library(rms)
library(texreg)
library(robustbase)

setwd("/Users/katharina/Dropbox/Projects/Yelp Challenge/Output")

## Train model on 70% of data, then calculate R^2 for remaining 30% predictions
df_std <- df
df_std <- subset(df_std, !is.na(food))
df_std <- subset(df_std, !is.na(service))
df_std <- subset(df_std, !is.na(money))

# Duplicate some rows without food mentioned to increase their weight in the model
nofood = subset(df_std, food == 0)
for (i in seq(1, 10)) {
  df_std <- smartbind(df_std, nofood)  
}

coded_rows = sum(!is.na(df_std$food))
training <- round(coded_rows * 0.7, 0)
test <- round(coded_rows * 0.3, 0)
train_rows <- seq(1, training)
test_rows <- seq(training + 1, training + test)

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

model1 <- glmrob(food ~ ., df_std[train_rows, vars1], x = T, y = T,
                 family = binomial(link = "logit"))
model2 <- glmrob(service ~ ., df_std[train_rows, vars2], x = T, y = T,
                 family = binomial(link = "logit"))
model3 <- glmrob(money ~ ., df_std[train_rows, vars3], x = T, y = T,
                 family = binomial(link = "logit"))
summary(model2)

outfile <- "models_classify.doc"
htmlreg(list(model1, model2, model3),
        center = F, file = outfile, digits = 2,
        custom.model.names = c("Food", "Service", "Money"),
        caption = paste0('Logistic Regression Predicting Topic Mentioned'),
        caption.above = T, inline.css = F, longtable = T)

outfile <- "models_classify.tex"
texreg(list(model1, model2, model3),
       center = F, file = outfile, digits = 2,
       custom.model.names = c("Food", "Service", "Money"),
       caption = paste0('Logistic Regression Predicting Topic Mentioned'),
       caption.above = T, inline.css = F, longtable = F)


# TO DO: get predictions for test set observations
coefs <- model1$coefficients
predictions <- exp(coefs[1]) * exp(df_std[train_rows, controls[[1]]] * coefs[[2]]) * exp(df_std[train_rows, controls[[2]]] * coefs[[3]]) * exp(df_std[train_rows, controls[[3]]] * coefs[[4]]) * exp(df_std[train_rows, controls[[4]]] * coefs[[5]])

exp(as.matrix(df_std[train_rows, controls]) %*% diag(coefs[2:5]))

predictions_food <- as.numeric(model1$fitted.values > 0.5)
predictions_service <- as.numeric(model2$fitted.values > 0.5)
predictions_money <- as.numeric(model3$fitted.values > 0.5)

actual_food <- df_std[test_rows, "food"]
actual_service <- df_std[test_rows, "service"]
actual_money <- df_std[test_rows, "money"]


actual_food <- df_std[train_rows, "food"]
actual_service <- df_std[train_rows, "service"]
actual_money <- df_std[train_rows, "money"]

# Compare true values to predictions
mean(actual_food != predictions_food)
mean(actual_food > predictions_food)
mean(actual_food < predictions_food)

residuals_food <- actual_food - predictions_food
