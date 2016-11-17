library(texreg)

setwd("/Users/katharina/Dropbox/Projects/Yelp Challenge/Output")

# food_pred_logit is always 1
# controls1 = c("service_pred_logit", "money_pred_logit")
# controls2 = c("service_pred_prob_logit", "money_pred_prob_logit",
              "food_pred_prob_logit")

# Use words present variables
controls1 = c("word_count")
controls2 = c(controls1, "food_present", "service_present", "money_present")


# Standardize independent variables
df_std <- df
for (var in c(controls1, controls2)) {
  m <- mean(df_std[, var], na.rm = T)
  s <- sd(df_std[, var], na.rm = T)
  df_std[, var] <- (df_std[, var] - m) / s
}

vars1 <- c(controls1, "polarity")
vars2 <- c(controls2, "polarity")
vars3 <- c(controls1, "stars_review")
vars4 <- c(controls2, "stars_review")

model1 <- lm(polarity ~ ., data = df_std[, vars1])
model2 <- lm(polarity ~ ., data = df_std[, vars2])
model3 <- lm(stars_review ~ ., data = df_std[, vars3])
model4 <- lm(stars_review ~ ., data = df_std[, vars4])

outfile <- "models.doc"
htmlreg(list(model1, model2, model3, model4),
        center = F, file = outfile, digits = 2,
        custom.model.names = c("Polarity", "Polarity", "Stars", "Stars"),
        caption = paste0('Linear Regression Predicting Review Outcome'),
        caption.above = T, inline.css = F, longtable = T)


## Train model on 70% of data, then calculate R^2 for remaining 30% predictions
training <- round(nrow(df_std) * 0.7, 0)
test <- round(nrow(df_std) * 0.3, 0)
train_rows <- seq(1, training)
model_train <- lm(stars_review ~ ., data = df_std[train_rows,
                                                  c(controls2, "stars_review")])
coefs <- model_train$coefficients
test_rows <- seq(training + 1, training + test)
predictions <- coefs[1] + rowSums(as.matrix(df_std[test_rows, controls2]) %*% diag(coefs[2:5]))
actual <- df_std[test_rows, "stars_review"]
residuals <- actual - predictions

## R^2 = (sum(residual^2)) / outcome variance
actual_mean = mean(actual, na.rm = T)
r_sq = 1 - (sum(residuals ** 2) / sum((actual - actual_mean) ** 2))
## 0.33 R-squared on test set
