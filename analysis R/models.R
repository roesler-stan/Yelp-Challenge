library(texreg)

setwd("/Users/katharina/Dropbox/Projects/Yelp Challenge/Output")

# Standardize independent variables
standardize_vars = c("word_count", "food_present", "service_present", "money_present")
df_std <- df
for (var in standardize_vars) {
  m <- mean(df_std[, var], na.rm = T)
  s <- sd(df_std[, var], na.rm = T)
  df_std[, var] <- (df_std[, var] - m) / s
}

# Squared word count, after standardization
# Can't square binary measures
df_std$word_count_sq = df_std$word_count ** 2

controls1 = c("word_count", "word_count_sq")
controls2 = c(controls1, "food_present", "service_present", "money_present")

vars1 <- c(controls1, "stars_review")
vars2 <- c(controls2, "stars_review")

## Train model on 70% of data, then calculate R^2 for remaining 30% predictions
training <- round(nrow(df_std) * 0.7, 0)
test <- round(nrow(df_std) * 0.3, 0)
train_rows <- seq(1, training)
test_rows <- seq(training + 1, training + test)
actual <- df_std[test_rows, "stars_review"]
actual_mean = mean(actual, na.rm = T)

model1 <- lm(stars_review ~ ., data = df_std[train_rows, vars1])
model2 <- lm(stars_review ~ ., data = df_std[train_rows, vars2])


coefs1 <- model1$coefficients
predictions1 <- coefs1[1] + rowSums(as.matrix(df_std[test_rows, controls1]) %*% diag(coefs1[2:3]))
residuals1 <- actual - predictions1
## R^2 = (sum(residual^2)) / outcome variance
r_sq1 = 1 - (sum(residuals1 ** 2) / sum((actual - actual_mean) ** 2))
r_sq1

coefs2 <- model2$coefficients
predictions2 <- coefs2[1] + rowSums(as.matrix(df_std[test_rows, controls2]) %*% diag(coefs2[2:6]))
residuals2 <- actual - predictions2
r_sq2 = 1 - (sum(residuals2 ** 2) / sum((actual - actual_mean) ** 2))
r_sq2


## Model 1 has 0.029 R-squared on test set, compared to 0.029 on training set
## Model 2 has 0.036 R-squared on test set, compared to 0.037 on training set

outfile <- "models.tex"
texreg(list(model1, model2),
        center = F, file = outfile, digits = 2,
        custom.model.names = c("Model 1", "Model 2"),
        caption = paste0('Linear Regression Predicting Review Stars'),
        caption.above = T, inline.css = F, longtable = F)
