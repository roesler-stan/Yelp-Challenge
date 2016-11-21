cor.test(df$polarity, df$service_present,
         method = "pearson", use="pairwise.complete.obs")

cor.test(df$polarity, df$service_pred_logit,
         method = "pearson", use="pairwise.complete.obs")

cor.test(df$polarity, df$service_pred_prob_logit,
         method = "pearson", use="pairwise.complete.obs")


cor.test(df$polarity, df$money_pred_logit,
         method = "pearson", use="pairwise.complete.obs")


tapply(df$polarity, df$food_present, FUN = mean, na.rm=TRUE)
tapply(df$polarity, df$service_present, FUN = mean, na.rm=TRUE)
tapply(df$polarity, df$money_present, FUN = mean, na.rm=TRUE)

tapply(df$polarity, df$money_pred_logit, FUN = mean, na.rm=TRUE)
tapply(df$polarity, df$service_pred_logit, FUN = mean, na.rm=TRUE)


tapply(df$stars_review, df$food_present, FUN = mean, na.rm=TRUE)
tapply(df$stars_review, df$service_present, FUN = mean, na.rm=TRUE)
tapply(df$stars_review, df$money_present, FUN = mean, na.rm=TRUE)


t.test(stars_review ~ food_present, data = df)
t.test(stars_review ~ service_present, data = df)
t.test(stars_review ~ money_present, data = df)
