length(unique(df$business_id))
length(unique(df$city))
length(unique(df$state))

df_coded = subset(df, !is.na(food))

table(df_coded$food_present, df_coded$food)
table(df_coded$service_present, df_coded$service)
table(df_coded$money_present, df_coded$money)

mean(df_coded$food_present == df_coded$food) # 78%
mean(df_coded$service_present == df_coded$service) # 85%
mean(df_coded$money_present == df_coded$money) # 90%