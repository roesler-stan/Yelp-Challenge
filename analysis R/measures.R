mean(df$food == df$food_present, na.rm=T)
mean(df$service == df$service_present, na.rm=T)
mean(df$money == df$money_present, na.rm=T)

mean(df$food > df$food_present, na.rm=T)
mean(df$food < df$food_present, na.rm=T)

mean(df$service > df$service_present, na.rm=T)
mean(df$service < df$service_present, na.rm=T)

mean(df$money > df$money_present, na.rm=T)
mean(df$money < df$money_present, na.rm=T)
