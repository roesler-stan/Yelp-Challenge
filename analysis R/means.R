mean(df$food, na.rm=T)
mean(df$service, na.rm=T)
mean(df$money, na.rm=T)

mean(df$food_present, na.rm=T)
mean(df$service_present, na.rm=T)
mean(df$money_present, na.rm=T)

df$any_present[!is.na(df$food)] = 0
df$any_present[df$food == 1] = 1
df$any_present[df$service == 1] = 1
df$any_present[df$money == 1] = 1
table(df$any_present)

df$any_present[!is.na(df$food_present)] = 0
df$any_present[df$food_present == 1] = 1
df$any_present[df$service_present == 1] = 1
df$any_present[df$money_present == 1] = 1
prop.table(table(df$any_present))
