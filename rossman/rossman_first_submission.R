#
#
#
train <- read.csv("~/Development/kaggle/rossman/data/train.csv")
train$Date <- as.Date(train$Date, format="%Y-%m-%d")
str(train)
models <- vector(mode="list", length=2000)

first <- min(train$Store)
last <- max(train$Store)


lm1 <- lm(Sales ~ DayOfWeek + Open + Promo + StateHoliday + SchoolHoliday, data=subset(train, Store==1))
summary(lm1)

for (store in first:last)
{
	models[[store]] <- lm(Sales ~ DayOfWeek + Open + Promo + StateHoliday + SchoolHoliday, data=subset(train, Store==store), na.action=na.omit)
}
#
# so now we have this list of models, one for each store such that the index of the list is the store store number 
#
summary(models[[3]])

test <- read.csv('~/Development/kaggle/rossman/data/test.csv',na.strings=c(""))
str(test)
test$Date <- as.Date(test$Date, format="%Y-%m-%d")

# check to see which columns have NA values in the test data
sapply(test, function(x) sum(is.na(x)))
#
# the only variable with NA values is the Open column.
#
na_df <- subset(test, is.na(test$Open))

#         Id Store DayOfWeek       Date Open Promo StateHoliday SchoolHoliday
# 480     480   622         4 2015-09-17   NA     1            0             0
# 1336   1336   622         3 2015-09-16   NA     1            0             0
# 2192   2192   622         2 2015-09-15   NA     1            0             0
# 3048   3048   622         1 2015-09-14   NA     1            0             0
# 4760   4760   622         6 2015-09-12   NA     0            0             0
# 5616   5616   622         5 2015-09-11   NA     0            0             0
# 6472   6472   622         4 2015-09-10   NA     0            0             0
# 7328   7328   622         3 2015-09-09   NA     0            0             0
# 8184   8184   622         2 2015-09-08   NA     0            0             0
# 9040   9040   622         1 2015-09-07   NA     0            0             0
# 10752 10752   622         6 2015-09-05   NA     0            0             0

# I'm going to put in a rule that sets Open to 1 if StateHoliday = 0, SchoolHoliday=0 and DayofWeek != 7
test$Open <- ifelse(is.na(test$Open) & test$StateHoliday != 0 & test$SchoolHoliday != 0 & test$DayOfWeek != 7, 1, 0) 

df <- data.frame(Id = numeric(max(test$Id)), Sales = numeric(max(test$Id)))

for (i in min(test$Id):max(test$Id))
{
	df$Id[i] <- i
	df$Sales[i] <- predict(models[[test$Store[i]]], newdata=test[i,])
}


write.csv(df, file='~/Development/kaggle/rossman/data/predict.csv', row.names=FALSE)
