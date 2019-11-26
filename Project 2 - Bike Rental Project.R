#Clear the environment
rm(list = ls())

rm(library)

#Load the Libraries
library(DataCombine)
library(ggplot2)
library(gridExtra)
library(caret)

library(usdm)
library(corrgram)
library(DMwR)
library(corrplot)
library(rpart)
library(randomForest)
library(rpart.plot)



#Set Working directory
setwd("C:/Users/rnp/Desktop/Arjun/Data Science/Project2")


# Load the bike_df.cvs files 
bike_df = read.csv("day.csv", header = T)

#****Explore the bike_df.csv file****#

#display the dataset
head(bike_df)

bike_df.head(5)

#view the dimension
dim(bike_df)

#view summary
summary(bike_df)


#view the structure of dataset
str(bike_df)

org_bike_df <- bike_df
dim(org_bike_df)

###
#****** Feature engineering*****#
###

#adding new columns by converting the categorical columns with actual values 
#and, normalized continous values to actual values w.r.t given in problem statement

#Create new columns and merge to exsiting dataset
bike_df$actual_temp <-bike_df$temp*39
bike_df$actual_feel_temp <-bike_df$atemp*50
bike_df$actual_hum = bike_df$hum * 100
bike_df$actual_windspeed <-bike_df$windspeed*67

bike_df$actual_season = factor(x = bike_df$season, levels = c(1,2,3,4), labels = c("Spring-(1)","Summer-(2)","Fall-(3)","Winter-(4)"))
bike_df$actual_yr = factor(x = bike_df$yr,levels = c(0,1),labels = c("2011-(0)","2012-(1)"))
bike_df$actual_holiday = factor(x = bike_df$holiday,levels = c(0,1),labels = c("Working day-(0)","Holiday/weekend-(1)"))
bike_df$actual_weathersit = factor(x = bike_df$weathersit,levels = c(1,2,3,4),labels = c("Clear-(1)","Cloudy/Mist-(2)","Light Snow/Rain/clouds-(3)","Heavy Rain/Snow/Fog(4)"))

#check the structure of the dataset after adding more columns
str(bike_df)

###
#Univariate analysis to see how the data is distributed
###

# Continous variable
hist_plot1 = ggplot(bike_df, aes(actual_temp))+theme_bw()+geom_histogram(fill='blue', bins = 20)+ggtitle("Distribution of temp")+theme(text = element_text(size = 10))
hist_plot2 = ggplot(bike_df, aes(actual_feel_temp))+theme_bw()+geom_histogram(fill='blue', bins = 20)+ggtitle("Distribution of feeling temp")+theme(text = element_text(size = 10))
hist_plot3 = ggplot(bike_df, aes(actual_hum))+theme_bw()+geom_histogram(fill='blue', bins = 20)+ggtitle("Distribution of Humidity")+theme(text = element_text(size = 10))
hist_plot4 = ggplot(bike_df, aes(actual_windspeed))+theme_bw()+geom_histogram(fill='blue', bins = 20)+ggtitle("Distribution of windspeed")+theme(text = element_text(size = 10))


#Plot the Histogram graph together for continous variable
gridExtra::grid.arrange(hist_plot1, hist_plot2, hist_plot3, hist_plot4,ncol=2)


#Categorical variable
bar_plot1 = ggplot(bike_df, aes(actual_season))+theme_bw()+geom_bar(fill='orange')+ggtitle("Counts of Season")+theme(text = element_text(size = 10))
bar_plot2 = ggplot(bike_df, aes(actual_yr))+theme_bw()+geom_bar(fill='blue')+ggtitle("Counts of Year")+theme(text = element_text(size = 10))
bar_plot3 = ggplot(bike_df, aes(actual_holiday))+theme_bw()+geom_bar(fill='lightblue')+ggtitle("Counts of Holiday/Weekend/Weekday")+theme(text = element_text(size = 10))
bar_plot4 = ggplot(bike_df, aes(actual_weathersit))+theme_bw()+geom_bar(fill='green')+ggtitle("Counts of weather")+theme(text = element_text(size = 10))

#Plot the Histogram graph together for continous variable
gridExtra::grid.arrange(bar_plot1,bar_plot2, bar_plot3, bar_plot4,ncol=2)

###
#*** Bivariate analysis*****#
###

#Continous variable
bike_df$actual_temp <- as.factor(bike_df$actual_temp)
bike_df$actual_feel_temp <- as.factor(bike_df$actual_feel_temp)
scatter_plot = ggplot(bike_df, aes(x=actual_temp, y=actual_feel_temp))+geom_point()+ggtitle("Distibution of Temp and Atemp")
plot(scatter_plot)

#### we can observer temp and atemp has a positve linear relation. 
#### Correlation give us idea about Linear relpationship b/w 2 continous variables

bike_df$actual_temp = as.numeric(bike_df$actual_temp)
bike_df$actual_feel_temp = as.numeric(bike_df$actual_feel_temp)


### Finding the correlation between temp and atemp

cor(bike_df$actual_temp, bike_df$actual_feel_temp)

### correlation - 0.9917016

#Continous and Categorical
bike_df$actual_temp <- as.integer(bike_df$actual_temp)
bike_df$actual_feel_temp = as.integer(bike_df$actual_feel_temp)
box_plot = ggplot(bike_df, aes(x=actual_season, y=actual_temp))+geom_boxplot()
plot(box_plot)


###
#****** PRE-PROCESSION *************#
###


#Finding the missing values in dataset
missing_value<-data.frame(missing_value=apply(bike_df,2,function(x){sum(is.na(x))}))
missing_value

#### NO missing values found

#Check for collinearity using correlation graph
corrgram(bike_df, order = F, upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")



###
#Detect multicollinearity
###

####We have already verifed that a  relation between "temp" and "atemp" during bivariate analysis. 
####both are strongly correlated to each other
####Now, will see if collinearity existence bewtween Continuous variable over target  using VIF method


vif_df <- bike_df[,c('temp','atemp','hum','windspeed')]
vif(vif_df)



### VIF values
#   Variables       VIF
# 1      temp 62.969819
# 2     atemp 63.632351
# 3       hum  1.079267
# 4 windspeed  1.126768
###

### From the above we can understand that "temp" and "atemp" have a high Variance inflation factor(VIF), 
### they have almost same variance within the dataset. So, we might need to drop one of the feature before 
### moving to model buliding otherwise will end-up buliding a model with high multicolinearity

###
#******** Outlier detection and removal ********#
###

box_plot1 = ggplot(aes_string(y = bike_df$temp), data = bike_df)+stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=10,outlier.size=1, notch=FALSE) +
  theme(legend.position="bottom")+labs(y=bike_df$temp)+ggtitle(paste("Box plot for temp"))

box_plot2 = ggplot(aes_string(y = bike_df$atemp), data = bike_df)+stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "blue" ,outlier.shape=10,outlier.size=1, notch=FALSE) +
  theme(legend.position="bottom")+labs(y=bike_df$atemp)+ggtitle(paste("Box plot for atemp"))

box_plot3 = ggplot(aes_string(y = bike_df$hum), data = bike_df)+stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "green" ,outlier.shape=10,outlier.size=1, notch=FALSE) +
  theme(legend.position="bottom")+labs(y=bike_df$hum)+ggtitle(paste("Box plot for hum"))

box_plot4 = ggplot(aes_string(y = bike_df$windspeed), data = bike_df)+stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "orange" ,outlier.shape=10,outlier.size=1, notch=FALSE) +
  theme(legend.position="bottom")+labs(y=bike_df$windspeed)+ggtitle(paste("Box plot for windspeed"))

gridExtra::grid.arrange(box_plot1,box_plot2,box_plot3,box_plot4, ncol=2, top='Outlier for continous variable')


### as you refer from the BOX PLOT generated, we can observe OUTLIERS in features "hum" and "windspeed"

#Removing the outlier from feature "hum"
#get the outlier values
hum_outliers <- boxplot(bike_df$hum, plot=FALSE)$out
hum_outliers

#display the outliers
bike_df[which(bike_df$hum %in% hum_outliers),]

#drop those outliers
bike_df <- bike_df[-which(bike_df$hum %in% hum_outliers),]


#Removing the outlier from feature "windspeed"
#get the outlier values
win_outliers <- boxplot(bike_df$windspeed, plot=FALSE)$out

#display the outliers
bike_df[which(bike_df$windspeed %in% win_outliers),]

#drop those outliers
bike_df <- bike_df[-which(bike_df$windspeed %in% win_outliers),]

dim(bike_df)


box_plot3 = ggplot(aes_string(y = bike_df$hum), data = bike_df)+stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "green" ,outlier.shape=10,outlier.size=1, notch=FALSE) +
  theme(legend.position="bottom")+labs(y=bike_df$hum)+ggtitle(paste("Box plot for hum"))

box_plot4 = ggplot(aes_string(y = bike_df$windspeed), data = bike_df)+stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "orange" ,outlier.shape=10,outlier.size=1, notch=FALSE) +
  theme(legend.position="bottom")+labs(y=bike_df$windspeed)+ggtitle(paste("Box plot for windspeed"))

gridExtra::grid.arrange(box_plot3,box_plot4, ncol=2, top='Box plot after Outlier removal')



#********* MODEL BULIDING *********#
colnames(bike_df)
#Drop the columns/features which are not needed

bike_df <- subset(bike_df,select = -c(instant, dteday, atemp, casual, registered, actual_temp, actual_feel_temp, actual_hum, actual_windspeed, actual_season, actual_yr, actual_holiday, actual_weathersit))

colnames(bike_df)


#####
#**** Liner regression model *****#
#####


# divide data into train and test

train_index = sample(1:nrow(bike_df), 0.8 * nrow(bike_df))
train <- bike_df[train_index,]
test <- bike_df[-train_index,]

#Inovke linear regression model
lr_model = lm(cnt ~., data = train)

#Summary of the model
summary(lr_model)


#prediction of test data

pred_lr = predict(lr_model, test[,-11])

#display actual vs predicted values
temp_df = data.frame("actual"=test[,11], "pred"=pred_lr)
head(temp_df)

#Calculate MAPE
MAPE = function(actual, pred){
 return(mean(abs((actual - pred)/actual)) * 100)
}

Mape <- MAPE(test[,11], pred_lr)
print(Mape) 

error_mat <- regr.eval(trues = test[,11], preds = pred_lr, stats = c("mae","mse","rmse","mape"))
print(error_mat)


####
#****** Decision Tree regressor model *******#
####


#Invoke Decision tree regression model
dt_model = rpart(cnt ~ ., data = train, method = "anova")

#Prediction of test data
pred_dt = predict(dt_model, test[,-11])

#display actual vs predicted values
temp_df = data.frame("actual"=test[,11], "pred"=pred_dt)
head(temp_df)

#calculate error metrices
error_mat <- regr.eval(trues = test[,11], preds = pred_dt, stats = c("mae","mse","rmse","mape"))

#calculate MAPE
Mape <- MAPE(test[,11], pred_dt)
print(Mape)
print(error_mat)

# Visualize the decision tree with rpart.plot
rpart.plot(dt_model, box.palette="RdBu", shadow.col="gray", nn=TRUE)


####
#************** Random Forest regression model **************#
####


#Invoke random forest regression model
rf_model = randomForest(cnt~., data = train, ntree = 80)

#Predict the test cases
pred_rf = predict(rf_model, test[,-11])

#display actual vs predicted values
temp_df = data.frame("actual"=test[,11], "pred"=pred_rf)
head(temp_df)


#calculate error metrices
error_mat <- regr.eval(trues = test[,11], preds = pred_rf, stats = c("mae","mse","rmse","mape"))

#calculate MAPE
Mape <- MAPE(test[,11], pred_rf)
print(Mape)
print(error_mat)
