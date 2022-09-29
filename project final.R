library(ggplot2)
library(ROCR)
library(caTools)
library(caret)
library(tidyverse)
library(gmodels)
library(pROC)
library(rpart)
library(rattle)
library(dplyr)
library(e1071)
library(corrplot)
library(prediction)
library(class)

#read the csv file
news <- read.csv("OnlineNewsPopularity1.csv", stringsAsFactors = FALSE)
head(news)

# Basic opreations
str(news)
summary(news$shares)
ggplot(news)+
  aes(x="news",y=shares)+
  geom_boxplot(fill="green")+
  theme_minimal()

# Check for any missing values:
sum(is.na(news))

#for removing outliers
news <- news[!news$n_unique_tokens==701,]

# Remove url and timedelta from the dataset using subset, as url and timedelta are non predictive variables 
news <- subset(news, select = -c(url, timedelta))
news$shares<- as.numeric(news$shares)
news_predict<-news

# Heat Map for Correlation beten all variables
cormatrix <- cor(news_predict)
heatmap(cormatrix)

#From the above heatmap it can be observed that certain groups of variables are close to each other.
#However, all variables have low correlation with the target variable - number of shares.



#Combining Plots for EDA for visual analysis
par(mfrow=c(2,2))
for(i in 2:length(news)){hist(news[,i],
                              xlab=names(news)[i] , main = paste("[" , i , "]" ,
                                                                 "Histogram of", names(news)[i])  )}

# From the above data distributions, the following can be observed:-
# 1. there is outlier in "n_unique_tokens", "n_non_stop_words", "n_non_stop_unique_tokens" and "average_token length".
# 2. The dataset could possibly have some missing values which might be coded as 0 and hence difficult to find
# 3. Based on the distributions, it can be seen that the data is skwed to some extent.


# Determining the effect of weekdays and weekends on number of shares
for (i in 31:37){
  boxplot(log(news$shares) ~ (news[,i]), xlab=names(news)[i] , ylab="shares")
}
# Observing the above box plots, it can be said that weekdays show some effect on shares.However, they can also be chosen to be not considered due to the very small effect. 
# However, weekends seem to have more considerable effect.


# Determining the effect of news categories on shares
for (i in 12:17){
  boxplot(log(news$shares) ~ (news[,i]), xlab=names(news)[i] , ylab="shares")
}
#Observing the boxplots, news categories also seem to have very small effect on the number of shares and can be ignored if required.

# Converting shares to response varaible with 1400 as median value 
news$shares <- as.factor(ifelse(news$shares >1400,1,0))

# get the number of 1's and 0's
table(news$shares)

# Results in percentage form
round(prop.table(table(news$shares)) * 100, digits = 1)

##APPLYING NAIVE BAYES ALGORITHAM

# Data Splicing
set.seed(1)
news_nb <- sample(2, nrow(news), replace=TRUE, prob=c(0.80, 0.20))

# Creating seperate dataframe for predictors
news.training <- news[news_nb==1, 1:59]
news.test <- news[news_nb==2, 1:59]
dim(news.testLabels)
# Creating seperate dataframe for 'shares' feature which is our target.
news.trainLabels <- news[news_nb==1, 59]
news.testLabels <- news[news_nb==2, 59]

# Training Naive Bayes Model
nbmodel <- naiveBayes(news.trainLabels ~ ., data=news.training)
nbmodel

# From the training data results it can be seen than 49.48% of the articles are popular

# Predicting on test data
nbprediction <- predict(nbmodel,news.test)
nbprob <- predict(nbmodel,news.test,type="raw",drop=F)

# Confusion matrix to check accuracy
confnb<-table(nbprediction,news.testLabels)
confnb

#  re able to classify 3699 out of 3966 "not popular" cases correctly
#  re able to classify 3613 out of 3927 "popular" cases correctly
# Hence the ability of Naive Bayes to predict the articles that are not popular is 93.27% and the ability to predict popular articles is 91.07%

#checking for accuracy 
Accuracy <- sum(diag(confnb))/sum(confnb)
Accuracy
summary(confnb)

#here the accuracy is 92.63%.

# Following is another code for confusion matrix
conf<-confusionMatrix(nbprediction,news.testLabels)

# Precision
nbprecision<-conf$byClass['Pos Pred Value'] 
nbprecision

# Recall
nbrecall<-conf$byClass['Sensitivity']
nbrecall

# F score
nb_fscore<-2*((nbprecision*nbrecall)/(nbprecision+nbrecall))
nb_fscore

************************************************************************************************************************************

  #LOGISTIC REGRESSION
  # Spliting the Data into Train and test. Since the dataset is large the split is 80% for training and 20% for test. 
set.seed(2)
news_dataset <- sample(2, nrow(news), replace=TRUE, prob=c(0.80, 0.20))

# Training and Test datasets.
news.training <- news[news_dataset==1,]
news.test <- news[news_dataset==2,]

# Training and Test labels.
news.trainLabels <- news[news_dataset==1, 59]
news.testLabels <- news[news_dataset==2, 59]

# Building a logistic regression model with all predictors.
model1=glm( shares~.,data =news.training,family='binomial')

# Looking at the models performace
summary(model1)

##  get the important features for the data but lets improve the model further and reduce the non significant features.

# Build a predict model with type response to calcualte accuracy and misclassification rate.
t1= predict(model1,news.test,type = 'response')
head(t1)


# Histogram of Predict response to find approporiate cut off.
hist(t1)

# Creating a confusion matrix
test1<- ifelse(t1>0.5,1,0)
testtable=table(test1,news.testLabels)
testtable

# calculating the Misclassification Rate
misClassificationrate=1-sum(diag(testtable))/sum(testtable)
misClassificationrate

# Calcualting Accuracy of the model.
accuracyrate=round(sum(diag(testtable))/sum(testtable),2)
accuracyrate

#Storing the accuracy 
logistic_accuracy<-accuracyrate
logistic_accuracy

# precision
logistic_precision<- round(testtable[1,1]/sum(testtable[,1]),2)
print(paste0("Precision for Logistic regression is: ", logistic_precision))

# Recall
logistic_recall<- round(testtable[1,1]/sum(testtable[,1]),2)
print(paste0("Recall for Logistic Regression: ", logistic_recall))

# F-Score
logistic_fscore<- round(2*(logistic_precision*logistic_recall)/(logistic_precision+logistic_recall),2)
print(paste0("F-Score for Logistic Regression: ", logistic_fscore))

# Buliding an ROC curve
t1= predict(model1,news.test,type = 'response')
t1= prediction(t1, news.testLabels)
roc<-performance(t1,'tpr','fpr')

plot(roc,colorize=T,main='ROC Curve',ylab='Sensitivity',xlab='1-Specificity')
abline(a=0,b=1)

# Area under the curve
auc <- performance(t1,'auc')
auc <- unlist(slot(auc,'y.values'))
auc <- round(auc,3)
auc
legend(0.6,0.4,auc,title = 'AUC',cex = 0.8)

Models<- c("NAIVE BAYES","LoGISTIC REGRESSION")

Accuracies<- c(Accuracy,logistic_accuracy)
Precisions<- c(nbprecision,logistic_precision)
Recall<- c(nbrecall,logistic_recall)
f_Measure<- c(nb_fscore,logistic_fscore)

Final_table<-rbind(Models,Accuracies,Precisions, Recall, f_Measure)
Final_table


