## LIBRARIES
## LIBRARIES
#install.packages("caret")
#install.packages("network")
library(rpart)   ## FOR Decision Trees
library(rattle)  ## FOR Decision Tree Vis
library(rpart.plot)
library(RColorBrewer)
#library(Cairo)
#library(network)
library(ggplot2)
library(tidyverse)
##If you install from the source....
#Sys.setenv(NOAWT=TRUE)
## ONCE: install.packages("wordcloud")
library(wordcloud)
## ONCE: install.packages("tm")

library(slam)
library(quanteda)
## ONCE: install.packages("quanteda")
## Note - this includes SnowballC
#library(SnowballC)

library(proxy)
## ONCE: if needed:  install.packages("stringr")
library(stringr)
## ONCE: 
#install.packages("pheatmap")
library(textmineR)
library(igraph)
library(caret)
library(pheatmap)
#library(lsa)
library(e1071)
library(ROSE)

###############
## Read in the dataset you want to work with....
#################################
#setwd("D:/Anly501")
RecordDatasetName="income_evaluation.csv"

RecordDF_A<-read.csv(RecordDatasetName, stringsAsFactors=TRUE)
head(RecordDF_A)

##########################################
##Cleaning the dataset to the required format
##########################################
#Dropping useless columns
RecordDF_A <- subset(RecordDF_A, select = -c(fnlwgt,relationship,education.num,capital.gain,capital.loss,native.country))

#Checking for NA values
number_NAs = sapply(RecordDF_A, function(x) sum(is.na(x)))
number_NAs

#It can be seen that there are no NA values in this dataset.
nrow(RecordDF_A)

# Let's check data types

str(RecordDF_A)

max(RecordDF_A$age)
#####################
## Our data is already clean and it is MIXED
## data. I will not normalize it.
######
## What is mixed data? It is data made up of many data types
##
#####################################################

## However, let's explore just a little bit to look for 
## BALANCE in the variables AND in the label
##
##  !!!!!! We will also need to remove the "name" column
##   Why??

########################################################

## Simple tables

apply(RecordDF_A, 2, table)  # 2 means columns

hist(RecordDF_A$income)
#Balancing our dataset
RecordDF_A_balanced <- ovun.sample(sex ~ ., data = RecordDF_A, method = "under", N = 19564 , seed = 1)$data

RecordDF_A_balanced <- ovun.sample(income ~ ., data = RecordDF_A_balanced, method = "under", N = 8346 , seed = 1)$data
RecordDF_A_balanced


table(RecordDF_A_balanced$age)


str(RecordDF_A_balanced)
##
## NOTE: Our data and label are balanced pretty well.

## Think about what you see. Are there columns to remove 
## from the data?
## 

## Here is a fancy method to use a function to
## create a bar graph for each variable. 

## Define the function on any dataframe input x
GoPlot <- function(x) {
  
  G <-ggplot(data=RecordDF_A, aes(.data[[x]], y="") ) +
    geom_bar(stat="identity", aes(fill =.data[[x]])) 
  
  return(G)
}

## Use the function in lappy
lapply(names(RecordDF_A), function(x) GoPlot(x))

############################################
## 
## Next - split into TRAIN and TEST data
##
## !!!! Sampling Matters !!!
##
## In our case, we will use random sampling without
## replacement.
##
## Why without replacement?
##
## !!!! IMPORTANT - always clean, prepare, etc. BEFORE
## splitting data into train and test. NEVER after.
##
######################################################
(DataSize=nrow(RecordDF_A_balanced)) ## how many rows?
(TrainingSet_Size<-floor(DataSize*(3/4))) ## Size for training set
(TestSet_Size <- DataSize - TrainingSet_Size) ## Size for testing set

## Random sample WITHOUT replacement (why?)
## set a seed if you want it to be the same each time you
## run the code. The number (like 1234) does not matter
set.seed(1234)

## This is the sample of row numbers
(MyTrainSample <- sample(nrow(RecordDF_A_balanced),
                         TrainingSet_Size,replace=FALSE))

## Use the sample of row numbers to grab those rows only from
## the dataframe....
(MyTrainingSET <- RecordDF_A_balanced[MyTrainSample,])
table(MyTrainingSET$income)

## Use the NOT those row numbers (called -) to get the
## other row numbers not in the training to use to create
## the test set.

## Training and Testing datasets MUST be disjoint. Why?
(MyTestSET <- RecordDF_A_balanced[-MyTrainSample,])
table(MyTestSET$income)

##Make sure your Training and Testing datasets are BALANCED

###########
## NEXT - 
## REMOVE THE LABELS from the set!!! - and keep them
################################################

#Train Set
TrainKnownLabels <- MyTrainingSET$income
MyTrainingSET_NL <- MyTrainingSET[ , -which(names(MyTrainingSET) %in% c("income"))]
head(MyTrainingSET_NL)

#Test Set
TestKnownLabels <- MyTestSET$income
MyTestSET_NL <- MyTestSET[ , -which(names(MyTestSET) %in% c("income"))]
head(MyTestSET_NL)

table(MyTrainingSET$race)
###################################################
##
##    SVM
##
##      First - train the model with your training data
##
##      Second - test the model - get predictions - compare
##               to the known labels you have.
###########################################################
#Polynomial
#####  We can "tune" the SVM by altering the cost ####
tuned_cost_P <- tune(svm,income~., data=MyTrainingSET,
                     kernel="polynomial", 
                     ranges=list(cost=c(.01,.1,1,10,100,100)))
summary(tuned_cost_P)  ## This shows that the best cost is 100


SVM_fit_P <- svm(income~., data=MyTrainingSET, 
                 kernel="polynomial", cost=100, 
                 scale=FALSE)
print(SVM_fit_P)

##Prediction --
pred_P <- predict(SVM_fit_P, MyTestSET_NL, type="class")

(Ptable <- table(pred_P, TestKnownLabels))

#Accuracy
Acc_P = sum(diag(Ptable))/sum(Ptable)
Acc_P

#Misclassification rate for polynomial
(MR_P <- 1 - sum(diag(Ptable))/sum(Ptable))

#Plot
plot(SVM_fit_P, MyTrainingSET, age ~ hours.per.week,slice=list(sex = 1 , education = 1, marital.status = 2,occupation = 10, race = 5, workclass = 1),main = 'Polynomial SVM')
pheatmap(Ptable,display_numbers = T,color = colorRampPalette(c('lightblue','white'))(100),cluster_rows = F, cluster_cols = F,main = "Heatmap of SVM predictions with Polynomial kernel",xlab = 'Prediction',ylab = 'Correct Label')

#Linear Kernel 
#####  We can "tune" the SVM by altering the cost ####
tuned_cost_L <- tune(svm,income~., data=MyTrainingSET,
                     kernel="linear", 
                     ranges=list(cost=c(.01,.1,1,10,100,100)))
summary(tuned_cost_L)  ## This shows that the best cost is .1

SVM_fit_L <- svm(income~., data=MyTrainingSET, 
                 kernel="linear", cost=.1, 
                 scale=FALSE)
print(SVM_fit_L)

##Prediction --
pred_L <- predict(SVM_fit_L, MyTestSET_NL, type="class")

(Ltable <- table(pred_L, TestKnownLabels))

#Accuracy
Acc_L = sum(diag(Ltable))/sum(Ltable)
Acc_L

#Misclassification rate for linear
(MR_L <- 1 - sum(diag(Ltable))/sum(Ltable))

#Plot
plot(SVM_fit_L, MyTrainingSET, age ~ hours.per.week, main = 'Linear SVM')
pheatmap(Ltable,display_numbers = T,color = colorRampPalette(c('lightblue','white'))(100),cluster_rows = F, cluster_cols = F,main = "Heatmap of SVM predictions with Linear kernel",xlab = 'Prediction',ylab = 'Correct Label')



#Radial Kernel 
#####  We can "tune" the SVM by altering the cost ####
tuned_cost_R <- tune(svm,income~., data=MyTrainingSET,
                     kernel="radial", 
                     ranges=list(cost=c(.01,.1,1,10,100,100)))
summary(tuned_cost_R)  ## This shows that the best cost is 10

SVM_fit_R <- svm(income~., data=MyTrainingSET, 
                 kernel="radial", cost=10, 
                 scale=FALSE)
print(SVM_fit_R)

##Prediction --
pred_R <- predict(SVM_fit_R, MyTestSET_NL, type="class")

(Rtable <- table(pred_R, TestKnownLabels))

#Accuracy
Acc_R = sum(diag(Rtable))/sum(Rtable)
Acc_R

#Misclassification rate for radial
(MR_R <- 1 - sum(diag(Rtable))/sum(Rtable))

#Plot
plot(SVM_fit_R, MyTrainingSET, age ~ hours.per.week, main = 'Radial SVM')
pheatmap(Rtable,display_numbers = T,color = colorRampPalette(c('lightblue','white'))(100),cluster_rows = F, cluster_cols = F,main = "Heatmap of SVM predictions with Radial kernel",xlab = 'Prediction',ylab = 'Correct Label')


