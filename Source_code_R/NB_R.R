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
MyTrainingSET <- MyTrainingSET[ , -which(names(MyTrainingSET) %in% c("income"))]
head(MyTrainingSET)

#Test Set
TestKnownLabels <- MyTestSET$income
MyTestSET <- MyTestSET[ , -which(names(MyTestSET) %in% c("income"))]
head(MyTestSET)


###################################################
##
##    Naive Bayes
##
##      First - train the model with your training data
##
##      Second - test the model - get predictions - compare
##               to the known labels you have.
###########################################################
str(MyTrainingSET)

NB<-naiveBayes(MyTrainingSET, 
               TrainKnownLabels, 
               laplace = 1)

summary(NB)

## Predict --------------------------------

NB_Pred <- predict(NB, MyTestSET)
NB_Pred[1:10]

data1 = table(NB_Pred,TestKnownLabels) ## one way to make a confu mat
pheatmap(data1,display_numbers = T,color = colorRampPalette(c('white','green'))(100),cluster_rows = F, cluster_cols = F)

confusionMatrix(NB_Pred,TestKnownLabels)

#Cross Validation
x <- MyTrainingSET 
y <- TrainKnownLabels
model = train(x,y,'nb')
#,trControl=trainControl(method='cv',number=10))
model$results

Predict <- predict(model,MyTestSET )
table(Predict,TestKnownLabels)
#Plot Variable performance
X <- varImp(model)
plot(X)

