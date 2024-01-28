## LIBRARIES
library(rpart)   ## FOR Decision Trees
library(rattle)  ## FOR Decision Tree Vis
library(rpart.plot)
library(RColorBrewer)
library(Cairo)
library(network)
library(ggplot2)
##If you install from the source....
#Sys.setenv(NOAWT=TRUE)
## ONCE: install.packages("wordcloud")
library(wordcloud)
#install.packages("randomForest")
library(randomForest)
library(slam)
library(quanteda)
## ONCE: install.packages("quanteda")
## Note - this includes SnowballC
#library(SnowballC)

library(proxy)
## ONCE: if needed:  install.packages("stringr")
library(stringr)
## ONCE: install.packages("textmineR")
library(textmineR)
library(igraph)
library(caret)
#library(lsa)
#install.packages("ROSE")
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

#Balancing our dataset
RecordDF_A_balanced <- ovun.sample(sex ~ ., data = RecordDF_A, method = "under", N = 19564 , seed = 1)$data

RecordDF_A_balanced <- ovun.sample(income ~ ., data = RecordDF_A_balanced, method = "under", N = 8346 , seed = 1)$data
RecordDF_A_balanced


table(RecordDF_A_balanced$age)

#Binning the continuous variables
for (i in 1:NROW(RecordDF_A_balanced)){
  if (RecordDF_A_balanced$age[i] >= 0 & RecordDF_A_balanced$age[i] <= 25){
    RecordDF_A_balanced$age[i] = "Young"}
  else if (RecordDF_A_balanced$age[i] >= 26 & RecordDF_A_balanced$age[i] <= 45){
    RecordDF_A_balanced$age[i] = "Middle-aged"}
  else if (RecordDF_A_balanced$age[i] >= 46 & RecordDF_A_balanced$age[i] <= 65){
    RecordDF_A_balanced$age[i] = "Senior"}
  else {
    RecordDF_A_balanced$age[i] = "Old"}
}

for (i in 1:NROW(RecordDF_A_balanced)){
  if (RecordDF_A_balanced$hours.per.week[i] >= 0 & RecordDF_A_balanced$hours.per.week[i] < 25){
    RecordDF_A_balanced$hours.per.week[i] = "Part-time"}
  else if (RecordDF_A_balanced$hours.per.week[i] >= 25 & RecordDF_A_balanced$hours.per.week[i] < 40){
    RecordDF_A_balanced$hours.per.week[i] = "Full-time-aged"}
  else if (RecordDF_A_balanced$hours.per.week[i] >= 40 & RecordDF_A_balanced$hours.per.week[i] < 60){
    RecordDF_A_balanced$hours.per.week[i] = "Over-time"}
  else {
    RecordDF_A_balanced$hours.per.week[i] = "Excess"}
}

#Converting to factor type
RecordDF_A_balanced$age<-as.factor(RecordDF_A_balanced$age)
RecordDF_A_balanced$hours.per.week<-as.factor(RecordDF_A_balanced$hours.per.week)


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
  
  G <-ggplot(data=RecordDF_A_balanced, aes(.data[[x]], y="") ) +
    geom_bar(stat="identity", aes(fill =.data[[x]])) 
  
  return(G)
}

## Use the function in lappy
lapply(names(RecordDF_A_balanced), function(x) GoPlot(x))

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
## REMOVE THE LABELS from the test set!!! - and keep them
################################################

(TestKnownLabels <- MyTestSET$income)
(MyTestSET <- MyTestSET[ , -which(names(MyTestSET) %in% c("income"))])


###################################################
##
##     Decision Trees
##
##      First - train the model with your training data
##
##      Second - test the model - get predictions - compare
##               to the known labels you have.
###########################################################
MyTrainingSET
str(MyTrainingSET)

## This code uses rpart to create decision tree
## Here, the ~ .  means to train using all data variables
## The MyTrainingSET#label tells it what the label is called
## In this dataset, the label is called "label".

## DT---------------------------------
DT <- rpart(MyTrainingSET$income ~ ., data = MyTrainingSET, method="class")
summary(DT)
plotcp(DT) ## This is the cp plot

(DT_Prediction= predict(DT, MyTestSET, type="class"))
## Confusion Matrix
table(DT_Prediction,TestKnownLabels) ## one way to make a confu mat
## VIS..................
fancyRpartPlot(DT,main="Default Decision Tree")


## DT2-----------------------------
### Example two with cp - a lower cp value is a bigger tree
## Let's make another tree...here we will use cp
DT2<-rpart(MyTrainingSET$income ~ ., data = MyTrainingSET,cp=.005, method="class")
## The small cp the larger the tree if cp is too small you have overfitting
summary(DT2)
(DT_Prediction2= predict(DT2, MyTestSET, type = "class"))
## ANother way to make a confusion matrix
table(DT_Prediction2,TestKnownLabels) ## one way to make a confu mat
fancyRpartPlot(DT2,cex=.6,main="Decision Tree (cp = 0.005)")


##DT3---------------------------------------------------------
## Example three with information gain and lower cp
## Let's make a third tree - here we use cp = 0 and 
## "information" instead of the default which is GINI
DT3<-rpart(MyTrainingSET$income ~ ., 
           data = MyTrainingSET,cp=0.01, method="class",
           parms = list(split="information"),minsplit=2)
## The small cp the larger the tree if cp is too small you have overfitting
summary(DT3)
(DT_Prediction3= predict(DT3, MyTestSET, type = "class"))
table(DT_Prediction3,TestKnownLabels) ## one way to make a confu mat
rattle::fancyRpartPlot(DT3,main="Decision Tree (cp = 0.01)", cex=.6)


##DT4---------------------------------------------------------
## Let's make a 4th tree - but here, we will only use SOME
## of the variables in the dataset to train the model
DT4<-rpart(MyTrainingSET$income ~ age + hours.per.week + education, 
           data = MyTrainingSET,cp=0, method="class",
           parms = list(split="information"),minsplit=2)
## The small cp the larger the tree if cp is too small you have overfitting
summary(DT4)

(DT_Prediction4= predict(DT4, MyTestSET, type = "class"))
table(DT_Prediction4,TestKnownLabels) ## one way to make a confu mat
rattle::fancyRpartPlot(DT4,main="Decision Tree with selective variables (cp = 0)", cex=.65)


DT3$variable.importance  
# before re-eval to add to 100


#Random forest
forest <- randomForest(MyTrainingSET$income ~ .,data = MyTrainingSET,ntree = 20,mtry = 5)
forest

plot(forest, main = "")
varImpPlot(forest)

g <- predict(forest, newdata = MyTrainingSET)
cf <- confusionMatrix(MyTrainingSET$income,g)

cf


varImpPlot(forest)



fit <- rpart(income~. , data =MyTrainingSET)
df <- data.frame(imp = fit$variable.importance)
df2 <- df %>% 
  tibble::rownames_to_column() %>% 
  dplyr::rename("variable" = rowname) %>% 
  dplyr::arrange(imp) %>%
  dplyr::mutate(variable = forcats::fct_inorder(variable))



ggplot2::ggplot(df2) +
  geom_col(aes(x = variable, y = imp),
           fill = "#9E555D",col = "#DD9698", show.legend = F) +
  theme_light()
