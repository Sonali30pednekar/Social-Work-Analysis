##install.packages("psych")
library(psych)
library(ggplot2)
library(plotly)
library(corrplot)

#Read in the dataset
dataset_Name<-"Age by Gender.csv"
age_by_gender <- read.csv(dataset_Name, na.string=c("", " "))

##print
head(age_by_gender, n=5)

## Have a look at the data types
str(age_by_gender)

## Get column names

(ColNames<-names(age_by_gender))

for(name in 1:length(ColNames)){
  cat(ColNames[name], "\n")
}

#Number of rows and columns before cleaning
(NumColumns <-ncol(age_by_gender))
(NumRows <-nrow(age_by_gender))

#Let's make tables of all the columns

lapply(age_by_gender,table)  
lapply(age_by_gender,summary) 

#Remove columns we do not need or want
(age_by_gender  <- subset(age_by_gender, select=-c(Gender)))
(age_by_gender  <- subset(age_by_gender, select=-c(ID.Age)))
(age_by_gender  <- subset(age_by_gender, select=-c(ID.Year)))
(age_by_gender  <- subset(age_by_gender, select=-c(ID.Workforce.Status)))
(age_by_gender  <- subset(age_by_gender, select=-c(ID.PUMS.Occupation)))
(age_by_gender  <- subset(age_by_gender, select=-c(Slug.PUMS.Occupation)))



str(age_by_gender)

# Checking for missing values
# Check the entire DF for missing values in total
is.na(age_by_gender) 

lapply(age_by_gender, is.na) 

## Using an inline function and sapply (for simplify apply)
sapply(age_by_gender, function(x) sum(is.na(x)))

#Formatting
age_by_gender$ID.Gender <- factor(age_by_gender$ID.Gender)
age_by_gender$Year <- factor(age_by_gender$Year)
age_by_gender$PUMS.Occupation <- factor(age_by_gender$PUMS.Occupation)
age_by_gender$Workforce.Status <- factor(age_by_gender$Workforce.Status)
age_by_gender$Age <- as.numeric(age_by_gender$Age)
age_by_gender$Total.Population <- as.numeric(age_by_gender$Total.Population)
age_by_gender$Record.Count <- as.numeric(age_by_gender$Record.Count)

str(age_by_gender)


table(age_by_gender$Age)  

## mean, variance 
summary(age_by_gender$Age)
nrow(age_by_gender)
sd(age_by_gender$Age, na.rm = T)

pairs(age_by_gender[,c(2,5)],na.rm = T, col = "blue")

lapply(age_by_gender,summary) 

(Temp<-age_by_gender[,c(2,5,9)])

pairs.panels(Temp)
corr.test(Temp, method = "spearman")

summary(age_by_gender)

age_by_gender[,c(2,5,6,7,9)]<-lapply(age_by_gender[,c(2,5,6,7,9)], as.integer)

str(age_by_gender)

#Visualizations
##Pie CHart of Gender
ggplot(age_by_gender, aes(x = "", fill = ID.Gender)) + 
  geom_bar(width = 1) +
  labs(fill="ID.Gender", title="Pie Chart of Gender")+
  coord_polar(theta = "y", start=0)

## Boxplot of Age
ggplot(data = age_by_gender, aes(y = Age)) +
  geom_boxplot(varwidth=T, fill="plum") +
  labs(title="Age Boxplot",x="Age",y="Count")+
  scale_x_discrete() +
  ylim(0,100)

##Correlation
corr_data<-age_by_gender[,c(2,5,6,7)]
corr_res<-cor(corr_data)
corrplot(corr_res,type="upper",order = "hclust",title = "Correlation between Pclass and Fare")
corr_res
write.table(corr_res,file = 'corr_res.txt',sep = ",",quote = FALSE,row.names = F)

##Scatterplot of Population
ggplot(age_by_gender, aes(x = Total.Population, y = 1:nrow(age_by_gender))) + 
  geom_point()+
  labs(y="Count", 
       x="Population", 
       title="Scatterplot of Population")

#Final Dataset
write.csv(age_by_gender,"cleaned_age_gender.csv", row.names = FALSE)
