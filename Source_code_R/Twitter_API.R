#install.packages("selectr")
#install.packages("rvest")
#install.packages("xml2")

library("selectr")
library("rvest")
library("xml2")

#install.packages("twitteR")
#install.packages("ROAuth")
#install.packages("rtweet")
library(rtweet)
library(twitteR)
library(ROAuth)
library(jsonlite)

#Reading text file containing twitter codes
filename="Twitter_code.txt"
(tokens<-read.csv(filename, header=TRUE, sep=","))

(consumerKey=as.character(tokens$consumerKey))
consumerSecret=as.character(tokens$consumerSecret)
access_Token=as.character(tokens$access_Token)
access_Secret=as.character(tokens$access_Secret)


requestURL='https://api.twitter.com/oauth/request_token'
accessURL='https://api.twitter.com/oauth/access_token'
authURL='https://api.twitter.com/oauth/authorize'

setup_twitter_oauth(consumerKey,consumerSecret,access_Token,access_Secret)
Search1<-twitteR::searchTwitter("#socialwork",n=10000, since="2020-03-01")
(Search_DF2 <- twListToDF(Search1))

(Search_DF2$text[1])

write.csv(Search_DF2,"test_file.csv")


FName = "MyFileExample.txt"
## Start the file
MyFile <- file(FName)
## Write Tweets to file
cat(unlist(Search_DF2), " ", file=MyFile, sep="\n\n\n")
close(MyFile)
