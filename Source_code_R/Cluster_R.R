#Reading in the cleaned file
file<-read.csv("income_data.csv")
head(file)

file_without_in = subset(file, select = -c(income) )


#Loading the required packages
library(NbClust)
library(cluster)
library(mclust)
library(amap)  ## for Kmeans (notice the cap K)
library(factoextra) ## for cluster vis, silhouette, etc.
library(purrr)
#install.packages("stylo")
library(stylo)  ## for dist.cosine
#install.packages("philentropy")
library(philentropy)  ## for distance() which offers 46 metrics
## https://cran.r-project.org/web/packages/philentropy/vignettes/Distances.html
library(SnowballC)
library(caTools)
library(dplyr)
library(textstem)
library(stringr)
library(wordcloud)
library(tm) ## to read in corpus (text data)
#install.packages("cl_predict")
#library(cl_predict)

str(file)
#Changing these to numeric
library(dplyr)
file_without_in <- file_without_in %>%
  mutate_all(as.numeric)
str(file_without_in)


#Calculating the distances
(eucl_dist=dist(file_without_in,method = "euclidean"))
#checking if minkowski with power=2 is the same as euclidean
(eucl_dist2=dist(file_without_in,method = "minkowski",p=2))
(man_dist=dist(file_without_in,method = "manhattan"))
(cos_sim <- stylo::dist.cosine(as.matrix(file_without_in)))
(mink_dist=dist(file_without_in,method = "minkowski",p=4))

Hist_E <- hclust(eucl_dist, method="ward.D2")
plot(Hist_E)

Hist_M1 <- hclust(man_dist, method="ward.D2")
plot(Hist_M1)

Hist_CS <- hclust(cos_sim, method="ward.D2")
plot(Hist_CS)

Hist_M2 <- hclust(mink_dist, method="ward.D2")
plot(Hist_M2)



#Before proceeding with kmeans , the optimal number of k is to be known , for that using the elbow method and silhouette method.

#1 Elbow method

fviz_nbclust(
  as.matrix(file_without_in), 
  kmeans, 
  k.max = 10,
  method = "wss",
  FUNcluster = hcut, ##Within-Cluster-Sum of Squared Errors 
  diss = get_dist(as.matrix(file_without_in), method = "euclidean")
)

fviz_nbclust(as.matrix(file_without_in),
             method = "wss",
             k.max = 10,
             FUNcluster = hcut)

#2 Silhouette method
fviz_nbclust(as.matrix(file_without_in),
             method = "silhouette",
             k.max = 10,
             FUNcluster = hcut)
#Elbow method and silhouette roughly suggest k = 4. 
#Will plot kmeans for k=4.

##MOVING ON TO K MEANS
k=4
kmeans_in<-kmeans(file_without_in,k)

(fviz_cluster(kmeans_in, data = file_without_in,
              ellipse.type = "convex",
              #ellipse.type = "concave",
              palette = "jco",
              #axes = c(1, 4), # num axes = num docs (num rows)
              ggtheme = theme_minimal()))


#Predicting a new vector's category
#Each row is a value of 8 values with either a 1 or 2.
set.seed(100)
(R1<-runif(8,1,2))
(R1<-sapply(R1,round))
(R1<-as.data.frame(R1))

#Similarly for R2 and R3
(R2<-runif(8,1,2))
(R2<-lapply(R2,round))
(R2<-as.data.frame(R2))

(R3<-runif(8,1,2))
(R3<-lapply(R3,round))
(R3<-as.data.frame(R3))

#Defining euclidean
euclidean <- function(a, b) sqrt(sum((a - b)^2))
#euclidean(R1,R2)#Works fine
#euclidean(R1,c2)

#For predicting a new category, the distance of this new point with one of the centers must be minimum.
distances=c()
predict<-function(newvec){
  for (i in 1:4){
    centers<-kmeans_in$centers
    ci<-centers[i,]
    val<-euclidean(newvec,ci)
    distances<-append(distances,val)
  }
  return(which(distances==min(distances)))
}

predict(R1) #Indicating R1 belongs to the 8th cluster.
#Clearing distance file and rerunning for R2 
distance=c()
print(paste("The cluster for R2 is",predict(R2))) #Cluster number is 4

distance=c()
predict(R3)# Cluster number is 3
print(paste("The cluster for R3 is",predict(R3))) 