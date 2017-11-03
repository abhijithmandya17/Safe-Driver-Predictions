library(randomForest)
library(foreach)
library(doSNOW)
library(ranger)
#-------------------------------------------DATA CLEANING------------------------------------------------
train1 <- read.csv('train.csv', header=T) #reading in the data
test1 <- read.csv('test.csv', header=T)
sum(is.na(train)) #no NA's

#Combine train and test to clean both 
all_data <- rbind(train1[,-2], test1)

#function to get the mode
getmode <- function(v) {
  uniqv <- unique(v)
  v <- v[v!=-1]
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

#REMOVING -1 VALUES
#Here I am removing -1 values for categorical variables and imputing the mode
#I will also be removing -1 values for numerical variables and imputing the mean

sum(all_data == -1) #There are 2116753 places where -1 is a value

cat1 <- c(3,5:6, 11:14, 17:19, 23:33, 53:58) #create list of cat variables
all_data[cat1] <- lapply(all_data[cat1], as.factor)

num1 <- c(2,4, 7:10,15:16, 20:22, 34:52) #create list of num variables
all_data[num1] <- lapply(all_data[num1], as.numeric)

#Change one hot encoding back to 1 column
colnames(all_data)[7:10] <- c(1,2,3,4)

all_data$ohe <- as.numeric(names(all_data[,7:10])[max.col(all_data[,7:10])])

all_data <- within(all_data, rm("1","2","3","4"))

apply(X= all_data,2,FUN=function(x) length(which(x== -1)))
#ps_ind_05_cat, ps_car_03_cat, ps_car_05_cat, ps_car_07_cat

#Drop above columns
all_data <- within(all_data, rm(ps_ind_05_cat, ps_car_03_cat, ps_car_05_cat, ps_car_07_cat))


#for loop to impute mode for categorical and mean for numerical
for (i in 2:ncol(all_data)){
  if (class(all_data[,i]) == 'factor'){
    all_data[,i][all_data[,i] == -1] <- getmode(all_data[,i])
  }
  else if (class(all_data[,i]) == 'numeric'){
    all_data[,i][all_data[,i] == -1] <- mean(all_data[,i])
  }
}

sum(all_data == -1) #There are now zero instances where -1 is a value

#Drop unused levels
all_data <- droplevels.data.frame(all_data)

#check level count 
col_levels <- lapply(all_data, function(x) nlevels(x))

#Check which column has a level count higher than 53(max levels Random forest works with)
which(col_levels>53) #ps_car_11_cat 

#Drop column from all data
all_data <- within(all_data, rm(ps_car_11_cat))


#Resplitting the data back into train and test
train <- all_data[1:595212,]
test <- all_data[595213:1488028,]
train <- cbind(train1$target, train)
colnames(train)[1] <- "target"


#check if -1
sum(all_data == -1) #0

#Drop unused levels
all_data <- droplevels.data.frame(all_data)

#check level count 
col_levels <- lapply(all_data, function(x) nlevels(x))

#Check which column has a level count higher than 53(max levels Random forest works with)
which(col_levels>53) #ps_car_11_cat 

#Drop column from all data
all_data <- within(all_data, rm(ps_car_11_cat))

#Separate data back to Train and Test
train <- all_data[1:595212,]
test <- all_data[595213:1488028,]
train <- as.data.frame(cbind(train1$target, train))
colnames(train)[1] <- "target"

#Remove unused data frames
rm(train1, test1, all_data)

#Subset of 10,000 observations
set.seed(1)
train_index <- sample(1:nrow(train), 100000, replace= FALSE)
train_subset <- train[train_index,]

#Create data sets for random forest
X.train = train_subset[, -c(1,2)]
Y.train = as.factor(train_subset[, 1])

summary(Y.train)[2]/(summary(Y.train)[1]+summary(Y.train)[2])*100

#Create test set for cross validation using gini scores
test_index <- sample(1:nrow(train), 100000, replace= FALSE)
test_subset <- train[test_index,]
X.test = test_subset[, -c(1,2)]
Y.test = as.factor(test_subset[, 1])

summary(Y.test)[2]/(summary(Y.test)[1]+summary(Y.test)[2])*100

#Samsize was chosen based on 1s in train data set
#Create clusters for paralellizing random forest cross vaidation for different m values and number of trees 
cluster = makeCluster(4, type = "SOCK")
registerDoSNOW(cluster)

set.seed(123)
rf1 <- foreach(mtry = c(3,6,7,49),ntree = c(100,500,1000), .combine="cbind", .multicombine=TRUE,
              .packages='randomForest') %dopar% {
                randomForest(X.train, Y.train, mtry = mtry, ntree = ntree,
                             importance = TRUE,sampsize = c(334,334))
              }

set.seed(12)

#Test with Gini score 
rf_cv <- randomForest(X.train, as.factor(Y.train), mtry = 3, ntree = 1000,
                                            sampsize = c(3615, 3615))
pred_cv <- predict(rf_cv, X.test, type = "prob")
 
preds <- pred_cv[,1]
normalized.gini.index(as.numeric(Y.test), pred_cv[,2])

#based on the cross validation, the best classification was produced using 1000 trees and mtry of 3

#1s in train data
summary(as.factor(train[,1]))[2]#21694

#Running that on whole dataset

rf_final <- randomForest(train[,-c(1,2)], as.factor(train[,1]), mtry = 3, ntree = 1000,
                         sampsize = c(21694, 21694))

#Create Predicitions for submission
pred <- predict(rf_final, test, type = "prob")

prediction <- data.frame(test$id,pred[,2])
colnames(prediction) = c("id", "target")
write.csv(prediction, file = "randomforest.csv", row.names = FALSE)

