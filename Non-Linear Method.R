library(randomForest)
library(foreach)
library(doSNOW)

#-------------------------------------------DATA CLEANING------------------------------------------------
train1 <- read.csv('train.csv', header=T) #reading in the data
test1 <- read.csv('test.csv', header=T)

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
sum(all_data == -1) #2116753

cat1 <- c(3,5:14, 17:19, 23:33, 53:58) #create list of cat variables
all_data[cat1] <- lapply(all_data[cat1], as.factor)

num1 <- c(2,4, 15:16, 20:22, 34:52) #create list of num variables
all_data[num1] <- lapply(all_data[num1], as.numeric)

#Impute Mode for factor/Mean for numeric where row value = -1 or missing
for (i in 2:58){
  if (class(all_data[,i]) == 'factor'){
    all_data[,i][all_data[,i] == -1] <- getmode(all_data[,i])
  }
  else if (class(all_data[,i]) == 'numeric'){
    all_data[,i][all_data[,i] == -1] <- mean(all_data[,i])
  }
}

#Check if -1
sum(all_data == -1) #0

#Drop unused levels
all_data <- droplevels.data.frame(all_data)

#check level count 
col_levels <- lapply(all_data[,c(3,5:14, 17:19, 23:33, 53:58)], function(x) nlevels(x))

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
train_subset <- sample(1:nrow(train), 10000, replace= FALSE)
train_subset <- train[train_subset,]

#Create data sets for random forest
X.train = train_subset[, -1]
Y.train = as.factor(train_subset[, 1])

summary(Y.train)[2]/(summary(Y.train)[1]+summary(Y.train)[1])*100

#Since the data is so imbalanced, a large m tends to give more acurate predictions.
#ntrees was fixed at 500 to save computational effort and cross vaidation producedno imporvement 
#Samsize was chosen based on 1s in train data set
#Create clusters for paralellizing random forest cross vaidation for different cut offs and number of trees 

cluster = makeCluster(4, type = "SOCK")
registerDoSNOW(cluster)

set.seed(123)
rf1 <- foreach(c =c(0.75, 0.80,0.85),ntree = c(100,500,1000), .combine="combine", .multicombine=TRUE,
              .packages='randomForest') %dopar% {
                randomForest(X.train, Y.train, mtry = 57, ntree = ntree,
                             importance = TRUE, cutoff =c(c,(1-c)),sampsize = c(382*4,382))
              }
#based on the cross validation, the best classification was produced using a cutoff of 0.80/0.20 and 500 trees

#1s in train data
summary(as.factor(train[,1]))[2]#21694

#Running that on whole dataset

rf_final <- randomForest(train[,-1], as.factor(train[,1]), mtry = 57, ntree = 500,
                         importance = TRUE, cutoff =c(0.80,0.20),sampsize = c(21694*4,21694))

#Make prediction and create submission
pred <- predict(rf_final, test, type="response")

prediction <- data.frame(test$id,pred)
colnames(prediction) = c("id", "target")
write.csv(prediction, file = "randomforest.csv", row.names = FALSE)
