#Tyler Lewris
#tal3fj
#Competition 4: Safe Driving

#Linear Method

setwd("~/Documents/UVA/UVA DS /SYS 6018 Data Mining/Competition 4 Safe Driving")

# Grading Criteria
# Data exploration
# Data cleaning (missing values, outliers, etc.)
# Rationale for the selected statistical modeling methods
# Correct implementation and use of statistical modeling methods
# Appropriate model selection approach (train/test, cross-validation, etc.)
# Thoroughly documented code (R comments)
# Appropriate use of functions to reduce code complexity and redundancy
# Clear thinking about the reflection questions

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

cat1 <- c(3,5:14, 17:19, 23:33, 53:58) #create list of cat variables
all_data[cat1] <- lapply(all_data[cat1], as.factor)

num1 <- c(2,4, 15:16, 20:22, 34:52) #create list of num variables
all_data[num1] <- lapply(all_data[num1], as.numeric)

#for loop to impute mode for categorical and mean for numerical
for (i in 2:58){
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
col_levels <- lapply(all_data[,c(3,5:14, 17:19, 23:33, 53:58)], function(x) nlevels(x))

#Check which column has a level count higher than 53(max levels Random forest works with)
which(col_levels>53) #ps_car_11_cat 

#Drop column from all data
all_data <- within(all_data, rm(ps_car_11_cat))

#Resplitting the data back into train and test
train <- all_data[1:595212,]
test <- all_data[595213:1488028,]
train <- cbind(train1$target, train)
colnames(train)[1] <- "target"

#-------------------------------------------DATA EXPLORATION------------------------------------------------
#First, I am going to subset the data as the full training data has 600,000 observations. Far more computationally
#demanding than my personal laptop can handle. Subsetting the data will allow for a faster analysis
#of linear models

#Subset of 10,000 observations
train_subset <- sample(1:nrow(train), 10000, replace= FALSE)
train_subset <- train[train_subset,]

#Linear model with all variables
sub_lm1 <- glm(target~. -id, data= train_subset, family="binomial")
summary(sub_lm1)
vif(sub_lm1)
#Removing variables based on vif

#Remove -ps_ind_09_bin - ps_ind_14 - ps_ind_16_bin -ps_ind_17_bin -ps_ind_18_bin

#Now let's look for interactions among variables
plot(train$ps_ind_15, train$ps_reg_01)
plot(train$ps_car_01_cat, train$ps_ind_16_bin)
plot(train$ps_car_03_cat, train$ps_ind_17_bin)

#------------------------------------Linear Method/Parametric Approach--------------------------------------
#NOTE: This is a continuation of the Data Exploration component of our project as I use 
#step functions on the subset data to identify important predictors

#Running Linear Models on FULL data set to see if the subset models are just as significant

lmall <- glm(target~. -id, data = train, family="binomial")
summary(lmall)

#Break down the model further to try and limit variables for more accurate predictions
#Reduce the noise

#List of 19 significant variables with p-values < 0.05.

#ps_ind_02_cat
#ps_ind_03
#ps_ind_04_cat
#ps_ind_05_cat
#ps_ind_07_bin
#ps_ind_08_bin
#ps_ind_15
#ps_ind_17_bin
#ps_reg_01
#ps_reg_03
#ps_car_01_cat
#ps_car_04_cat
#ps_car_06_cat
#ps_car_07_cat
#ps_car_09_cat
#ps_car_12
#ps_car_13      
#ps_car_14
#ps_car_15

lm1 <- glm(target~ ps_ind_02_cat + ps_ind_03 +ps_ind_04_cat + ps_ind_05_cat+ ps_ind_07_bin + 
            ps_ind_08_bin + ps_ind_15 + ps_ind_17_bin + ps_reg_01 + ps_reg_03 + ps_car_01_cat+ 
            ps_car_04_cat + ps_car_06_cat + ps_car_07_cat + ps_car_09_cat + ps_car_12 + ps_car_13 +
            ps_car_14 + ps_car_15, data = train, family = "binomial")
summary(lm1)
#So here, all variables have significance < 0.05. However, there are two variables that
#have many factor levels that are not significant - remove those

#List of 15 variables

#   ps_ind_02_cat
#   ps_ind_03
#   ps_ind_04_cat
#   ps_ind_05_cat
#   ps_ind_07_bin
#   ps_ind_08_bin
#   ps_ind_15
#   ps_reg_01
#   ps_reg_03
#   ps_car_07_cat
#   ps_car_09_cat
#   ps_car_12
#   ps_car_13
#   ps_car_14
#   ps_car_15

#Break down the model further to try and limit variables for more accurate predictions
#Reduce the noise

lm2 <- glm(target~ ps_ind_02_cat+ps_ind_03+ ps_ind_04_cat+ps_ind_05_cat+ps_ind_07_bin+ps_ind_08_bin+
             ps_ind_15+ps_reg_01+ps_reg_03+ps_car_07_cat+ps_car_09_cat+ps_car_12+
             ps_car_13+ps_car_14+ps_car_15, data = train, family = "binomial")
summary(lm2)

#So, I have identified a list of 15 variables that are significant in this logistic regression. 
#I will now further break down these models by subsetting the data and using step functions.

#-----------------------------------Evaluating and Comparing Models Section 1----------------------------------------------

library(MASS)

#Running the linear models on a subset of the data for ease of use with step functions to evaluate
#predictor importance based on AIC scores

sub_lm1 <- glm(target~ ps_ind_02_cat + ps_ind_03 +ps_ind_04_cat + ps_ind_05_cat+ ps_ind_07_bin + 
            ps_ind_08_bin + ps_ind_15 + ps_ind_17_bin + ps_reg_01 + ps_reg_03 + ps_car_01_cat+ 
            ps_car_04_cat + ps_car_06_cat + ps_car_07_cat + ps_car_09_cat + ps_car_12 + ps_car_13 +
            ps_car_14 + ps_car_15, data = train_subset, family = "binomial")

step1 <- stepAIC(sub_lm1, direction= "both")

# Step:  AIC=3023.07
# target ~ ps_ind_05_cat + ps_ind_07_bin + ps_ind_08_bin + ps_ind_15 + 
#   ps_ind_17_bin + ps_reg_01 + ps_reg_03 + ps_car_07_cat + ps_car_13 + 
#   ps_car_14
# 
# Df Deviance    AIC
# <none>               2991.1 3023.1
# - ps_ind_08_bin  1   2993.2 3023.2 *
# - ps_car_14      1   2993.2 3023.2 *
# - ps_reg_03      1   2993.2 3023.2 *
# + ps_ind_04_cat  1   2990.2 3024.2
# + ps_car_15      1   2990.3 3024.3
# + ps_car_12      1   2990.6 3024.6
# + ps_ind_03      1   2991.0 3025.0
# - ps_car_07_cat  1   2995.4 3025.4 
# - ps_ind_07_bin  1   2995.4 3025.4 *
# - ps_ind_15      1   2995.8 3025.8 *
# - ps_ind_17_bin  1   2996.4 3026.4 *
# - ps_reg_01      1   2997.7 3027.7 *
# + ps_ind_02_cat  3   2989.8 3027.8
# + ps_car_09_cat  4   2988.1 3028.1
# - ps_car_13      1   2999.8 3029.8 *
# + ps_car_01_cat 11   2976.6 3030.6
# + ps_car_04_cat  9   2980.8 3030.8
# - ps_ind_05_cat  6   3011.3 3031.3 *
# + ps_car_06_cat 17   2978.8 3044.8

sub_lm2 <- glm(target~ ps_ind_02_cat+ps_ind_03+ ps_ind_04_cat+ps_ind_05_cat+ps_ind_07_bin+ps_ind_08_bin+
             ps_ind_15+ps_reg_01+ps_reg_03+ps_car_07_cat+ps_car_09_cat+ps_car_12+
             ps_car_13+ps_car_14+ps_car_15, data = train_subset, family = "binomial")

step2 <- stepAIC(sub_lm2, direction= "both")

# Step:  AIC=3026.44
# target ~ ps_ind_05_cat + ps_ind_07_bin + ps_ind_08_bin + ps_ind_15 + 
#   ps_reg_01 + ps_reg_03 + ps_car_07_cat + ps_car_13 + ps_car_14
# 
# Df Deviance    AIC
# <none>               2996.4 3026.4
# - ps_reg_03      1   2998.5 3026.5 *
# - ps_ind_08_bin  1   2999.0 3027.0 *
# - ps_car_14      1   2999.2 3027.2 *
# + ps_ind_04_cat  1   2995.6 3027.6
# + ps_car_15      1   2995.9 3027.9
# + ps_car_12      1   2996.1 3028.1
# + ps_ind_03      1   2996.3 3028.3
# - ps_ind_07_bin  1   3001.0 3029.0 *
# - ps_car_07_cat  1   3001.1 3029.1 *
# - ps_ind_15      1   3001.3 3029.3 *
# + ps_car_09_cat  4   2993.1 3031.1
# + ps_ind_02_cat  3   2995.2 3031.2
# - ps_reg_01      1   3003.9 3031.9 *
# - ps_car_13      1   3006.1 3034.1 *
# - ps_ind_05_cat  6   3016.9 3034.9 *

#We have now identified the most predictive variables based off a stepwise regression and will
#use these variables for our third and final model

# target ~ ps_ind_05_cat + ps_ind_07_bin + ps_ind_08_bin + ps_ind_15 + 
#   ps_ind_17_bin + ps_reg_01 + ps_reg_03 + ps_car_07_cat + ps_car_13 + 
#   ps_car_14

#------------------------------------------Comparing Models Section 2----------------------------------------------

# compare models
#lm1 copied from above
lm1 <- glm(target~ ps_ind_02_cat + ps_ind_03 +ps_ind_04_cat + ps_ind_05_cat+ ps_ind_07_bin + 
             ps_ind_08_bin + ps_ind_15 + ps_ind_17_bin + ps_reg_01 + ps_reg_03 + ps_car_01_cat+ 
             ps_car_04_cat + ps_car_06_cat + ps_car_07_cat + ps_car_09_cat + ps_car_12 + ps_car_13 +
             ps_car_14 + ps_car_15, data = train, family = "binomial")
#lm2 copied from above
lm2 <- glm(target~ ps_ind_02_cat+ps_ind_03+ ps_ind_04_cat+ps_ind_05_cat+ps_ind_07_bin+ps_ind_08_bin+
             ps_ind_15+ps_reg_01+ps_reg_03+ps_car_07_cat+ps_car_09_cat+ps_car_12+
             ps_car_13+ps_car_14+ps_car_15, data = train, family = "binomial")
#lm3 is the model taken from our stepwise regression
lm3 <- glm(target ~ ps_ind_05_cat + ps_ind_07_bin + ps_ind_08_bin + ps_ind_15 + 
             ps_ind_17_bin + ps_reg_01 + ps_reg_03 + ps_car_07_cat + ps_car_13 + 
             ps_car_14, data = train, family = "binomial")


anova(lm1, lm2)
anova(lm1, lm3)
anova(lm2, lm3)
#anova tables confirm lm3 is our best model

#Lets further evaluate lm3
coefficients(lm3) # model coefficients
confint(lm3, level=0.95) # CIs for model parameters 
summary(fitted(lm3)) # predicted values
residuals(lm3) # residuals
plot(lm3)
anova(lm3) # anova table

#------------------------ROC Curve for Visualization
#model <- glm(target ~ ps_reg_01+ps_car_06_cat, data = train, family = "binomial")
#summary(fitted(model))

predict <- predict(lm3, type = 'response')
summary(predict)

#ROC Curve for visualization
library(ROCR)
ROCRpred <- prediction(predict, train$target)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))
#------------------------ROC Curve for Visualization

#---------------------------------------------K-Fold Cross Validation--------------------------------------------
#I am selecting K-fold Cross-Validation for its ease of use and will be 
#using 5 as my k-fold value. To derive this value, I performed three different K-fold CV's using k = 3,
#k=5, and k=10. For the purpose of this assignment, K=5 made the most sense both computationally and
#logically. 

# K-fold cross-validation
library(boot) #necessary library for logistic regression models

#K = 5 in K-fold cross-validation 
cv.glm(data = train, lm1, K=5)
#$delta
#[1] 0.03487166 0.03487034


#K = 5 in K-fold cross-validation 
cv.glm(data = train, lm2, K=5)
#$delta
#[1] 0.03488396 0.03488343


#K = 5 in K-fold cross-validation 
cv.glm(data = train, lm3, K=5)
#$delta
#[1] 0.03486965 0.03486930

#lm3 produces the lowest Train MSE



#THRESHOLD EVALUATION
#Now that we have selected a linear model, I'd like to see which threshold gives us the highest
#predictive power based on our train data
deltas= rep(NA, 3) #create empty list of 10 

for (i in 0:1) {
  predict <- predict(lm3, type = 'response')
  deltas[i] = cv.glm(data = train, lm3, K=5)$delta[1]
}

plot(0:1, deltas)
which.min(deltas)
points(which.min(deltas), deltas[which.min(deltas)], col="blue", cex=2, pch=20)

#Threshold value of 0.036 will be used.
#This means that any prediction < 0.036 will be assigned a binary value of 0
#and any prediciton >= 0.036 will be assigned a binary value of 1


#---------------------------------------------Gini Index--------------------------------------------------
#Here I am using an unnormalized.gini.index function to calculate the potential score I would receive
#by submitting my predictions on Kaggle

#' Calculates unnormalized Gini index from ground truth and predicted probabilities.
#' @param ground.truth Ground-truth scalar values (e.g., 0 and 1)
#' @param predicted.probabilities Predicted probabilities for the items listed in ground.truth
#' @return Unnormalized Gini index.
unnormalized.gini.index = function(ground.truth, predicted.probabilities) {
  
  if (length(ground.truth) !=  length(predicted.probabilities))
  {
    stop("Actual and Predicted need to be equal lengths!")
  }
  
  # arrange data into table with columns of index, predicted values, and actual values
  gini.table = data.frame(index = c(1:length(ground.truth)), predicted.probabilities, ground.truth)
  
  # sort rows in decreasing order of the predicted values, breaking ties according to the index
  gini.table = gini.table[order(-gini.table$predicted.probabilities, gini.table$index), ]
  
  # get the per-row increment for positives accumulated by the model 
  num.ground.truth.positivies = sum(gini.table$ground.truth)
  model.percentage.positives.accumulated = gini.table$ground.truth / num.ground.truth.positivies
  
  # get the per-row increment for positives accumulated by a random guess
  random.guess.percentage.positives.accumulated = 1 / nrow(gini.table)
  
  # calculate gini index
  gini.sum = cumsum(model.percentage.positives.accumulated - random.guess.percentage.positives.accumulated)
  gini.index = sum(gini.sum) / nrow(gini.table) 
  return(gini.index)
}

#' Calculates normalized Gini index from ground truth and predicted probabilities.
#' @param ground.truth Ground-truth scalar values (e.g., 0 and 1)
#' @param predicted.probabilities Predicted probabilities for the items listed in ground.truth
#' @return Normalized Gini index, accounting for theoretical optimal.
normalized.gini.index = function(ground.truth, predicted.probabilities) {
  
  model.gini.index = unnormalized.gini.index(ground.truth, predicted.probabilities)
  optimal.gini.index = unnormalized.gini(ground.truth, ground.truth)
  return(model.gini.index / optimal.gini.index)
}

#Unnormalized gini index to calculate the potential Kaggle submission score
unnormalized.gini.index(train$target, predict)

#---------------------------------------------Write Output--------------------------------------------------
#Here I am writing my predictions to a csv in the correct Kaggle submission format

test_predictions = predict(lm3, newdata = test, type = "response")
summary(test_predictions)

test_predictions <- as.data.frame(test_predictions)

#THRESHOLD EVALUATION - chosen from above
test_predictions[test_predictions<0.036] <- 0
test_predictions[test_predictions>=0.036] <- 1

id <- test$id
target <- test_predictions
prediction <- data.frame(id,target)
colnames(prediction) = c("id", "target")
write.csv(prediction, file = "linear.csv", row.names = FALSE)
