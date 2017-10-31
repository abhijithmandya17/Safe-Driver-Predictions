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

for (i in 2:58){
  if (class(all_data[,i]) == 'factor'){
    all_data[,i][all_data[,i] == -1] <- getmode(all_data[,i])
  }
  else if (class(all_data[,i]) == 'numeric'){
    all_data[,i][all_data[,i] == -1] <- mean(all_data[,i])
  }
}

sum(all_data == -1) #0

train <- all_data[1:595212,]
test <- all_data[595213:1488028,]
train <- as.data.frame(cbind(train1$target, train))
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

#Variable selection by using the step function
subsetstep <- step(sub_lm1, direction = "both")

#First linear model to include all parameters and see if we can do some sort of analysis on independent variables
lm1 <- glm(target~ . -id, data = train, family= "binomial")
summary(lm1)

#Lets plot significant variables against target
plot(train$ps_ind_15, train$target)
plot(train$ps_ind_16_bin, train$target)
plot(train$ps_ind_17_bin, train$target)
plot(train$ps_reg_01, train$target)
plot(train$ps_reg_02, train$target)
plot(train$ps_reg_03, train$target)
plot(train$ps_car_01_cat, train$target)
plot(train$ps_car_03_cat, train$target)
plot(train$ps_car_04_cat, train$target)
plot(train$ps_car_06_cat, train$target)
plot(train$ps_car_07_cat, train$target)
plot(train$ps_car_09_cat, train$target)
plot(train$ps_car_11_cat, train$target)
plot(train$ps_car_12, train$target)
plot(train$ps_car_13, train$target)

#Now let's look for interactions among variables
plot(train$ps_ind_15, train$ps_reg_01)
plot(train$ps_car_01_cat, train$ps_ind_16_bin)
plot(train$ps_car_03_cat, train$ps_ind_17_bin)

#------------------------------------Linear Method/Parametric Approach--------------------------------------

#Identified most significant variables in linear model 1 to break down variables at an individual level
lm2 <- glm(target ~ ps_ind_15 + ps_ind_16_bin + ps_ind_17_bin+ps_reg_01+ps_reg_02+ps_reg_03+
                        ps_car_01_cat+ps_car_03_cat+ps_car_04_cat+ps_car_06_cat+ps_car_07_cat+ps_car_09_cat+
                        ps_car_11_cat+ps_car_12+ps_car_13, data= train_subset, family="binomial")
summary(lm2)

#Break down the model further to try and limit variables for more accurate predictions
#Reduce the noise

lm3 <- glm(target ~ ps_ind_15+ps_ind_17_bin+ps_reg_02+ps_reg_03+ps_car_01_cat+
             ps_car_03_cat+ps_car_04_cat+ps_car_07_cat+ps_car_11_cat+ps_car_13, data= train_subset, 
           family = "binomial")
summary(lm3)
summary(fitted(lm3))

#Break down the model further
#Reduce the noise

lm4 <- glm(target~ ps_ind_15 + ps_reg_02 +ps_reg_03+ ps_car_01_cat+ps_car_03_cat+
             ps_car_07_cat+ ps_car_13, data=train_subset, family = "binomial")
summary(lm4)
summary(fitted(lm4))
#Much more significant model

#------------------------ROC Curve for Visualization
#model <- glm(target ~ ps_reg_01+ps_car_06_cat, data = train, family = "binomial")
#summary(fitted(model))

predict <- predict(lm4, type = 'response')
table(train_subset$target, predict > 0.0001)

#ROC Curve for visualization
library(ROCR)
ROCRpred <- prediction(predict, train_subset$target)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))
#------------------------ROC Curve for Visualization


library(MASS)
step <- stepAIC(lm2, direction= "both")

# Df Deviance     AIC
# <none>                 20718 -308896
# - ps_reg_01       1    20718 -308892
# - ps_ind_16_bin   1    20718 -308890
# - ps_car_12       1    20718 -308889
# - ps_car_06_cat  17    20719 -308885
# - ps_car_04_cat   9    20719 -308869
# - ps_car_11_cat 103    20726 -308869
# - ps_reg_02       1    20719 -308867
# - ps_reg_03       1    20720 -308824
# - ps_car_03_cat   1    20721 -308797
# - ps_car_09_cat   4    20722 -308786
# - ps_car_07_cat   1    20722 -308779
# - ps_ind_15       1    20723 -308744
# - ps_car_01_cat  11    20725 -308724
# - ps_car_13       1    20724 -308719
# - ps_ind_17_bin   1    20728 -308608
step$anova

lmstep <- glm(as.numeric(target)~ ps_reg_01 + ps_ind_16_bin + ps_car_12 + ps_car_06_cat + ps_car_04_cat, 
              data= train)
summary(fitted(lmstep))

step2 <- step(lm2, direction= c("both"))


# Stepwise Regression

fit <- lm(y~x1+x2+x3,data=mydata)
step <- stepAIC(fit, direction="both")
step$anova # display results

#----------------------------------------------Evaluating Models----------------------------------------------
#Here I will be using anova tables and K-fold Cross Validation to identify the best models


# Other useful functions 
coefficients(lm2) # model coefficients
confint(lm2, level=0.95) # CIs for model parameters 
summary(fitted(lm2)) # predicted values
residuals(lm2) # residuals
plot(lm2)
anova(lm2) # anova table

#                 Df Deviance Resid. Df Resid. Dev
# NULL                          594995      20866
# ps_ind_15       1   9.5225    594994      20856
# ps_ind_16_bin   1  10.0572    594993      20846
# ps_ind_17_bin   1  17.5502    594992      20829
# ps_reg_01       1   8.2815    594991      20820
# ps_reg_02       1  16.8527    594990      20804
# ps_reg_03       1   5.3128    594989      20798
# ps_car_01_cat  11  14.3511    594978      20784
# ps_car_03_cat   1   9.6097    594977      20774
# ps_car_04_cat   9  15.3781    594968      20759
# ps_car_06_cat  17   7.5842    594951      20751
# ps_car_07_cat   1   6.0681    594950      20745
# ps_car_09_cat   4   5.3902    594946      20740
# ps_car_11_cat 103  14.6327    594843      20725
# ps_car_12       1   1.2427    594842      20724
# ps_car_13       1   6.2203    594841      20718


#----------------------------------------------Comparing Models----------------------------------------------

# compare models
lm1 <- glm(target~ . -id, data = train, family= "binomial")
lm2 <- glm(target ~ ps_ind_15 + ps_ind_16_bin + ps_ind_17_bin+ps_reg_01+ps_reg_02+ps_reg_03+
             ps_car_01_cat+ps_car_03_cat+ps_car_04_cat+ps_car_06_cat+ps_car_07_cat+ps_car_09_cat+
             ps_car_11_cat+ps_car_12+ps_car_13, data= train_subset, family="binomial")
lm3 <- glm(target ~ ps_ind_15+ps_ind_17_bin+ps_reg_02+ps_reg_03+ps_car_01_cat+
             ps_car_03_cat+ps_car_04_cat+ps_car_07_cat+ps_car_11_cat+ps_car_13, data= train_subset, 
           family = "binomial")
lm4 <- glm(target~ ps_ind_15 + ps_reg_02 +ps_reg_03+ ps_car_01_cat+ps_car_03_cat+
             ps_car_07_cat+ ps_car_13, data=train, family = "binomial")

anova(lm1, lm2)
anova(lm3, lm4)

#lm2 performs better than lmstep
anova(lm2, lmstep)


#---------------------------------------------K-Fold Cross Validation--------------------------------------------
#I am selecting K-fold Cross-Validation for its ease of use and will be 
#using 5 as my k-fold value. To derive this value, I performed three different K-fold CV's using k = 3,
#k=5, and k=10. For the purpose of this assignment, K=5 made the most sense both computationally and
#logically. 

# K-fold cross-validation
library(boot) #necessary library for logistic regression models

#K = 3 in K-fold cross-validation
cv.glm(data = train_subset, lm2, K=3) 

# $call
# cv.glm(data = train, glmfit = lm2, K = 3)
# 
# $K
# [1] 3
# 
# $delta
# [1] 0.0349 0.0349

cv.glm(data = train, lmstep, K=3)

# $call
# cv.glm(data = train, glmfit = lmstep, K = 3)
# 
# $K
# [1] 3
# 
# $delta
# [1] 0.035 0.035

#K = 5 in K-fold cross-validation
cv.glm(data = train, lmstep, K=5)

# $call
# cv.glm(data = train, glmfit = lmstep, K = 5)
# 
# $K
# [1] 5
# 
# $delta
# [1] 0.035 0.035

cv.glm(data = train_subset, lm2, K=5)
# $delta
# [1] 0.03352681 0.03334087

cv.glm(data = train_subset, lm3, K=5)
# $delta
# [1] 0.03339226 0.03323767

cv.glm(data = train_subset, lm4, K=5)
# $delta
# [1] 0.03265526 0.03263143

#lm4 produces the lowest Train MSE


#---------------------------------------------Gini Index--------------------------------------------------
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




#---------------------------------------------Write Output--------------------------------------------------
#Here I am writing to a csv my predictions in the appropriate format

test_predictions = predict(lm4, newdata = test, type = "response")
summary(test_predictions)

summary(train$target)
573518/595482 #96% is 0
21694/595482 #4% is 1
test_predictions <- as.data.frame(test_predictions)
nrow(test_predictions) #892816 rows

#THRESHOLD EVALUATION
test_predictions[test_predictions<0.055] <- 0
test_predictions[test_predictions>=0.055] <- 1

zero <- sum(test_predictions==0) #635095
one <- sum(test_predictions == 1) #257721
zero/nrow(test_predictions) #91% 0's

id <- test$id
target <- test_predictions
prediction <- data.frame(id,target)
colnames(prediction) = c("id", "target")
write.csv(prediction, file = "linear.csv", row.names = FALSE)
