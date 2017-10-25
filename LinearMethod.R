#Tyler Lewris
#tal3fj
#Competition 4: Safe Driving

#Linear Method

setwd("~/Documents/UVA/UVA DS /SYS 6018 Data Mining/Competition 4 Safe Driving")

#----------------------------------------------DATA CLEANING------------------------------------------
train <- read.csv('train.csv', header=T) #reading in the data
nrow(train)
head(train)
summary(train)
sum(is.na(train)) #no NA's

cat <- c(2,4,6:15, 18:20, 24:34, 54:59) #create list of cat variables
train[cat] <- lapply(train[cat], as.factor)

num <- c(3,5, 16:17, 21:23, 35:53) #create list of num variables
train[num] <- lapply(train[num], as.numeric)

sapply(train, class) #check class of all columns

train[train == -1] <- 0
# train <- na.omit(train)
nrow(train)

#----------------------------------------------Linear Method----------------------------------------------
lm1 <- glm(as.numeric(target)~ ., data = train)
summary(lm1)

step(lm1, "both")

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

#Write output
prediction <- data.frame(id,target)
write.csv(prediction, file = "basic.csv", row.names = FALSE)
