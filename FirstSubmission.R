#Basic First Submission
setwd("~/Documents/UVA/UVA DS /SYS 6018 Data Mining/Competition 4 Safe Driving")

test <- read.csv('test.csv', header=T) #reading in the data
id <- test$id
target <- seq(0, 1, 1/892815)
prediction <- data.frame(id,target)
write.csv(prediction, file = "basic.csv", row.names = FALSE)