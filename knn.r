########################### problem1 #######################
#Problem Statement: -
#A glass manufacturing plant, uses different Earth elements to design a new glass based on customer requirements for that they would like to automate the process of classification as it's a tedious job to manually classify it, help the company reach its objective by correctly classifying the Earth elements, by using KNN Algorithm

#load the dataset
glass <- read.csv("C:\\Users\\DELL\\Downloads\\glass.csv")

# Exploratory Data Analysis
# table of Type
table(glass$Type)
str(glass$Type)

# recode Type as a factor
glass$Type <- factor(glass$Type)
str(glass$Type)

# table or proportions with more informative labels
round(prop.table(table(glass$Type)) * 100, digits = 2)

#scaling the data except the output column
glass[-c(10)] <- scale(glass[-c(10)])

# confirm that normalization worked
summary(glass$RI)

#data partition
set.seed(1234)
ind <- sample(2 , nrow(glass) , replace = TRUE , prob = c(0.8 , 0.2))
glass_train <- glass[ind == 1 , ]
glass_test <- glass[ind == 2 , ]

#---- Training a model on the data ----

# load the "class" library
install.packages("class")
library(class)

glass_test_pred <- knn(train = glass_train, test = glass_test,
                      cl = glass_train$Type, k = 21)

## ---- Evaluating model performance ---- ##
confusion_test <- table(x = glass_test$Type, y = glass_test_pred)
confusion_test

Accuracy <- sum(diag(confusion_test))/sum(confusion_test)
Accuracy 

# Training Accuracy to compare against test accuracy
glass_train_pred <- knn(train = glass_train, test = glass_train, cl = glass_train$Type, k=21)

confusion_train <- table(x = glass_train$Type, y =  glass_train_pred)
confusion_train

Accuracy_train <- sum(diag(confusion_train))/sum(confusion_train)
Accuracy_train

##plotting accuracy for different k values (1 <= k <= 39)
pred.train <- NULL
pred.val <- NULL
error_rate.train <- NULL
error_rate.val <- NULL
accu_rate.train <- NULL
accu_rate.val <- NULL
accu.diff <- NULL
error.diff <- NULL

for (i in 1:39) {
  pred.train <- knn(train = glass_train, test = glass_train, cl = glass_train$Type, k = i)
  pred.val <- knn(train = glass_train, test = glass_test, cl = glass_train$Type, k = i)
  error_rate.train[i] <- mean(pred.train!=glass_train$Type)
  error_rate.val[i] <- mean(pred.val != glass_test$Type)
  accu_rate.train[i] <- mean(pred.train == glass_train$Type)
  accu_rate.val[i] <- mean(pred.val == glass_test$Type)  
  accu.diff[i] = accu_rate.train[i] - accu_rate.val[i]
  error.diff[i] = error_rate.val[i] - error_rate.train[i]
}

knn.error <- as.data.frame(cbind(k = 1:39, error.train = error_rate.train, error.val = error_rate.val, error.diff = error.diff))
knn.accu <- as.data.frame(cbind(k = 1:39, accu.train = accu_rate.train, accu.val = accu_rate.val, accu.diff = accu.diff))

library(ggplot2)
errorPlot = ggplot() + 
  geom_line(data = knn.error[, -c(3,4)], aes(x = k, y = error.train), color = "blue") +
  geom_line(data = knn.error[, -c(2,4)], aes(x = k, y = error.val), color = "red") +
  geom_line(data = knn.error[, -c(2,3)], aes(x = k, y = error.diff), color = "black") +
  xlab('knn') +
  ylab('ErrorRate')
accuPlot = ggplot() + 
  geom_line(data = knn.accu[,-c(3,4)], aes(x = k, y = accu.train), color = "blue") +
  geom_line(data = knn.accu[,-c(2,4)], aes(x = k, y = accu.val), color = "red") +
  geom_line(data = knn.accu[,-c(2,3)], aes(x = k, y = accu.diff), color = "black") +
  xlab('knn') +
  ylab('AccuracyRate')

# Plot for Accuracy
plot(knn.accu[, c(4)], type = "b", xlab = "K-Value", ylab = "DifferenceInAccu") 

# Plot for Error
plot(knn.error[, c(4)], type = "b", xlab = "K-Value", ylab = "DifferenceInError") 

######################## problem2 #############################
#Problem Statement: -
#A National Park, in India is dealing with a problem of segregation of its species based on the different attributes it has so that they can have cluster of species together rather than  manually classify them, they have taken painstakingly  collected the data and would like you to help them out with a classification model for their  business objective to be achieved, by using KNN Algorithm  classify the different species and draft your 

library(readr)
zoo_data <- read_csv("C:\\Users\\DELL\\Downloads\\Zoo.csv")
#droping animal name
zoo <- zoo_data[-c(1)]

# table of Type
table(zoo$type)

# recode Type as a factor
zoo$type <- factor(zoo$type)
str(zoo)

# table or proportions with more informative labels
round(prop.table(table(zoo$type)) * 100, digits = 2)

#scaling the data except output column
zoo[-c(17)] <- scale(zoo[-c(17)])

# confirm that scaling worked
summary(zoo$milk)

#data partition
set.seed(1234)
ind <- sample(2 , nrow(zoo) , replace = TRUE , prob = c(0.8 , 0.2))
zoo_train <- zoo[ind == 1 , ]
zoo_test <- zoo[ind == 2 , ]

#---- Training a model on the data
# load the "class" library
install.packages("class")
library(class)

zoo_test_pred <- knn(train = zoo_train, test = zoo_test,cl = zoo_train$type,k = 5)

##Evaluating model performance
confusion_test <- table(x = zoo_test$type, y = zoo_test_pred)
confusion_test

Accuracy <- sum(diag(confusion_test))/sum(confusion_test)
Accuracy 

# Training Accuracy to compare against test accuracy
zoo_train_pred <- knn(train = zoo_train, test = zoo_train, cl = zoo_train$type, k=5)

confusion_train <- table(x = zoo_train$type, y = zoo_train_pred)
confusion_train

Accuracy_train <- sum(diag(confusion_train))/sum(confusion_train)
Accuracy_train

# Create the cross tabulation of predicted vs. actual
CrossTable(x = zoo_test$type, y = zoo_test_pred, prop.chisq=FALSE)

#plotting accuracy for different k values (1 <= k <= 39)
pred.train <- NULL
pred.val <- NULL
error_rate.train <- NULL
error_rate.val <- NULL
accu_rate.train <- NULL
accu_rate.val <- NULL
accu.diff <- NULL
error.diff <- NULL

for (i in 1:39) {
  pred.train <- knn(train = zoo_train, test = zoo_train, cl = zoo_train$type, k = i)
  pred.val <- knn(train = zoo_train, test = zoo_test, cl = zoo_train$type, k = i)
  error_rate.train[i] <- mean(pred.train!=zoo_train$type)
  error_rate.val[i] <- mean(pred.val != zoo_test$type)
  accu_rate.train[i] <- mean(pred.train == zoo_train$type)
  accu_rate.val[i] <- mean(pred.val == zoo_test$type)  
  accu.diff[i] = accu_rate.train[i] - accu_rate.val[i]
  error.diff[i] = error_rate.val[i] - error_rate.train[i]
}

knn.error <- as.data.frame(cbind(k = 1:39, error.train = error_rate.train, error.val = error_rate.val, error.diff = error.diff))
knn.accu <- as.data.frame(cbind(k = 1:39, accu.train = accu_rate.train, accu.val = accu_rate.val, accu.diff = accu.diff))

library(ggplot2)
errorPlot = ggplot() + 
  geom_line(data = knn.error[, -c(3,4)], aes(x = k, y = error.train), color = "blue") +
  geom_line(data = knn.error[, -c(2,4)], aes(x = k, y = error.val), color = "red") +
  geom_line(data = knn.error[, -c(2,3)], aes(x = k, y = error.diff), color = "black") +
  xlab('knn') +
  ylab('ErrorRate')
plot(errorPlot)

accuPlot = ggplot() + 
  geom_line(data = knn.accu[,-c(3,4)], aes(x = k, y = accu.train), color = "blue") +
  geom_line(data = knn.accu[,-c(2,4)], aes(x = k, y = accu.val), color = "red") +
  geom_line(data = knn.accu[,-c(2,3)], aes(x = k, y = accu.diff), color = "black") +
  xlab('knn') +
  ylab('AccuracyRate')
plot(accuPlot)

# Plot for Accuracy
plot(knn.accu[, c(4)], type = "b", xlab = "K-Value", ylab = "DifferenceInAccu") 

# Plot for Error
plot(knn.error[, c(4)], type = "b", xlab = "K-Value", ylab = "DifferenceInError") 

########################################END#########################################





