#~~~~~~~~~~~~~~~~~~~~~~~~~~
# Name: Sadurshan Ravindran
# UoW ID: w1833588
# IIT ID: 20200596
#~~~~~~~~~~~~~~~~~~~~~~~~~~

# importing the necessary libraries 
library(readxl)
library(neuralnet)
library(forecast)
library(caret)
library(MLmetrics)
library(nnfor)
library(useful)
library(ie2misc)

df = read_excel("UoW_load.xlsx") # reading the data set
#View(df)

colnames(df) <- c("Dates", "Nine", "Ten", "Eleven") # renaming the column names 

# NORMALIZE THE DATA
normalize = function(x) {
  return (as.numeric(x - min(x)) / as.numeric(max(x) - min(x)))
}

df_normalised = df # a clone copy of the dataset 
df_normalised$Dates <- as.numeric(df_normalised$Dates)  # normalizing the dates 
df_normalised <- as.data.frame(lapply(df_normalised, normalize)) 

#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------
# AR Approach 

#determining the best lag 
Lag = df$Eleven
pacf(Lag, lag = 15)

# Splitting the data into training and testing 
trainingDataSet <- df_normalised[1:430, ]    # training set
testingDataSet  <- df_normalised[431:493, ]  # testing set

for (x in 1:10) {
  # the neural network model
  model <- neuralnet(Eleven ~ Dates+Eleven,
                     hidden=c(5,x),
                     #hidden=c(5,x),
                     data = trainingDataSet,
                     act.fct = "logistic",
                     linear.output = TRUE,
                     err.fct = "sse",
                     learningrate = 0.1)
  
  predicted = predict(model, testingDataSet)  # predicted output from the model
  actual    = data.frame(testingDataSet)      # actual output

  mae  = round(mae (actual$Eleven, predicted) * 100, digits = 4)
  rmse = round(rmse(actual$Eleven, predicted) * 100, digits = 4)
  mape = round(MAPE(actual$Eleven, predicted) * 100, digits = 4)
  
  print("****************************")
  print(paste("Nodes: ",x))
  print(paste("MAE  : " , mae , " %" , sep = ""))
  print(paste("RMSE : " , rmse, " %" , sep = ""))
  print(paste("MAPE : " , mape, " %" , sep = ""))
  print("****************************")
}

learning_rate = 0
while (learning_rate <= 0.1) {
  model <- neuralnet(Eleven ~ Dates+Eleven,
                     hidden=c(5,2),
                     # hidden=c(6,6),
                     data = trainingDataSet,
                     act.fct = "logistic",
                     linear.output = TRUE,
                     err.fct = "sse",
                     learningrate = learning_rate)
  
  predicted = predict(model, testingDataSet)
  actual    = data.frame(testingDataSet)
  
  mae  = round(mae (actual$Eleven, predicted) * 100, digits = 4)
  rmse = round(rmse(actual$Eleven, predicted) * 100, digits = 4)
  mape = round(MAPE(actual$Eleven, predicted) * 100, digits = 4)
  
  print("****************************")
  print(paste("Learning Rate:", learning_rate))
  print(paste("MAE  : " , mae , " %" , sep = ""))
  print(paste("RMSE : " , rmse, " %" , sep = ""))
  print(paste("MAPE : " , mape, " %" , sep = ""))
  print("****************************")
  
  learning_rate = learning_rate + 0.02
}

model <- neuralnet(Eleven ~ Dates+Eleven,
                   hidden=c(5,2),
                   data = trainingDataSet,
                   act.fct = "logistic",
                   linear.output = TRUE,
                   err.fct = "sse",
                   learningrate = 0.1)

predicted = predict(model, testingDataSet)  # predicted output from the model
actual    = data.frame(testingDataSet)      # actual output

plot(model)

testingDataSet$Number = seq.int(nrow(testingDataSet))
plot(testingDataSet$Number,testingDataSet$Eleven, main = "Actual VS Predicted", xlab = "Number",ylab = "Rate", col = "black", type = "l")
lines(testingDataSet$Number, predicted, col="red") 
legend("bottomright",legend = c("Actual", "Predicted"),col = 1:2, lty = 1,cex = 0.50)

#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------
# NARX Approach 

for (x in 1:1) {
  model <- neuralnet(Eleven ~ Dates+Eleven+Ten+Nine,
                     hidden=c(2,3),
                     data = trainingDataSet,
                     act.fct = "logistic",
                     linear.output = TRUE,
                     err.fct = "sse",
                     learningrate = 0.02)
  
  predicted = predict(model, testingDataSet)
  actual    = data.frame(testingDataSet)
  
  mae  = round(mae (actual$Eleven, predicted) * 100, digits = 4)
  rmse = round(rmse(actual$Eleven, predicted) * 100, digits = 4)
  mape = round(MAPE(actual$Eleven, predicted) * 100, digits = 4)
  
  print("****************************")
  print(paste("Hidden Layer c=(2,3)"))
  print(paste("MAE  : " , mae , " %" , sep = ""))
  print(paste("RMSE : " , rmse, " %" , sep = ""))
  print(paste("MAPE : " , mape, " %" , sep = ""))
  print("****************************")
}

learning_rate = 0
while (learning_rate <= 0.1) {
  model <- neuralnet(Eleven ~ Dates+Eleven+Ten+Nine,
                     #hidden=c(2),
                     hidden=c(2,3),
                     data = trainingDataSet,
                     act.fct = "logistic",
                     linear.output = TRUE,
                     err.fct = "sse",
                     learningrate = learning_rate)
  
  predicted = predict(model, testingDataSet)
  actual    = data.frame(testingDataSet)
  
  mae  = round(mae (actual$Eleven, predicted) * 100, digits = 4)
  rmse = round(rmse(actual$Eleven, predicted) * 100, digits = 4)
  mape = round(MAPE(actual$Eleven, predicted) * 100, digits = 4)
  
  print("****************************")
  print(paste("Learning Rate:", learning_rate))
  print(paste("MAE  : " , mae , " %" , sep = ""))
  print(paste("RMSE : " , rmse, " %" , sep = ""))
  print(paste("MAPE : " , mape, " %" , sep = ""))
  print("****************************")
  
  learning_rate = learning_rate + 0.02
}

model <- neuralnet(Eleven ~ Dates+Eleven+Ten+Nine,
                   hidden=c(2,3),
                   data = trainingDataSet,
                   act.fct = "logistic",
                   linear.output = TRUE,
                   err.fct = "sse",
                   learningrate = 0.02)

predicted = predict(model, testingDataSet)  # predicted output from the model
actual    = data.frame(testingDataSet)      # actual output

plot(model)

testingDataSet$Number = seq.int(nrow(testingDataSet))
plot(testingDataSet$Number,testingDataSet$Eleven, main = "Actual VS Predicted", xlab = "Number",ylab = "Rate", col = "black", type = "l")
lines(testingDataSet$Number, predicted, col="red") 
legend("bottomright",legend = c("Actual", "Predicted"),col = 1:2, lty = 1,cex = 0.50)







