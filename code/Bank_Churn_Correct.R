library(rpart)
library(dplyr)
library(imbalance)
library(DMwR)
library(caret)
library(randomForest)
library(xgboost)
library(ROCR)
library(Ckmeans.1d.dp)
library(vip)
library(ggplot2)
library(pROC)

# read parameters
args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
  stop("USAGE: Rscript Bank_Churn_Correct.R --input data.csv --modeloutput results/model_performance.csv 
       --sampleoutput results/sample_performance.csv --fsoutput results/fs_performance.csv", call.=FALSE)
}

# parse parameters
i<-1 
while(i < length(args))
{
  if(args[i] == "--input"){
    input <- args[i+1]
    i <- i+1
  }else if(args[i] == "--modeloutput"){
    modeloutput <- args[i+1]
    i <- i+1
  }else if(args[i] == "--sampleoutput"){
    sampleoutput <- args[i+1]
    i <- i+1
  }else if(args[i] == "--fsoutput"){
    fsoutput <- args[i+1]
    i <- i+1
  }else{
    stop(paste("Unknown flag", args[i]), call.=FALSE)
  }
  i <- i+1
}
if(!'--input' %in% args){
  stop("error:missing --input", call. = FALSE)
}
if(!'--modeloutput' %in% args){
  stop("error:missing --modeloutput", call. = FALSE)
}
if(!'--sampleoutput' %in% args){
  stop("error:missing --sampleoutput", call. = FALSE)
}
if(!'--fsoutput' %in% args){
  stop("error:missing --fsoutput", call. = FALSE)
}
input <- 'BankChurners.csv'
data <- read.csv(input)
data <- data[,2:21]
data['Avg_Trans_Amt'] <- data$Total_Trans_Amt/data$Total_Trans_Ct

data$Customer_Age_G[data$Customer_Age > 25 & data$Customer_Age <= 35] <- 0
data$Customer_Age_G[data$Customer_Age > 35 & data$Customer_Age <= 45] <- 1
data$Customer_Age_G[data$Customer_Age > 45 & data$Customer_Age <= 55] <- 2
data$Customer_Age_G[data$Customer_Age > 55 & data$Customer_Age <= 65] <- 3
data$Customer_Age_G[data$Customer_Age > 65 ] <- 4
data$Customer_Age_G <- as.factor(data$Customer_Age_G)

data$Months_on_book_G[data$Months_on_book >= 0 & data$Months_on_book <= 10] <- 0
data$Months_on_book_G[data$Months_on_book > 10 & data$Months_on_book <= 20] <- 1
data$Months_on_book_G[data$Months_on_book > 20 & data$Months_on_book <= 30] <- 2
data$Months_on_book_G[data$Months_on_book > 30 & data$Months_on_book <= 40] <- 3
data$Months_on_book_G[data$Months_on_book > 40 ] <- 4
data$Months_on_book_G <- as.factor(data$Months_on_book_G)

data <- subset(data, select = -c(Credit_Limit, Total_Trans_Amt, Customer_Age, Months_on_book))

data$Attrition_Flag = as.factor(data$Attrition_Flag)

orig_label <- data$Attrition_Flag
dummy <- dummyVars(" Attrition_Flag ~ .", data=data)
dum_data <- data.frame(predict(dummy, newdata = data))
dum_data$Attrition_Flag <- orig_label
income_cat_index <- which(dum_data$Income_Category.Unknown==1)
dum_data[income_cat_index,"Income_Category..60K....80K"] <- 1
dum_data <- subset(dum_data, select = -c(Income_Category.Unknown))
edu_cat_index <- which(dum_data$Education_Level.Unknown==1)
dum_data[edu_cat_index,"Education_Level.High.School"] <- 1
dum_data <- subset(dum_data, select = -c(Education_Level.Unknown))

set.seed(3456)

MWMOTE_ <- function(data){
  add <- table(data$Attrition_Flag)[2] - table(data$Attrition_Flag)[1]
  newMWMOTE <- mwmote(data, numInstances = add, classAttr = 'Attrition_Flag')
  result <- rbind(data, newMWMOTE)
  return(result)
}

SMOTE_ <- function(data){
  result <- SMOTE(Attrition_Flag ~ ., data, perc.over = 200, perc.under = 150)
  return(result)
}

decision_tree <- function(train, valid, test){
  model <- rpart(Attrition_Flag~., data=train)
  train_pred <- predict(model, type='class')
  valid_pred <- predict(model, newdata=valid, type="class")
  train_matrix <- table(real=train$Attrition_Flag, predict=train_pred)
  train_pre <- sum(diag(train_matrix))/sum(train_matrix) 
  valid_matrix <- table(real=valid$Attrition_Flag, predict=valid_pred)
  valid_pre <- sum(diag(valid_matrix))/sum(valid_matrix)
  pred <- prediction(predict(model, valid)[,2], valid$Attrition_Flag)
  test_pred <- predict(model, newdata=test, type="class")
  test_matrix <- table(real=test$Attrition_Flag, predict=test_pred)
  test_pre <- sum(diag(test_matrix))/sum(test_matrix) 
  auc <- performance(pred, "auc")
  auc <- round(auc@y.values[[1]], 2)
  result <- data.frame(train=train_pre, valid=valid_pre, test=test_pre, auc=auc)
  return(list(result=result, model=model))
}

random_forest <- function(train, valid, test){
  model <- randomForest(factor(Attrition_Flag) ~ ., data = train, ntree=10)
  train_pred <- predict(model, type='class')
  valid_pred <- predict(model, newdata=valid, type="class")
  train_matrix <- table(real=train$Attrition_Flag, predict=train_pred)
  train_pre <- sum(diag(train_matrix))/sum(train_matrix) 
  valid_matrix <- table(real=valid$Attrition_Flag, predict=valid_pred)
  valid_pre <- sum(diag(valid_matrix))/sum(valid_matrix)
  test_pred <- predict(model, newdata=test, type="class")
  test_matrix <- table(real=test$Attrition_Flag, predict=test_pred)
  test_pre <- sum(diag(test_matrix))/sum(test_matrix) 
  pred <-  prediction(predict(model, valid, type='prob')[,2], valid$Attrition_Flag)
  auc <- performance(pred, "auc")
  auc <- round(auc@y.values[[1]], 2)
  result <- data.frame(train=train_pre, valid=valid_pre, test=test_pre, auc=auc)
  return(list(result=result, model=model))
}

XG_Boost <- function(train, valid, test){
  new_train <- Matrix::sparse.model.matrix(Attrition_Flag ~ .-1, data = train)
  new_valid <- Matrix::sparse.model.matrix(Attrition_Flag ~ .-1, data = valid)
  new_test <- Matrix::sparse.model.matrix(Attrition_Flag ~ .-1, data = test)
  train_label = train[,"Attrition_Flag"] == 'Attrited Customer' 
  valid_label = valid[,"Attrition_Flag"] == 'Attrited Customer'
  test_label = test[,"Attrition_Flag"] == 'Attrited Customer'
  dtrain <- xgb.DMatrix(data = new_train, label=train_label)
  dvalid <- xgb.DMatrix(data = new_valid, label=valid_label)
  dtest <- xgb.DMatrix(data = new_test, label=test_label)
  model <- xgb.train(data = dtrain, max.depth=6, eta=0.3, nthread = 2,
                     nround = 15, eval.metric = "error", objective = "binary:logistic")
  
  train_pred <- predict(model, new_train)
  valid_pred <- predict(model, new_valid)
  test_pred <- predict(model, new_test)
  
  train_matrix <- table(real=train$Attrition_Flag, predict=train_pred>0.5)
  train_pre <- (train_matrix[1,2]+train_matrix[2,1])/sum(train_matrix)
  valid_matrix <- table(real=valid$Attrition_Flag, predict=valid_pred>0.5)
  valid_pre <- (valid_matrix[1,2]+valid_matrix[2,1])/sum(valid_matrix)
  test_matrix <- table(real=test$Attrition_Flag, predict=test_pred>0.5)
  #print(test_matrix)
  test_pre <- (test_matrix[1,2]+test_matrix[2,1])/sum(test_matrix)
  #print(test_pre)
  recall <- test_matrix[1, 2]/(test_matrix[1, 1]+test_matrix[1, 2])
  pred <- prediction(valid_pred, valid$Attrition_Flag)
  roc_test <- roc(valid$Attrition_Flag, valid_pred, algorithm = 2)
  auc <- auc(roc_test)
  result <- data.frame(train=train_pre, valid=valid_pre, test=test_pre, auc=auc, recall=recall)
  return(list(result=result, model=model))
}

#cross-validate
cross_valid <- function(data, k, sample_func, model_func){
  data.index <- sample(x=1:nrow(data), size=ceiling(0.8*nrow(data)))
  test <- data[-data.index,]
  data <- data[data.index,]
  spec = rep(1/k, k)
  g = sample(cut(seq(nrow(data)), nrow(data)*cumsum(c(0,spec)), labels = 1:k))
  res = split(data, g)
  
  i <- 1
  output <- data.frame()
  max_valid_precision <- 0
  for(i in 1:k){
    result <- data.frame()
    valid <- res[[i]]
    if(is.null(sample_func)){
      train <- bind_rows(res[-c(i)])
    }else{
      train <- sample_func(bind_rows(res[-c(i)]))
    }
    result <- model_func(train, valid, test)$result
    model <- model_func(train, valid, test)$model
    if(result$valid>=max_valid_precision){
      max_valid_precision <- result$valid 
      best_model <- model
    }
    output <- rbind(output, result)
  }
  
  
  avg <- data.frame(fold=k, training=round(mean(output$train),4), 
                    validation=round(mean(output$valid),4),
                    test=round(mean(output$test),4),
                    auc=round(mean(output$auc),4),
                    recall=round(mean(output$recall),4))
  return(list(avg=avg, model=best_model))
}

model_result <- data.frame()
sample_list <- c(MWMOTE_, SMOTE_, 'None')
model_list <- c(decision_tree, random_forest, XG_Boost)
s_list <- c('MWMOTE', 'SMOTE', 'None')
m_list <- c('decision_tree', 'random_forest', 'XG_Boost')
i <- 0
for(model in model_list){
  i <- i+1
  avg <- cross_valid(dum_data, 5, NULL, model)$avg
  if('recall' %in% colnames(avg)){
    avg <- subset(avg, select = -recall)
  }
  info <- data.frame(model=m_list[i])
  avg <- cbind(avg, info)
  model_result <- rbind(model_result, avg)
}
#cross_valid(dum_data, 5, NULL, XG_Boost)
i <- 0
sample_result <- data.frame()
for(sampling in sample_list){
  i <- i+1
  if(i != length(s_list)){
    avg <- cross_valid(dum_data, 5, sampling, XG_Boost)$avg
    info <- data.frame(sample=s_list[i])
    avg <- cbind(avg, info)
    sample_result <- rbind(sample_result, avg)
  }else{
    avg <- cross_valid(dum_data, 5, NULL, XG_Boost)$avg
    info <- data.frame(sample=s_list[i])
    avg <- cbind(avg, info)
    sample_result <- rbind(sample_result, avg)
  }
}

# Feature Importance
model <- cross_valid(dum_data, 5, SMOTE_, XG_Boost)$model
xgb_impr <- xgb.importance(colnames(dum_data[,-42]),model)
xgb.ggplot.importance(importance_matrix = xgb_impr, xlab = "Relative importance",top_n = 19)
vip(model, num_features = 19, geom = "point", horizontal = FALSE, 
    aesthetics = list(color = "orange", shape = 17, size = 3.5)) + theme(text = element_text(size=7), axis.text.x = element_text(angle=45, hjust=1))

bst_model_result <- cross_valid(dum_data, 5, SMOTE_, XG_Boost)$avg
selVars <- xgb_impr[,Feature][1:8]
selVars <- c(selVars,"Attrition_Flag")
fs_bst_model_result <- cross_valid(dum_data[,selVars], 5, SMOTE_, XG_Boost)$avg
fs_result <- rbind(bst_model_result,fs_bst_model_result)
fs_label <- c("All","Feature Importance")
fs_result <- cbind(fs_result,fs_label)


# Null Model
set.seed(3456)
data.index <- sample(x=1:nrow(dum_data), size=ceiling(0.8*nrow(dum_data)))
test <- dum_data[-data.index,]
new_test <- Matrix::sparse.model.matrix(Attrition_Flag ~ .-1, data = test)
test_pred <- predict(model, new_test)

logLikelihood <- c()
pseudoR2 <- c()
logLikelihood <- c(logLikelihood, round(sum(ifelse(test$Attrition_Flag=="Attrited Customer",log(test_pred),log(1-test_pred))), digits = 4))
pNull <- sum(ifelse(test$Attrition_Flag=="Attrited Customer",1,0))/dim(test)[[1]]
deviance_model <- -2 * sum(ifelse(test$Attrition_Flag=="Attrited Customer",log(test_pred),log(1-test_pred))) 
deviance_null <- -2 * (sum(ifelse(test$Attrition_Flag=="Attrited Customer",1,0))*log(pNull) + sum(ifelse(test$Attrition_Flag=="Attrited Customer",0,1))*log(1-pNull))
pseudoR2 <- c(pseudoR2, round(1 - (deviance_model/deviance_null),digits = 4))  

# Excel Output
write.table(model_result, file=modeloutput, row.names = F, quote = F, sep = ',')
write.table(sample_result, file=sampleoutput, row.names = F, quote = F, sep = ',')
write.table(fs_result, file=fsoutput, row.names = F, quote = F, sep = ',')