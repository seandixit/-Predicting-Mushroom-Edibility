
#B365 Final Project

library(ggplot2) # plotting and heatmap
library(rpart) # decision tree
library(rpart.plot) # to plot rpart decision tree
library(dplyr)    # this loads %>%
library(purrr)    # to use map function
library(caret) # confusion matrix
library(reshape2) # displaying heat map of confusion matrix
library(ROCR) # plotting roc curves

# *Fetching Data (UC Irvine ML Repository Mushroom Data Set converted into csv)
data = read.csv2("C:/Users/Postw/OneDrive/Documents/R Files/final_project/agaricus-lepiota.csv",stringsAsFactors=TRUE,sep=",");

# *Data Labeling
colnames(data) <- c("class_label", "cap_shape", "cap_surface", 
                    "cap_color", "bruises", "odor", 
                    "gill_attachment", "gill_spacing", "gill_size", 
                    "gill_color", "stalk_shape", "stalk_root", 
                    "stalk_surface_above_ring", "stalk_surface_below_ring", "stalk_color_above_ring", 
                    "stalk_color_below_ring", "veil_type", "veil_color", 
                    "ring_number", "ring_type", "spore_print_color", 
                    "population", "habitat")

levels(data$class_label) <- c("edible", "poisonous")
levels(data$cap_shape) <- c("bell", "conical", "flat", "knobbed", "sunken", "convex")
levels(data$cap_color) <- c("buff", "cinnamon", "red", "gray", "brown", "pink", 
                            "green", "purple", "white", "yellow")
levels(data$cap_surface) <- c("fibrous", "grooves", "scaly", "smooth")
levels(data$bruises) <- c("no", "yes")
levels(data$odor) <- c("almond", "creosote", "foul", "anise", "musty", "none", "pungent", "spicy", "fishy")
levels(data$gill_attachment) <- c("attached", "free")
levels(data$gill_spacing) <- c("close", "crowded")
levels(data$gill_size) <- c("broad", "narrow")
levels(data$gill_color) <- c("buff", "red", "gray", "chocolate", "black", "brown", "orange", 
                             "pink", "green", "purple", "white", "yellow")
levels(data$stalk_shape) <- c("enlarging", "tapering")
levels(data$stalk_root) <- c(NA, "bulbous", "club", "equal", "rooted")
levels(data$stalk_surface_above_ring) <- c("fibrous", "silky", "smooth", "scaly")
levels(data$stalk_surface_below_ring) <- c("fibrous", "silky", "smooth", "scaly")
levels(data$stalk_color_above_ring) <- c("buff", "cinnamon", "red", "gray", "brown", "pink", 
                                         "green", "purple", "white", "yellow")
levels(data$stalk_color_below_ring) <- c("buff", "cinnamon", "red", "gray", "brown", "pink", 
                                         "green", "purple", "white", "yellow")
levels(data$veil_type) <- "partial"
levels(data$veil_color) <- c("brown", "orange", "white", "yellow")
levels(data$ring_number) <- c("none", "one", "two")
levels(data$ring_type) <- c("evanescent", "flaring", "large", "none", "pendant")
levels(data$spore_print_color) <- c("buff", "chocolate", "black", "brown", "orange", 
                                    "green", "purple", "white", "yellow")
levels(data$population) <- c("abundant", "clustered", "numerous", "scattered", "several", "solitary")
levels(data$habitat) <- c("wood", "grasses", "leaves", "meadows", "paths", "urban", "waste")

summary(data) # after pre-processing

# *Data Cleaning (removes 4 features in total)
# removing useless column 
# veil_type has only one attribute value (removes 1 feature)
data = data[,-17] 
# finding missing values
print(paste0("Number of missing values in data: ", sum(is.na(data)))) # outputs 2480 missing values

# we consider two options: 
# ignoring the column with the missing values (removes 1 feature)
print(paste0("Number of missing values in stalk_root column: " , sum(is.na(data$stalk_root)))) # outputs 2480 missing values
print(paste0("Are all missing values in stalk_root?: ", sum(is.na(data$stalk_root)) == sum(is.na(data)))) # all missing values belong to stalk_root

data_without_stalk_root = data[,-12]
print(paste0("Number of columns in data_without_stalk_root: ", length(data_without_stalk_root)))

print(paste0("Number of missing values in data_without_stalk_root: ", sum(is.na(data_without_stalk_root)))) 

# ignoring the rows with the missing values
data_with_stalk_root = na.omit(data)
print(paste0("Number of rows omitted from data for data_with_stalk_root: ", nrow(data) - nrow(data_with_stalk_root)))

print(paste0("Number of missing values in data_with_stalk_root: ", sum(is.na(data_with_stalk_root)))) 

# (we set data here as either data_without_stalk_root or data_with_stalk_root
data = data_without_stalk_root

# *Pearson Correlation
numerical_data = sapply(data, unclass)

correlation = cor(numerical_data)
print(correlation) # correlation matrix
# if block is too light or dark, it is good correlation
# correlation heatmap
ggplot(data = melt(correlation), aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile(aes(fill = value)) + 
  geom_text(aes(label = round(value, 1))) +
  theme(axis.text.x=element_text(angle=90,hjust=1)) +
  ggtitle("Correlation Heatmap")

print(sort(abs(correlation[,1]), decreasing = TRUE))
# fitting labels
par(las=2) # make label text perpendicular to axis
par(mar=c(4,11,4,2)) # increase y-axis margin.
# plot bar plot of correlation with features
barplot(sort(abs(correlation[,1]), decreasing = TRUE)[-1], horiz=TRUE, xlim=c(0,.60))

# eliminating multicollinear features: increases model accuracy 
# from the correlation heatmap, the following variables are correlated with one another
# (denoted by unusually light spots in the heatmap)
# ring_type and bruises
# veil_color and gill_attachment
# we remove the ones with lower correlation with class label:
# gill_color and gill_attachement
data = data[,!names(data) %in% c("gill_attachment", "ring_type")]

# selecting top 10 features for rpart (only uses 10 features)
print(ncol(data))
numerical_data = sapply(data, unclass)
correlation = cor(numerical_data)
best_features = head(sort(abs(correlation[,1]), decreasing = TRUE)[-1], 10)
print(best_features)
data = data[,names(data) %in% c("class_label", "gill_size", "gill_color", "bruises", "gill_spacing", "stalk_surface_above_ring", "stalk_surface_below_ring", "population", "habitat", "ring_number", "cap_surface")]

# fitting labels
par(las=2) # make label text perpendicular to axis
par(mar=c(4,11,4,2)) # increase y-axis margin.
# plot bar plot of correlation with features
barplot(best_features, horiz=TRUE, xlim=c(0,.60))

# *Partition Data 
set.seed(101)
train_split_ratio = 0.6 # holdout
train_ind = sample(seq_len(nrow(data)), size=floor(train_split_ratio * nrow(data)))
train = data[train_ind,]
test = data[-train_ind,]
print(paste0("Size of training set: " , nrow(train)))
print(paste0("Size of testing set: " , nrow(test)))

# checking if partitions are balanced
print("Data Table: ")
print(round(prop.table(table(data$class_label)), 2)) # dataset is balanced
print("Train Data Table: ")
print(round(prop.table(table(train$class_label)), 2)) #  balanced
print("Test Data Table: ")
print(round(prop.table(table(test$class_label)), 2)) #  balanced

# *Decision Tree

# training (using cross-validation for best alpha)

# penalty matrix: 
# penalty 8 times bigger for Type I errors
penalty.matrix <- matrix(c(0,1,8,0), byrow=TRUE, nrow=2)
fit = rpart(class_label ~ ., data = train, method = "class", minsplit=0, cp = 0, parms = list(loss = penalty.matrix)) # cp = 0: dont prune tree
printcp(fit) # cross-validation to find alpha
plotcp(fit, main="Relationship between x-val relative error and size of tree")
rpart.plot(fit, main="Tree before pruning", box.palette = "GnBu", branch.lty = 3, shadow.col = "gray", nn = TRUE)

# pruning with alpha = cp that gives lowest xerror
bestCP = fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]
print(bestCP)
tree = prune(fit, cp=bestCP)
pruned_fit = prune(fit, cp = bestCP)
rpart.plot(pruned_fit, main="Tree after pruning", box.palette = "GnBu", branch.lty = 3, shadow.col = "gray", nn = TRUE)

# training error
ctrain = train[,1]
predict_tree = predict(pruned_fit, newdata = train, type="vector")
predict_tree[predict_tree==1] = "edible"
predict_tree[predict_tree==2] = "poisonous"
trainerrors = sum(predict_tree != ctrain)
print(paste0("num errors on training: ", trainerrors)) 

# testing error
ctest = test[,1] # "c" : class, 1 = edible, 2 = poisonous
predict_tree = predict(pruned_fit, newdata = test, type="vector")
ctest = as.numeric(ctest)
testerrors = sum(predict_tree != ctest)
print(paste0("num errors on testing: ", testerrors)) 

# create confusion matrix 
print(confusionMatrix(as.factor(ctest), as.factor(predict_tree)))
cm = confusionMatrix(as.factor(ctest), as.factor(predict_tree))
# 95% CI : (confidence interval)
# Accuracy : 0.9991

# plot roc curve
predict_tree = prediction(predict_tree, ctest)
roc_DT = performance(predict_tree,"tpr","fpr")
plot(roc_DT, lwd = 2)
abline(a = 0, b = 1) 

# *Multiple Logistic Regression 
# map categorical values to numeric values
data_numeric = sapply(data, unclass)
train_numeric = sapply(train, unclass)
test_numeric = sapply(test, unclass)

scaledTrain = as.data.frame(train_numeric)

# applying as.factor to each column of the scaled training set
scaledTrain <- scaledTrain %>% map_df(function(.x) as.factor(.x))

# glm model
logistic_model = glm(class_label ~ -1 + ., data = as.data.frame(scaledTrain), family = "binomial")
print(summary(logistic_model))
print(logistic_model)

# prediction
test_numeric = sapply(test, unclass)
scaledTest = as.data.frame(test_numeric)
#scaledTest = as.data.frame(scale(test_numeric))
scaledTest <- scaledTest %>% map_df(function(.x) as.factor(.x))
logistic_pred = predict(logistic_model, scaledTest, type="response")

# testing error
ctest = scaledTest[,1] # "c" : class
logistic_pred[logistic_pred>=0.5] = 2 # poisonous 
logistic_pred[logistic_pred<0.5] = 1 # edible
testerrors = sum(logistic_pred != ctest)
print(paste0("num errors on testing: ", testerrors)) 
print(paste0("accuracy of logistic regression: ", mean(logistic_pred == ctest)))

# logistic regression performance 
confusion_matrix = table(as.matrix(ctest), as.matrix(logistic_pred))
print(confusion_matrix)
err_metric=function(CM)
{
  TN =CM[1,1]
  TP =CM[2,2]
  FP =CM[1,2]
  FN =CM[2,1]
  precision =(TP)/(TP+FP)
  recall_score =(FP)/(FP+TN)
  f1_score=2*((precision*recall_score)/(precision+recall_score))
  accuracy_model  =(TP+TN)/(TP+TN+FP+FN)
  False_positive_rate =(FP)/(FP+TN)
  False_negative_rate =(TN)/(TN+FP)
  print(paste("Precision: ",round(precision,2)))
  print(paste("Accuracy: ",round(accuracy_model,2)))
  print(paste("Recall: ",round(recall_score,2)))
  print(paste("False Positive rate of the model: ",round(False_positive_rate,2)))
  print(paste("Specificity: ",round(False_negative_rate,2)))
  print(paste("f1 score of the model: ",round(f1_score,2)))
}
err_metric(confusion_matrix)
#err_metric(as.table(matrix(c(1:4), data=c(1543, 3, 0, 1704), ncol=2, nrow=2, dimnames=list(c("edible","poisonous"),c("edible","poisonous")))))

# roc curve
logistic_pred = prediction(logistic_pred, ctest)
roc = performance(logistic_pred,"tpr","fpr")
plot(roc, lwd = 2)
abline(a = 0, b = 1) 


# display confusion matrices for comparison (in same order of axes)

# logistic regression confusion matrix
print(as.table(matrix(c(1:4), data=c(1458, 88, 66, 1638), ncol=2, nrow=2, dimnames=list(c("edible","poisonous"),c("edible","poisonous")))))

# decision tree w/ penalty matrix 
print(as.table(matrix(c(1:4), data=c(1546, 0, 9, 1695), ncol=2, nrow=2, dimnames=list(c("edible","poisonous"),c("edible","poisonous")))))


