##################### learn xgboost #######################
#                                                         #
#                                                         #
#                                                         #
#                                                         #
###########################################################
require(xgboost)
data("agaricus.train", package = "xgboost")
data("agaricus.test", package = "xgboost")

train <- agaricus.train
test <- agaricus.test

##### Discover the dimensionality of the datasets #########
#                                                         #
#                                                         #
#                                                         #
#                                                         #
###########################################################
dim(train$data) 
dimnames(train$data) #<------ the names of the attributes?
class(train$data)[1]
class(train$label) #<---------- the outcome in train set.


####################### training model ####################
#                                                         #
#                                                         #
#                                                         #
###########################################################
# each variable is a list containing two things, label and data. The next step is to feed this data to xgboost. Besides the data, we need to train the model with some other parameters:
# nrounds: the number of decision trees in the final model
# objective: the training objective to use, where "binary: logistic" means a binary classifier.
# max.depth: the depth of each tree.
#nthread: number of cpu threads we are going to use.
# verbose: 0, no message printed; 1, print evaluation metric; 2, also print information about tree.
# eta: step size shrinkage used in update to prevent overfitting. range [0,1]
# xgboost is a simple function to train model
model <- xgboost(data = train$data, label = train$label, nrounds = 30, objective = "binary:logistic", max.depth = 15, nthread = 2, eta = 1, verbose = 2, min_child_weight = 50)

preds <- predict(model, test$data)
print(length(preds))
print(head(preds)) #<------- these are probability to be classified as 1, are not final result.

# Need transformation, round up to 1.
predictions <- as.numeric(preds > 0.5)
head(predictions)
# in xgboost it provides a function xgb.cv to do cross validation
# just add a param: nfold
cv.res <- xgb.cv(data = train$data, label = train$label, nfold = 5, nrounds = 2, objective = "binary:logistic")
# early stopping
cv_res_early_stop <- xgb.cv(data = train$data, label = train$label, nfold = 5, nrounds = 20, early.stop.round = 3, objective = "binary:logistic", maximize = F)
# 3 means if the performance is not getting better for 3 steps, then the program will stop.
# maximize = F means our goal is not to maximize the evaluation, where the default evaluation metric for binary classification is the classification error rate.


################# measuring model performance #####################
err <- mean(predictions != test$label) # computes the vector of error between actual and computed probabilities;
# then computes the average error itself.
print(paste("test-error = ", err))

################## advanced features ######################
#                                                         #
#                                                         #
#                                                         #
#                                                         #
###########################################################
# contains the features, target and other side informations, like weights, missing values...
dense_train <- xgb.DMatrix(data = train$data, label = train$label)
dense_test <- xgb.DMatrix(data = test$data, label = test$label)

watchlist <- list(train = dense_train, test = dense_test) # contains both train and test sets!!!
# model:
advanced_bst <- xgb.train(data = dense_train, max.depth = 2, eta = 1, nround = 2, nthread = 2, watchlist = watchlist, eval.metric = "logloss",  eval.metric = "auc",  eval.metric = "error", objective = "binary:logistic")

pred_advanced_bst <- predict(advanced_bst, dense_train, outputmargin = T)
# the parameter outputmargin indicates that we don't need a logistic trasnformation of the result
# and we add this info to the original train set so the model knows where to start
setinfo(dense_train, "base_margin", pred_advanced_bst)
model2 <- xgboost(data = dense_train, nrounds = 2, objective = "binary:logistic")
#################### Linear Boosting ######################
#                                                         #
#                                                         #
#                                                         #
#                                                         #
###########################################################
# add booster = "gblinear", remove eta = 1.
# model:
linear_bst <- xgb.train(data = dense_train, booster = "gblinear", max.depth = 2,  nround = 2, nthread = 2, watchlist = watchlist, eval.metric = "logloss",  eval.metric = "auc",  eval.metric = "error", objective = "binary:logistic")
# 2nd model got 0, overfitting!
# but what should we do next if we find the overfitting and try to fix it?
linear_pred <- predict(linear_bst, test$data)
linear_pred <- as.numeric(linear_pred > 0.5)
linear_bst_err <- mean(linear_pred != test$label)
print(paste("test-error = ", linear_bst_err))

############## Manipulating xgb.DMatrix ###################
#                                                         #
#                                                         #
#                                                         #
#                                                         #
###########################################################
#
# information extraction
label <- getinfo(dense_test, "label")

# view feature importance/influence from the learnt model
importance_matrix <- xgb.importance(agaricus.train$data@Dimnames[[2]], model = model) #<------- simple model here
print(importance_matrix)
xgb.plot.importance(importance_matrix)

# Deepness
# There is more than one way to understand the structure of the trees, besides plotting them all. Since there are all binary trees, we can have a clear figure in mind if we get to know the depth of each leaf.
xgb.plot.deepness(model = model)
# cannot find the function...

# view the trees from a model
xgb.plot.tree(model = advanced_bst)

# save model to R's raw vector
rawVec <- xgb.save.raw(advanced_bst)

print(class(rawVec))

# load binary model to R
advanced_bst_raw <- xgb.load(rawVec)

pred_raw <- predict(advanced_bst_raw, test$data)
pred_raw <- as.numeric(pred_raw > 0.5)
pred_raw_err <- mean(pred_raw != test$label)
print(paste("test-error = ", pred_raw_err))
# result is identical to what we got from advanced_bst

################# Handle Missing Values ###################
#                                                         #
#                                                         #
#                                                         #
#                                                         #
###########################################################
# When using a feature with missing values to do splitting, xgboost will aswsign a direction to the missing values instead of a numerical value.
# Specifically, xgboost guides all the data points with missing values to the left and right respectively, then choose the direction with a higher gain with regard to the objective.

dat <- matrix(rnorm(128), 64, 2) 
label_dat <- sample(0:1, nrow(dat), replace = T) #<------ randomly assign an outcome?

for (i in 1:nrow(dat)) {
  ind <- sample(2, 1)
  dat[i, ind] <- NA
}

missing_data_model <- xgboost(data = dat, label = label_dat, missing = NA, nrounds = 2, objective = "binary:logistic")
# practically, the default value of missing is exactly NA, therefore we don't even need to specify it in a standard case.

################## Model Inspection #######################
#                                                         #
#                                                         #
#                                                         #
#                                                         #
###########################################################
xgb.plot.tree(model = advanced_bst, feature_names = agaricus.train$data@Dimnames[[2]])
# what if we have many trees?
# "ensembled" tree visualization
xgb.plot.multi.trees(model = model, feature_names = agaricus.train$data@Dimnames[[2]], features.keep = 3)
# cannot find the function????????
