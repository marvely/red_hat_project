######## a working script from kaggle ###########
library(data.table)
library(FeatureHashing)
library(Matrix)
library(xgboost)

############# basic data preparation via data.table ################
people <- fread("~/Documents/Kaggle Project/redhat/people.csv", showProgress = F)
# get the logical col names as well as change the cols to logi...
p_logi <- names(people)[which(sapply(people, is.logical))] 

# convert to 0 1 for logi cols in people
for (col in p_logi) set(people, j = col, value = as.integer(people[[col]]))

train <- fread("~/Documents/Kaggle Project/redhat/act_train.csv", showProgress = F)

d1 <- merge(train, people, by = "people_id", all.x = T)
Y <- d1$outcome
d1[, outcome := NULL]

################### process categorical features via FeatureHashing #################
b <- 2^22  # the performance of the model doesn't appear to improve much above a hash size of 2^22
# so it's optimized, how? just by looking at the model evaluation? :(
# essentially you want the hash size to be large enough that you aren't losing too much information through collisions. The prime suspect would be the group_1 feature - with 30,000 unique values - which might be masking some of the other features if the hash size is too small.
#  the choice of hash.size are also part of the tuning and the validation stage is still useful even if some of the process is manual.

f <- ~. - people_id - activity_id - date.x - date.y - 1

X_train <- hashed.model.matrix(f, d1, hash.size = b)

sum(colSums(X_train) > 0) 
# check how many cols of the sparseMatrix are occupied by at least one row of the training data
# this one line is the only reason to load the Matrix library :)

################# validate xgboost model #################
set.seed(75786)
unique_p <- unique(d1$people_id)
valid_p <- unique_p[sample(1: length(unique_p), 30000)]

valid <- which(d1$people_id %in% valid_p)
model <- (1:length(unique_p))[-valid]

# list of 4 parameters
param <- list(objective = "binary:logistic", eval_metric = "auc", booster = "gblinear", eta = 0.03)

# group dataset in a xgb.DMatrix, you can even add other meta data in it.
dmodel <- xgb.DMatrix(X_train[model, ], label = Y[model])
dvalid <- xgb.DMatrix(X_train[valid, ], label = Y[valid])
# advanced function to train model
ml <- xgb.train(data = dmodel, param, nrounds = 10, watchlist = list(model = dmodel, valid = dvalid), print_every_n = 10)

################### retrain on all data and predict for test set #####################
dtrain <- xgb.DMatrix(X_train, label = Y)
m2 <- xgb.train(data = dtrain, param, nrounds = 50, watchlist = list( train = dtrain), print.every.n = 10)

test <- fread("~/Documents/Kaggle Project/redhat/act_test.csv", showProgress = F)
d2 <- merge(test, people, by = "people_id", all.x = T)

X_test <- hashed.model.matrix(f, d2, hash.size = b)
dtest <- xgb.DMatrix(X_test)

out <- predict(m2, dtest)
sub <- data.frame(activity_id = d2$activity_id, outcome = out)
write.csv(sub, file = "sub.csv", row.names = F)
summary(sub$outcome)