##############################################################################
##############################################################################
# Simple example using the bart function from the dbarts package to run a 
# Bayesian Additive Regression Tree Model for binary prediction
# The example uses the Heart Disease Data Set from the UCI ML Repository
##############################################################################
##############################################################################


library(tidyverse)
library(dbarts)
library(bayesplot)
library(posterior)
library(pROC)
library(rpart)
library(rpart.plot)

# Load data from ucimlrepo
HD_Repo <- ucimlrepo::fetch_ucirepo("Heart Disease")
full_data <- HD_Repo$data$features %>% mutate(heart_disease = as.numeric(HD_Repo$data$targets$num >0 ))

# check for missing values
which(is.na(full_data), arr.ind = T)


# For sake of demonstration, we mean impute and use indicator function
full_data <- full_data  %>% 
            mutate(across(everything(), ~ as.integer(is.na(.)), .names = "{.col}_missing")) %>%
            mutate(across(where(is.numeric), ~ifelse(is.na(.), mean(.,na.rm=T), . ))) %>%
            select(where(~n_distinct(.)>1))


# Prepare data to be put into bart function by establishing test and training set
train_proportion <- 0.8

set.seed(1)
train_index <- 1:nrow(full_data) %in% 
  sample(1:nrow(full_data), train_proportion*nrow(full_data), replace = F) 

outcome <- "heart_disease"
train_features <- model.matrix(~., data = full_data %>% filter(train_index) %>% select(-all_of(outcome)))
train_outcome <- full_data %>% filter(train_index) %>% pull(all_of(outcome))

test_features <- model.matrix(~., data = full_data %>% filter(!train_index) %>% select(-all_of(outcome)))
test_outcome <- full_data %>% filter(!train_index) %>% pull(all_of(outcome))


# Run bart function from dbarts package
# We use the generic priors from Chipman (2010), and default 200 trees
tictoc::tic()
BART_model <- bart(x.train   = train_features,
                   y.train   = train_outcome,
                   nskip     = 25000,
                   ndpost    = 10000,
                   keepevery = 10,
                   nchain    = 3,
                   x.test    = test_features)
tictoc::toc()
# 60 sectonds

n_iter <- BART_model$call$ndpost/ BART_model$call$keepevery
n_chains <- BART_model$call$nchain
n_obs_train <- nrow(train_features)
n_obs_test <- nrow(test_features)


# Extract training data set predictions into an array, and apply pnorm to obtain probabilities
yhat_train_post <- array(BART_model$yhat.train, dim = c(n_iter,n_chains,n_obs_train)) %>% pnorm
dimnames(yhat_train_post) <- list(NULL, paste0("Chain", 1:n_chains), paste0("Yhat_Obs_", 1:n_obs_train))

# Examine R-hat and ESS on observation-level training posterior probabilities
train_diagnostics <- summarize_draws(yhat_train_post, c("rhat", "ess_bulk"))
summary(train_diagnostics)

# Visualize "worst" R-hat and ESS observation-level posterior probabilites
worst_rhats <- train_diagnostics %>% arrange(rhat) %>% slice_head(n=6)
worst_ess <- train_diagnostics %>% arrange(-ess_bulk) %>% slice_head(n=6)

mcmc_trace(yhat_train_post, pars = worst_rhats$variable)
mcmc_trace(yhat_train_post, pars = worst_rhats$variable)



### Use regression tree to help understand what variables or simple
### interactions(if any) can be used to explain the model's posterior mean predictions

yhat_train_means <- apply(yhat_train_post, 3, mean)

rpart_mod <- full_data %>% filter(train_index) %>%
  select(-all_of(outcome)) %>% 
  rpart(yhat_train_means ~., data=., model = TRUE)

rpart.plot(rpart_mod, 
           fallen.leaves=F,
           faclen      = 0,
           clip.facs   = TRUE,
           compress    = TRUE,
           ycompress   = TRUE,
           shadow.col  = "#9fa2a9",
           type        = 4,
           extra       = 1,
           under       = TRUE,
           box.palette = "#c8c9cc",
           varlen      = 0,
           main= "HD Training Probs Split by Features")

# high thal, high cp, high ca => HD


#### Evaluate Test Set Preditions

# Extract test predictions into array and take the mean across iterations
yhat_test_post <- array(BART_model$yhat.test, dim = c(n_iter,n_chains,n_obs_test)) %>% pnorm
dimnames(yhat_test_post) <- list(NULL, paste0("Chain", 1:n_chains), paste0("Yhat_Obs_", 1:n_obs_test))
yhat_test_means <- apply(yhat_test_post, 3, mean)


# Create df with probability and threshold based prediction for analysis
prediction_df <- data.frame(yhat_test_prob = yhat_test_means,
                            yhat_test_binary = factor(as.numeric(yhat_test_means>0.5), c(1,0)),
                            y_test = factor(test_outcome,c(1,0)))


# Use caret to produce confusion matrix and relevant perfomance metrics, pROC to obtain AUC
Metrics <- prediction_df %>% with(caret::confusionMatrix(data = yhat_test_binary, reference = y_test))

# Confustion Matrix
Metrics$table

# Perfrmance Metrics
Metrics%>%
  .[["byClass"]] %>%
  .[c("Sensitivity","Specificity", "Pos Pred Value", "Neg Pred Value")] %>% round(3)

# AUC
prediction_df %>% with(roc(y_test, yhat_test_prob, levels = c(0,1)))




