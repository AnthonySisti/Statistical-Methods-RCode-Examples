##############################################################################
##############################################################################
# Simple example using the bcf function from the bcf package maintained by 
# Mathematica for conducting Bayesian causal inference
# The example uses the Data from the Current Population Survey on participation in the National Supported Work Demonstration (NSW) job-training program experiment.
# Disclaimer: The focus of this demo is not causal inference design, but rather just an implementation of BCF and related functions
# Do not interpret results as rigorously evaluated causal effects
##############################################################################
##############################################################################

# devtools::install_github("https://github.com/mathematica-mpr/bcf-1/tree/feature/random-effects",force = TRUE)
library(tidyverse)
library(dbarts)
library(bcf)
library(bayesplot)
library(posterior)
library(rpart)
library(rpart.plot)
library(MatchIt)
library(cobalt)

# Read in nsw_mixtape data set from causaldata
data(nsw_mixtape, package = "causaldata")
glimpse(nsw_mixtape)

# join csw_mixtape data set from causaldata
data(cps_mixtape, package = "causaldata")
glimpse(cps_mixtape)

nsw_cps_mixtape <- rbind(nsw_mixtape%>% select(-data_id),
                         cps_mixtape %>% select(-data_id))

# For the sake of simplicity in this demo we model the difference in income from
# 1975 as a normal random variable
# This is flawed, in rigrorus a causal analysis we would model the income 
# in 1975 using a hurdle/zero-inflated model

full_data <- nsw_cps_mixtape %>%
  mutate(treat = factor(treat, c(0,1)),
         change_wage= re78-re75,
         re75_0 = as.numeric(re75==0)) %>%
  select(-re78)

outcome = "change_wage"

# Summarize some relevant variables

full_data %>%
  group_by(treat) %>%
  summarise(n=n(),
            across(everything(),mean))


ps_train_features <- full_data %>% select(-treat,-all_of(outcome)) %>% data.matrix()
ps_train_outcome <- full_data %>% pull(treat)

# Fit propensity score model using bart from dbarts package
# We use the generic priors from Chipman (2010), and default 200 trees
tictoc::tic()
BART_model <- bart(x.train   = ps_train_features,
                   y.train   = ps_train_outcome,
                   nskip     = 2500,
                   ndpost    = 1000,
                   keepevery = 10,
                   nchain    = 3)
tictoc::toc()

n_iter <- BART_model$call$ndpost/ BART_model$call$keepevery
n_chains <- BART_model$call$nchain
n_obs_train <- nrow(ps_train_features)

# Extract training data set predictions into an array, and apply pnorm to obtain probabilities
yhat_train_post <- array(BART_model$yhat.train, dim = c(n_iter,n_chains,n_obs_train)) %>% pnorm
dimnames(yhat_train_post) <- list(NULL, paste0("Chain", 1:n_chains), paste0("Yhat_Obs_", 1:n_obs_train))

# Examine R-hat and ESS on observation-level training posterior probabilities
train_diagnostics <- summarize_draws(yhat_train_post, c("rhat", "ess_bulk"))
summary(train_diagnostics)


# Add propensity score to data set
full_data$ps <- colMeans(pnorm(BART_model$yhat.train))

ggplot(full_data, aes(x=ps, group = treat, fill = treat)) +
  geom_density(alpha=0.5, bw=0.02) + theme_bw()




# We estimate the ATT  to understand how people who are likely to enroll
# might benefit from this program
# Use one-to-one nearest neighbor matching without replacement
# and discard  controls
# because of our flawed outcome variable, we exact match on re_75 being 0
m.out <- matchit(treat ~. - change_wage,
                  data = full_data,
                  method = "nearest",
                  ratio = 1,
                  distance = full_data$ps,
                  exact = ~re75_0)

# Extract matched dataset
matched_data <- match.data(m.out)

# Estimate ATT (difference in income_increase) 
matched_data %>%
  group_by(treat) %>%
  summarise(mean_outcome = mean(change_wage), .groups = "drop") %>%
  mutate(ATO = mean_outcome[treat==1] - mean_outcome[treat==0])
  

# Summarize before and after match
summary(m.out)

# Love plot (visualizing covariate balance) 
love.plot(m.out, binary = "std")

# view new PS distributions by treatment
ggplot(matched_data, aes(x=ps, group = treat, fill = treat)) +
  geom_density(alpha=0.5, bw=0.01) + theme_bw()


train_features <- matched_data %>%  
                    select(-treat,-all_of(outcome), -ps,
                           -weights, -distance, -subclass) %>% 
                    data.matrix()
train_outcome <-  matched_data %>%  pull(all_of(outcome))
train_ps <- matched_data %>% pull(ps)
train_treat <- matched_data %>% pull(treat)

tictoc::tic()
fit_bcf <- bcf(y           = train_outcome,
               z           = train_treat,
               x_control   = train_features,
               x_moderate  = train_features,
               pihat       = train_ps, 
               nburn       = 2500, 
               nsim        = 1000,
               n_chains    = 3,
               n_cores     = 3,
               n_threads   = 1,
               nthin       = 5,
               verbose     = 0,
               random_seed = 1,
               include_random_effects = FALSE,
               log_file   = tempfile(fileext='.log'),
               simplified_return = FALSE,
)
tictoc::toc()

n_iter <- 1000
n_chains <- 3
n_obs_train <- nrow(train_features)

# Extract taus into an matrix and array
tau_matrix <-fit_bcf$tau 
tau_train_post <- array(tau_matrix, dim = c(n_iter,n_chains,n_obs_train))
dimnames(tau_train_post) <- list(NULL, paste0("Chain", 1:n_chains), paste0("tau_Obs_", 1:n_obs_train))

# Examine R-hat and ESS on observation-level training posterior taus
train_diagnostics <- summarize_draws(tau_train_post, c("rhat", "ess_bulk"))
summary(train_diagnostics)


# Visualize "worst" R-hat and ESS observation-level posterior probabilites
worst_rhats <- train_diagnostics %>% arrange(rhat) %>% slice_head(n=6)
worst_ess <- train_diagnostics %>% arrange(-ess_bulk) %>% slice_head(n=6)

mcmc_trace(tau_train_post, pars = worst_rhats$variable)
mcmc_trace(tau_train_post, pars = worst_rhats$variable)



# Posterior distribution of ATE, due to small sample size 
# there is a lot of uncertainty
ATO_df <- data.frame(ATO = rowMeans(tau_matrix))

summarise(ATO_df,
          ATO_post_mean = mean(ATO),
          ATO_L95_CrI = quantile(ATO,c(0.025)),
          ATO_U95_CrI = quantile(ATO,c(0.975)),
          ATO_Median = median(ATO))

# If we had more data, we may be able to recover more precise estimates
# of CATE'S for the covariate profiles among the matched population, however
# we cannot be very certain with this smaller data set
taus_df <-data.frame(tau = colMeans(tau_matrix),
                                    L_95_tau = apply(tau_matrix  ,2,quantile,0.025),
                                    U_95_tau = apply(tau_matrix  ,2,quantile,0.975),
                                    Index = rank(colMeans(tau_matrix)))


ggplot(taus_df, aes(Index, tau)) + geom_linerange(aes(ymin = L_95_tau, ymax=U_95_tau)) +
  geom_line(col="red",linewidth=1.5) + ggtitle("CATE estimates among matched covariate profiles via BCF") +
  ylab("Estimated Effect (With 95% CrI Bands)") + xlab("Point estimate Rank (small to large)") + theme_bw() 

### Use regression tree to help understand what variables or simple
### interactions(if any) can be used to explain the model's posterior tau predictions

tau_train_means <- colMeans(tau_matrix)

rpart_mod <- matched_data  %>%
  select(-treat,-all_of(outcome), -ps,
         -weights, -distance, -subclass) %>% 
  rpart(tau_train_means ~., data=., model = TRUE, cp=0.001,
        maxdepth=3)

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
           main= "Tau Posterior Mean Split by Features")


# When considering just the point estimates, the program appears to have a largest 
# positive effect for non-black participants, but also participants
# with less income in 1975


  
