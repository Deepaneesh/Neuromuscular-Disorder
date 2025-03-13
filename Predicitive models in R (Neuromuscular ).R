library(dplyr)
library(tidyverse)
library(ggplot2)
library(lsr)
library(readr)
library(FactoMineR)   # For PCA and MCA
library(factoextra)   # For visualization
library(caret)        # For train-test split
library(glmnet)       # For Lasso/Ridge/ElasticNet
library(xgboost)      # For XGBoost
library(randomForest) # For Random Forest
library(e1071)        # For SVM
library(rsample)      # For Train - Test Split
library(caret)        # For confusion Matrix
library(lightgbm)     # For LightGBM
library(class)
library(nnet)
library(ROCR)
library(ranger)
library(catboost)
library(forcats)
library(psych)
library(mvnormtest)
library(rpart)
library(rpart.plot)
## dataset 
neuromuscular_synthetic_data=read.csv(file.choose(),header = T)
data=neuromuscular_synthetic_data
summary(data)


# converting the same Variable to a factor
data$Gender <- as.factor(data$Gender)
data$Disorder <- as.factor(data$Disorder)
data$Treatment <- as.factor(data$Treatment)
data$Adherence_to_Treatment <- as.factor(data$Adherence_to_Treatment)
data$Dosage <- as.factor(data$Dosage)
data$Early_Stage <- as.factor(data$Early_Stage)
data$Cured <- as.factor(data$Cured)
data$Died_with_Disease <- as.factor(data$Died_with_Disease)
data$Died_while_Treatment <- as.factor(data$Died_while_Treatment)
data$Muscle_Strength_Improvement <- as.factor(data$Muscle_Strength_Improvement)
data$Functional_Mobility_Improvement <- as.factor(data$Functional_Mobility_Improvement)
data$Respiratory_Function_Improvement <- as.factor(data$Respiratory_Function_Improvement)
data$Quality_of_Life_Improvement <- as.factor(data$Quality_of_Life_Improvement)
data$Major_Issue_Type <- as.factor(data$Major_Issue_Type)

# new variable declared 
data$Severity_score_factor=cut(data$Severity_Score, breaks=c(-1,3,6,10),right = T) %>% as.factor()
levels(data$Severity_score_factor)=c("Low","Medium","High")
data=data %>% select(-c("Patient_ID"))
summary(data)
View(data)


# Third objective #############################################################################################################################################################################################################################################################
# Selecting the  variables related to the patient's condition
data_patient_condition=data %>% select(c("Age", "Gender", "BMI", "Duration_Years", "Severity_Score",
                                         "Disease_Activity_Score", "Genetic_Mutation_Score", "Rate_of_Progression",
                                         "Muscle_Strength", "NCV", "Respiratory_Volume", "CK_Levels",
                                         "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic",
                                         "Adherence_to_Treatment", "Dosage", "Number_of_Treatments", "Early_Stage"))
data_patient_condition %>% summary()
View(data_patient_condition)
ncol(data_patient_condition)

# spliting the 80% of data in training set and 20 % to testing set based on the Early stage 
set.seed(123)
library(rsample)
data_split1 = initial_split(data_patient_condition,prop=0.7,strata=Early_Stage)
train_data1 = training(data_split1)
test_data1 = testing(data_split1)

# Building model for predicting th Early Stage by using Several algorithms (Which does not provide Equcation for the model)


## 1. Logistic Regression ============================================================================================================================

# Fit the model
model1 = glm(Early_Stage ~ ., data = train_data1, family = "binomial")
summary(model1)
# Predict on test data
test_pred1 = predict(model1, newdata = test_data1, type = "response")
train_pred1 = predict(model1, newdata = train_data1, type = "response")
# Convert to binary
test_pred11 = ifelse(test_pred1 > 0.5, "Yes", "No") %>% factor()
train_pred11 = ifelse(train_pred1 > 0.5, "Yes", "No") %>% factor()

# Confusion matrix
confusionMatrix(test_pred11, test_data1$Early_Stage)
confusionMatrix(train_pred11, train_data1$Early_Stage)

## 2. Random Forest ====================================================================================================================================

# Load required libraries
library(caret)
library(randomForest)

# Load dataset (assuming your data is named 'data')
# Ensure the target variable is a factor with levels 'Yes' and 'No'


# Set seed for reproducibility
set.seed(123)

# Define training control with 10-fold cross-validation
train_control1 <- trainControl(method = "cv", number = 5)

# Define tuning grid for mtry (number of predictors randomly sampled at each split)
tune_grid1 <- expand.grid(mtry = c(1:5,seq(6, 18, by = 2)))

# Train the Random Forest model
rf_model1 <- train(
 Early_Stage ~ ., 
  data = train_data1, 
  method = "rf", 
  trControl = train_control1, 
  tuneGrid = tune_grid1
)

# Print model results
print(rf_model1)


return1=rf_model1$results %>% as.data.frame()

return1 %>% select(c("mtry","Accuracy","Kappa"))  %>% filter(mtry==2| mtry==12 | mtry==1 ) %>% arrange(desc(Accuracy))

# Plot model performance across different mtry values
return1 %>% ggplot(aes(x = mtry, y = Accuracy)) +
  geom_line(colour="Blue") +
  geom_point() +
  labs(title = "Random Forest Model Performance",
       x = "mtry",
       y = "Accuracy") +
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5))+
  scale_x_continuous(breaks = seq(1, 18, by = 2))+
  scale_y_continuous(limits=c(0.86,1),breaks = seq(0.86, 1,by=0.02))

# Making the random forest model with mtry = 2
rf_model11 <- randomForest(Early_Stage ~ ., data = train_data1, mtry = 2, importance = TRUE)
rf_model11

# Predict on test data and training set 
test_pred2 = predict(rf_model11, newdata = test_data1)
train_pred2 = predict(rf_model11, newdata = train_data1)

# Confusion matrix
confusionMatrix(test_pred2, test_data1$Early_Stage)
confusionMatrix(train_pred2, train_data1$Early_Stage)

## 3. SVM ==========================================================================================================================================
library(e1071)

# Define parameter grid
set.seed(123)

# Load required libraries
library(caret)

# Define training control with 5-fold cross-validation
train_control2 <- trainControl(method = "cv", number = 5)

# Define tuning grid for SVM
tune_grid2 <- expand.grid(
  C = 10^seq(-3, 3, by = 1),       # C values: {0.001, 0.01, 0.1, 1, 10, 100, 1000}
  sigma = 10^seq(-4, 1, by = 1)    # gamma values: {0.0001, 0.001, 0.01, 0.1, 1, 10}
)


# Train the SVM model with Radial Kernel
svm_model2 <- train(
  Early_Stage ~ ., 
  data = train_data1, 
  method = "svmRadial",    # SVM with Radial Basis Function (RBF) Kernel
  trControl = train_control2, 
  tuneGrid = tune_grid2
)

# Print model summary
print(svm_model2)
return2=svm_model2$results %>% as.data.frame()
return2$sigma <- format(return2$sigma, scientific = FALSE)
return2=return2 %>% arrange(C,sigma) %>% select(c("C","sigma","Accuracy","Kappa"))
return2$no=1:nrow(return2)
return2 %>% select(c(no,C,sigma,Accuracy,Kappa)) %>% arrange(desc(Accuracy)) %>% head(5)

# Best parameters
return2 %>% ggplot(aes(x=no,y=Accuracy))+ geom_line(colour="Blue") +
  geom_point() +
  labs(title = "SVM PERFORMANCE",
       x = "Combination Number",
       y = "Accuracy") +
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5))+
  scale_x_continuous(limits=c(1,42),breaks = seq(1,42,5))+
  scale_y_continuous(limits=c(0.6,1),breaks = seq(0.6,1,by=0.05))

# Performing the model C=10 , sigma=0.01
set.seed(123)

# Train SVM with the best C and gamma
svm_final_model21 <- svm(Early_Stage ~ ., data = train_data1, 
                       kernel = "radial", 
                       cost = 100,      # Replace with the best C value
                       gamma = 0.001) # Replace with the best gamma value

# Print model summary
summary(svm_final_model21)

# predicting the values for test data & train data
test_pred31=predict(svm_final_model21,test_data1)
train_pred31=predict(svm_final_model21,train_data1)

# confusion Matrix 
confusionMatrix(test_pred31,test_data1$Early_Stage)
confusionMatrix(train_pred31,train_data1$Early_Stage)

## 4.Neural Network ====================================================================================================
library(nnet)
library(caret)

# Ensure target variable is a factor
train_data1$Early_Stage <- as.factor(train_data1$Early_Stage)

# Check for missing values

# Define the hyperparameter grid (only size and decay)
grid4 <- expand.grid(
  size = seq(5, 50, 10),          # Reduce size for better training
  decay = 10^seq(-4, -1, length.out = 5) # Regularization
)

# Define training control (10-fold cross-validation)
control4 <- trainControl(method = "cv", number = 10, classProbs = TRUE)

# Train the model using grid search
set.seed(123)
nn_tuned4 <- train(
  Early_Stage ~ ., 
  data = train_data1,     
  method = "nnet",
  tuneGrid = grid4,
  trControl = control4,
  trace = FALSE, 
  maxit = 100  # Increase iterations for better convergence
)

# Print best hyperparameter
return4=nn_tuned4$results %>% as.data.frame()
return4$Combination=1:nrow(return4)
return41=return4 %>% select(c("Combination","size","decay","Accuracy","Kappa")) %>% arrange(desc(Accuracy))
return41 %>% head(5)

# Plot performance of different models
return4 %>% ggplot(aes(x=Combination,y=Accuracy))+ geom_line(colour="Blue") +
  geom_point() +
  labs(title = "Neural Network Performance",
       x = "Combination Number",
       y = "Accuracy") +
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5))+
  scale_x_continuous(limits=c(1,25),breaks = seq(1,25,4))+
  scale_y_continuous(limits=c(0.6,1),breaks = seq(0.6,1,by=0.05))

# Train the model with the best hyperparameters
set.seed(123)
nn_final_model4 <- nnet(
  Early_Stage ~ ., 
  data = train_data1, 
  size = 5,    # Best size
  decay = 0.1, # Best decay
  maxit = 100, 
  trace = FALSE
)
# predicting the values for test data & train data
test_pred41=predict(nn_final_model4,test_data1,type="class")
train_pred41=predict(nn_final_model4,train_data1,type="class")

# confusion Matrix
confusionMatrix(factor(test_pred41),test_data1$Early_Stage)
confusionMatrix(factor(train_pred41),train_data1$Early_Stage)

## 5. CatBoost ================================================================================================================================
# Data changing for Catboost
data35=data_patient_condition
str(data35)
cat_cols35=c("Gender","Adherence_to_Treatment","Dosage")

# Splitting the data
set.seed(123)
data_split35 = initial_split(data35,prop=0.7,strata=Early_Stage)
train_data35 = training(data_split35)
test_data35 = testing(data_split35)

# Converting the categorical variables to factors
train_data35[cat_cols35] <- lapply(train_data35[cat_cols35], as.factor)
test_data35[cat_cols35] <- lapply(test_data35[cat_cols35], as.factor)

# Converting the target variable to 0 and 1
train_data35$Early_Stage <- as.integer(as.factor(train_data35$Early_Stage)) - 1
test_data35$Early_Stage <- as.integer(as.factor(test_data35$Early_Stage)) - 1

# Training the model
train_pool35 <- catboost.load_pool(
  data = train_data35 %>% select(-Early_Stage),
  label = train_data35$Early_Stage,
  cat_features = cat_cols35
)
# testing model
test_pool35 <- catboost.load_pool(
  data = test_data35 %>% select(-Early_Stage),
  label = test_data35$Early_Stage,
  cat_features = cat_cols35
)

# Training the model
model35 <- catboost.train(
  train_pool35,
  params = list(
    loss_function = "Logloss",  # For binary classification
    iterations = 1000,
    depth = 6,
    learning_rate = 0.1,
    verbose = 100
  )
)
# Predicting and converting the values to factor
pred35train=catboost.predict(model35,train_pool35,prediction_type = "Class") %>% as.factor()
pred35test=catboost.predict(model35,test_pool35,prediction_type = "Class") %>% as.factor()

# Releveling the factor ( predicted)
levels(pred35train)=c("No","Yes")
levels(pred35test)=c("No","Yes")

# Releveling the factor ( actual)

cat_train=train_data35$Early_Stage %>% as.factor()
cat_test=test_data35$Early_Stage %>% as.factor()

# Releveling the factor ( actual)
levels(cat_train)=c("No","Yes")
levels(cat_test)=c("No","Yes")

# confusion Matrix
confusionMatrix(pred35test,cat_test)
confusionMatrix(pred35train,cat_train)

# Forth objective #############################################################################################################################################################################################################################################################

data %>% colnames()

library(FactoMineR)
library(tidyverse)
library(caret)
library(rsample)

# Selecting the variables related to predicting the major issue
data4=read.csv(file.choose(),header = T)
colnames(data4)
ncol(data4)
data_major_issue <- data4 %>% select(
  "Severity_Score","Severity_Interaction","Disease_Activity_Score","Genetic_Mutation_Score",
  "Rate_of_Progression",
 "Muscle_Strength","NCV","Respiratory_Volume","CK_Levels",
  "Age","BMI","Duration_Years","Major_Issue_Type")

data_major_issue$Major_Issue_Type=as.factor(data_major_issue$Major_Issue_Type)


# importing dim reduced data
dim_red_4=read.csv(file.choose(),header = T)
summary(dim_red_4)
nrow(dim_red_4)
dim_red_4=dim_red_4 %>% select(-c("sno"))
data_major_issue=cbind(data_major_issue,dim_red_4)
summary(data_major_issue)
cor(data_major_issue[,-13]) %>% KMO()
bartlett.test(data_major_issue[,-13])

# Splitting the data
set.seed(123)
obj4_data_split = initial_split(data_major_issue,prop=0.7,strata=Major_Issue_Type)
obj4_train_data = training(obj4_data_split)
obj4_test_data = testing(obj4_data_split)

## 1. Random Forest ====================================================================================================================================

# Load required libraries
library(caret)
library(randomForest)

# Load dataset (assuming your data is named 'data')
# Ensure the target variable is a factor with levels 'Yes' and 'No'


# Set seed for reproducibility
set.seed(123)

# Define training control with 10-fold cross-validation
train_control41 <- trainControl(method = "cv", number = 5)

# Define tuning grid for mtry (number of predictors randomly sampled at each split)
tune_grid41 <- expand.grid(mtry = c(1:5,seq(6, 40, by = 5)))

# Train the Random Forest model
rf_model41 <- train(
  Major_Issue_Type ~ ., 
  data = obj4_train_data, 
  method = "rf", 
  trControl = train_control41, 
  tuneGrid = tune_grid41
)

# Print model results
print(rf_model41)


return41=rf_model41$results %>% as.data.frame()

return41 %>% select(c("mtry","Accuracy","Kappa"))  %>% arrange(desc(Accuracy)) %>% head(5)

# Plot model performance across different mtry values
return41 %>% ggplot(aes(x = mtry, y = Accuracy)) +
  geom_line(colour="Blue") +
  geom_point() +
  labs(title = "Random Forest Model Performance",
       x = "mtry",
       y = "Accuracy") +
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5))+
  scale_x_continuous(breaks = seq(1, 40, by = 2))+
  scale_y_continuous(limits=c(0.7,1),breaks = seq(0.7, 1,by=0.05))

# Making the random forest model with mtry = 2
rf_model411 <- randomForest(Major_Issue_Type ~ ., data = obj4_train_data, mtry = 26, importance = TRUE)
rf_model411

# Predict on test data and training set 
test_pred42 = predict(rf_model411, newdata = obj4_test_data)
train_pred42 = predict(rf_model411, newdata = obj4_train_data)

# Confusion matrix
confusionMatrix(test_pred42,obj4_test_data$Major_Issue_Type)
confusionMatrix(train_pred42, obj4_train_data$Major_Issue_Type)


## 2. GLMNET ==========================================================================================================================================


# Define training control with 5-fold cross-validation
library(glmnet)

# Data for Glmnet
glmnet_train_data4=obj4_train_data
glmnet_test_data4=obj4_test_data

# Convert data to matrix format required by glmnet
x_train_obj4 <- model.matrix(Major_Issue_Type ~ ., data = glmnet_train_data4)[, -1]  # Remove intercept column
y_train_obj4 <- glmnet_train_data4$Major_Issue_Type  # Response variable (binary: 0/1)

x_test_obj4 <- model.matrix(Major_Issue_Type ~ ., data = glmnet_test_data4)[, -1]
y_test_obj4 <- glmnet_test_data4$Major_Issue_Type


# Define alpha grid (elastic net mixing parameter)
alpha_grid_obj4 <- seq(0, 1, by = 0.1)  # Grid search from 0 (ridge) to 1 (lasso)

# Create an empty data.frame to store results
cv_results_df_obj4 <- data.frame(
  alpha = numeric(),
  best_lambda = numeric(),
  min_deviance = numeric(),
  Testaccuracy = numeric(),
  Trainaccuracy = numeric()
)

# Perform cross-validation for each alpha and store results in data.frame
set.seed(123)
for (alpha_value in alpha_grid_obj4) {
  cv_model <- cv.glmnet(x_train_obj4, y_train_obj4, family = "multinomial", alpha = alpha_value, nfolds = 10)
  best_lambda <- cv_model$lambda.min
  
  # Train best model
  best_model <- glmnet(x_train_obj4, y_train_obj4, family = "multinomial", alpha = alpha_value, lambda = best_lambda)
  
  # Predict on test and train data
  test_predict <- predict(best_model, newx = x_test_obj4, type = "class") %>% factor()
  train_predict <- predict(best_model, newx = x_train_obj4, type = "class") %>% factor()
  
  # Compute confusion matrices
  test_accuracy <- confusionMatrix(test_predict, y_test_obj4)$overall["Accuracy"]
  train_accuracy <- confusionMatrix(train_predict, y_train_obj4)$overall["Accuracy"]
  
  # Append results to the data frame
  cv_results_df_obj4 <- rbind(cv_results_df_obj4, data.frame(
    alpha = alpha_value,
    best_lambda = best_lambda,  # Optimal lambda
    min_deviance = min(cv_model$cvm),  # Cross-validation error
    Testaccuracy = test_accuracy,
    Trainaccuracy = train_accuracy
  ))
}

cv_results_df_obj4=cv_results_df_obj4 %>% arrange(desc(Trainaccuracy),desc(Testaccuracy))
cv_results_df_obj4

## ggplot for glmnet 
cv_results_df_obj4 %>% select(c(alpha,Trainaccuracy)) %>% 
  ggplot(aes(x=alpha,y=Trainaccuracy))+geom_point()+geom_line(colour="blue")+
  labs(
    title="GLMNET Accuracy",
    x="Alpha",
    y="Train Accuracy"
  )+
  theme(plot.title = element_text(hjust=0.5,face="bold"))+
  scale_x_continuous(limits = c(0,1),breaks = seq(0,1,by=0.1))+
  scale_y_continuous(limits=c(0.75,0.85),breaks = seq(0.75,0.85,by=0.01))

# Fitting the best model 
best_alpha_obj4= cv_results_df_obj4$alpha[1]

# glmnet model with best alpha
model_obj4=cv.glmnet(x_train_obj4, y_train_obj4, family = "multinomial", alpha = best_alpha_obj4)
best_lambda <- model_obj4$lambda.min
plot(model_obj4)
best_lambda

best_model_obj4 <- glmnet(x_train_obj4, y_train_obj4, family = "multinomial", alpha = best_alpha_obj4, lambda = best_lambda)
options(scipen = 999) 
coef(best_model_obj4)

# Predict on test data
test_pred_obj42=predict(best_model_obj4, newx = x_test_obj4, type = "class") %>% factor()
train_pred_obj42=predict(best_model_obj4, newx = x_train_obj4, type = "class") %>% factor()
# confusion matrix
confusionMatrix(test_pred_obj42, y_test_obj4)
confusionMatrix(train_pred_obj42, y_train_obj4)

## 3. XGBoost ==========================================================================================================================================


# Load necessary libraries
library(tidymodels)
library(xgboost)
library(dplyr)
library(caret)
data_for_xg_boost_obj4=data_major_issue
K_obj43=data_for_xg_boost_obj4$Major_Issue_Type %>% levels()
# Convert categorical variable correctly
data_for_xg_boost_obj41 <- data_for_xg_boost_obj4 %>%
  mutate(across(everything(), as.numeric))

# Ensure target variable is numeric starting from 0
data_for_xg_boost_obj41$Major_Issue_Type <- as.integer(factor(data_for_xg_boost_obj41$Major_Issue_Type)) - 1  

# Split the data (70% train, 30% test)
set.seed(123)
data_split_obj43 <- initial_split(data_for_xg_boost_obj41, prop = 0.7, strata = Major_Issue_Type)
xgb_train_data4 <- training(data_split_obj43)
xgb_test_data4 <- testing(data_split_obj43)

# Convert datasets to XGBoost DMatrix
dtrain_obj4 <- xgb.DMatrix(data = as.matrix(xgb_train_data4 %>% select(-Major_Issue_Type)), 
                           label = xgb_train_data4$Major_Issue_Type)
dtest_obj4 <- xgb.DMatrix(data = as.matrix(xgb_test_data4 %>% select(-Major_Issue_Type)), 
                          label = xgb_test_data4$Major_Issue_Type)

# Define hyperparameter grid
grid_obj4 <- expand.grid(
  max_depth = c(3, 6, 9),    
  eta = c(0.01, 0.1, 0.3),   
  nrounds = c(50, 100, 150)  
)

# Tune XGBoost without manual loops
results_obj4 <- grid_obj4 %>%
  rowwise() %>%
  mutate(
    model = list(
      xgb.train(
        params = list(
          objective = "multi:softmax",  
          num_class = length(unique(xgb_train_data4$Major_Issue_Type)), # Specify number of classes
          max_depth = max_depth,    
          eta = eta,
          eval_metric = "mlogloss"
        ),
        data = dtrain_obj4,
        nrounds = nrounds,
        verbose = 0
      )
    ),
    preds_train = list(predict(model[[1]], dtrain_obj4)),  # Predictions on training set
    preds_test = list(predict(model[[1]], dtest_obj4)),    # Predictions on testing set
    
    # Training Set Accuracy
    accuracy_train = mean(preds_train[[1]] == xgb_train_data4$Major_Issue_Type),
    
    # Testing Set Accuracy
    accuracy_test = mean(preds_test[[1]] == xgb_test_data4$Major_Issue_Type)
  ) %>%
  select(max_depth, eta, nrounds, accuracy_train, accuracy_test) %>%
  as.data.frame()  # Convert to proper table format

# Add serial numbers to results
results_obj4$S.no <- 1:nrow(results_obj4)

# Display top 5 models sorted by accuracy
top_models <- results_obj4 %>% arrange(desc(accuracy_train), desc(accuracy_test)) %>% head(5)
print(top_models)

# Select the best model hyperparameters
best_params <- top_models[1,]  # Pick the best row

# Train final model with the best hyperparameters
best_model <- xgb.train(
  params = list(
    objective = "multi:softmax",
    num_class = length(unique(xgb_train_data4$Major_Issue_Type)),
    max_depth = best_params$max_depth,
    eta = best_params$eta,
    eval_metric = "mlogloss"
  ),
  data = dtrain_obj4,
  nrounds = best_params$nrounds,
  verbose = 0
)

# Make final predictions on test set
test_predictions_obj43 <- predict(best_model, dtest_obj4) %>% factor()
levels(test_predictions_obj43)=K_obj43
xgb_test_data4$Major_Issue_Type=factor(xgb_test_data4$Major_Issue_Type)
levels(xgb_test_data4$Major_Issue_Type)=K_obj43

train_predictions_obj43=predict(best_model,dtrain_obj4) %>% factor()
levels(train_predictions_obj43)=K_obj43
xgb_train_data4$Major_Issue_Type=factor(xgb_train_data4$Major_Issue_Type)
levels(xgb_train_data4$Major_Issue_Type)=K_obj43
# Generate confusion matrix
conf_matrix_test_obj43 <- confusionMatrix(test_predictions_obj43,(xgb_test_data4$Major_Issue_Type))
print(conf_matrix_test_obj43)
conf_matrix_train_obj43=confusionMatrix(train_predictions_obj43,(xgb_train_data4$Major_Issue_Type))
print(conf_matrix_train_obj43)

## 4.SVM ==================================================================================================
data_for_obj4_svm4=data_major_issue
data_for_obj4_svm4
# data Spliting 
data_split_obj4_svm=initial_split(data_for_obj4_svm4,0.7,Major_Issue_Type)
train_data_for_obj4_svm4=training(data_split_obj4_svm)
test_data_for_obj4_svm4=testing(data_split_obj4_svm)

# Define training control with 5-fold cross-validation
train_control_obj4_svm <- trainControl(method = "cv", number = 5)

# Define tuning grid for SVM
tune_grid_obj4_svm <- expand.grid(
  C = 10^seq(-3, 3, by = 1),       # C values: {0.001, 0.01, 0.1, 1, 10, 100, 1000}
  sigma = 10^seq(-4, 1, by = 1)    # gamma values: {0.0001, 0.001, 0.01, 0.1, 1, 10}
)


# Train the SVM model with Radial Kernel
svm_model_obj4_svm <- train(
  Major_Issue_Type ~ ., 
  data = train_data_for_obj4_svm4, 
  method = "svmRadial",    # SVM with Radial Basis Function (RBF) Kernel
  trControl = train_control_obj4_svm, 
  tuneGrid = tune_grid_obj4_svm
)

# Print model summary
print(svm_model_obj4_svm)
return_obj4_svm=svm_model_obj4_svm$results %>% as.data.frame()
return_obj4_svm=return_obj4_svm%>% arrange(C,sigma) %>% select(c("C","sigma","Accuracy","Kappa"))
return_obj4_svm$no=1:nrow(return_obj4_svm)
return_obj4_svm %>% select(c(no,C,sigma,Accuracy,Kappa)) %>% arrange(desc(Accuracy)) %>% head(5)

# Best parameters
return_obj4_svm %>% ggplot(aes(x=no,y=Accuracy))+ geom_line(colour="Blue") +
  geom_point() +
  labs(title = "SVM PERFORMANCE",
       x = "Combination Number",
       y = "Accuracy") +
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5))+
  scale_x_continuous(limits=c(1,42),breaks = seq(1,42,5))+
  scale_y_continuous(limits=c(0.46,0.76),breaks = seq(0.46,0.76,by=0.03))

# Performing the model C=10 , sigma=0.01
set.seed(123)

# Train SVM with the best C and gamma
obj4_svm_final_model4 <- svm(Major_Issue_Type ~ ., data = train_data_for_obj4_svm4, 
                         kernel = "radial", 
                         cost = 100,      # Replace with the best C value
                         gamma = 0.01) # Replace with the best gamma value

# Print model summary
summary(obj4_svm_final_model4)

# predicting the values for test data & train data
train_pred_for_obj4_svm4=predict(obj4_svm_final_model4,train_data_for_obj4_svm4)
test_pred_for_obj4_svm4=predict(obj4_svm_final_model4,test_data_for_obj4_svm4)

# confusion Matrix 
confusionMatrix(train_pred_for_obj4_svm4,train_data_for_obj4_svm4$Major_Issue_Type)
confusionMatrix(test_pred_for_obj4_svm4,test_data_for_obj4_svm4$Major_Issue_Type)

## 5.Decision Tree ==========================================================
obj4_decision_tree_train_data=obj4_train_data
obj4_decision_tree_test_data=obj4_test_data

tunegrid_obj4_decision_tree=tuneGrid <- expand.grid(
  cp = seq(0.001, 0.05, by = 0.005)  # Complexity parameter tuning
)
control_obj4_decision_tree <- trainControl(method = "cv", number = 10) 

dt_tuned_obj4_decision_tree =train(
  Major_Issue_Type ~ ., data = obj4_decision_tree_train_data,
  method = "rpart",    # Use "rpart" instead of "rpart2"
  trControl = control_obj4_decision_tree ,
  tuneGrid = tunegrid_obj4_decision_tree,
  parms = list(split = "gini")  # Use Gini impurity (default)
)
print(dt_tuned_obj4_decision_tree)

print(dt_tuned_obj4_decision_tree$bestTune)

best_model_obj4_decision_tree=dt_tuned_obj4_decision_tree$bestTune

# predict on test and & train data

obj4_predicted_testdata_decision_tree=predict(dt_tuned_obj4_decision_tree,obj4_decision_tree_test_data)

obj4_predicted_traindata_decision_tree=predict(dt_tuned_obj4_decision_tree,obj4_decision_tree_train_data)

#Confusion Matrix
confusionMatrix(obj4_predicted_testdata_decision_tree,obj4_decision_tree_test_data$Major_Issue_Type)

confusionMatrix(obj4_predicted_traindata_decision_tree,obj4_decision_tree_train_data$Major_Issue_Type)


# Fifth Objective ######################################












