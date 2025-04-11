Libraries
library(dplyr)
library(tidyverse)
library(ggplot2)
library(lsr)
library(readr)
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
summary(data)
data=data %>% select(-c("Patient_ID","Treatment_Interaction","Lifestyle_Factor"))

# Chi square Test 

# Chi- square test for Adherence to Treatment vs Early_stage

xtabs(~Adherence_to_Treatment+Early_Stage,data=data) %>% addmargins()
chi_sq1=chisq.test(data$Adherence_to_Treatment,data$Early_Stage)
chi_sq1
chi_sq1$expected
data %>% group_by(Adherence_to_Treatment,Early_Stage) %>% summarise(count=n()) %>% 
  ggplot(aes(x=Adherence_to_Treatment,y=count,fill=Early_Stage)) +
  geom_bar(stat="identity",position="dodge") + theme_minimal() +
  labs(title="Adherence to Treatment vs Early stage",x="Adherence to Treatment",y="Count",fill= "Early Stage") + 
  theme(
    plot.title = element_text(hjust=0.5,face="bold"))+
  geom_text(aes(label=count),vjust=1.5,position=position_dodge(0.9),size=3.5)

## cramer's v test for Adherence to Treatment vs Early_stage
cramersV(data$Adherence_to_Treatment,data$Early_Stage)
### Result
# There is no relationship between Adherence to Treatment and Early stage as the p-value is greater than 0.05 and Cramer's V value is 0.0

# Chi- square for Cured vs Severity_score_factor
xtabs(~Cured+Severity_score_factor,data=data) %>% addmargins()
chi_sq2=chisq.test(data$Cured,data$Severity_score_factor)
chi_sq2
chi_sq2$expected
data %>% group_by(Severity_score_factor,Cured) %>% summarise(count=n()) %>% 
  ggplot(aes(x=Severity_score_factor,y=count,fill=Cured)) +
  geom_bar(stat="identity",position="dodge") + theme_minimal() +
  labs(title="Severity_score_factor vs Cured",x="Severity_score_factor",y="Count",fill= "Cured") + 
  theme(
    plot.title = element_text(hjust=0.5,face="bold"))+
  geom_text(aes(label=count),vjust=1.5,position=position_dodge(0.9),size=3.5)

## cramer's v test for Cured vs Severity_score_factor
cramersV(data$Cured,data$Severity_score_factor)
### Result
## The chi-square p-value is less than 0.05, indicating a significant relationship between the variables; to assess the strength of this association, we used Cramér's V, which yielded a value of 0.363291, suggesting a moderate association between the variables.

# chi-square for Treatment vs Died_while_treatment
xtabs(data=data,~Treatment+Died_while_Treatment) %>% addmargins()
chi_sq3=chisq.test(data$Treatment,data$Died_while_Treatment)
chi_sq3
chi_sq3$expected
data %>% group_by(Treatment,Died_while_Treatment) %>% summarise(count=n()) %>% 
  ggplot(aes(x=Treatment,y=count,fill=Died_while_Treatment)) +
  geom_bar(stat="identity",position="dodge") + theme_minimal() +
  labs(title="Treatment vs Cured",x="Treatment",y="Count",fill= "Died_while_Treatment") + 
  theme(
    plot.title = element_text(hjust=0.5,face="bold"),
    legend.position = "bottom")+
  geom_text(aes(label=count),vjust=1.5,position=position_dodge(0.9),size=3.5)

##cramer's v test for Treatment vs Died_while_treatment
cramersV(data$Treatment,data$Died_while_Treatment)
### Result 
## Hence the chi square value is greater than 0.05 , so there is no relationship between the Treatment and Died while Treatment

# Chi-square for Major issue type vs Quality of life improvement

xtabs(data=data,~Major_Issue_Type+Quality_of_Life_Improvement) %>% addmargins()
chi_sq4=chisq.test(data$Major_Issue_Type,data$Quality_of_Life_Improvement)
chi_sq4
chi_sq4$expected
data %>% group_by(Major_Issue_Type,Quality_of_Life_Improvement) %>% summarise(count=n()) %>% 
  ggplot(aes(x=Major_Issue_Type,y=count,fill=Quality_of_Life_Improvement)) +
  geom_bar(stat="identity",position="dodge") + theme_minimal() +
  labs(title="Major Issue Type vs Quality of Life Improvement",x="Major Issue Type",y="Count",fill= "Quality of Life Improvement") + 
  theme(
    plot.title = element_text(hjust=0.5,face="bold"),
    legend.position = "bottom")+
  geom_text(aes(label=count),vjust=1.5,position=position_dodge(0.9),size=3.5)
## cramer's v test for Major issue type vs Quality of life improvement
cramersV(data$Major_Issue_Type,data$Quality_of_Life_Improvement)
### Result 
## The Chi-square p-value is less than 0.05, indicating a significant relationship between Major Issue Type and Quality of Life Improvement; using Cramér's V to assess the strength of this relationship yielded a value of 0.17893, suggesting a weak to moderate association between the variables.

# Conditional Chi square 

# chi square for Major issue Type vs adherence to treatment
xtabs(data=data,~Major_Issue_Type+Adherence_to_Treatment) %>% addmargins()
chi_sq56=chisq.test(data$Major_Issue_Type,data$Adherence_to_Treatment)
chi_sq56
chi_sq56$expected
data %>% group_by(Major_Issue_Type,Adherence_to_Treatment) %>% summarise(count=n()) %>% 
  ggplot(aes(x=Major_Issue_Type,y=count,fill=Adherence_to_Treatment)) +
  geom_bar(stat="identity",position="dodge") + theme_minimal() +
  labs(title="Major Issue Type vs Adherence to Treatment",x="Major Issue Type",y="Count",fill= "Adherence to Treatment") + 
  theme(
    plot.title = element_text(hjust=0.5,face="bold"),
    legend.position = "bottom")+
  geom_text(aes(label=count),vjust=1.5,position=position_dodge(0.9),size=3.5)
## cramer's v test for Major issue type vs adherence to treatment
cramersV(data$Major_Issue_Type,data$Adherence_to_Treatment)
# Now we spliting the data based on Early stage to chack the relationship between major issue type and adherence to tretment is differed by early stage or not
# chi-square for Major issue Type vs adherence to treatment given that the early stage is yes
xtabs(data=data,~Major_Issue_Type+Adherence_to_Treatment+Early_Stage) %>% addmargins()
conditional_data11=data %>% filter(Early_Stage=="Yes")
chi_sq5=chisq.test(conditional_data11$Major_Issue_Type,conditional_data11$Adherence_to_Treatment)
chi_sq5
chi_sq5$expected
fisher.test(conditional_data11$Major_Issue_Type,conditional_data11$Adherence_to_Treatment)
conditional_data11 %>% group_by(Major_Issue_Type,Adherence_to_Treatment) %>% summarise(count=n()) %>% 
  ggplot(aes(x=Major_Issue_Type,y=count,fill=Adherence_to_Treatment)) +
  geom_bar(stat="identity",position="dodge") + theme_minimal() +
  labs(title="Major Issue Type vs Adherence to Treatment  ",subtitle = "Filtered by Early stage is yes",x="Major Issue Type",y="Count",fill= "Adherence to Treatment") + 
  theme(
    plot.title = element_text(hjust=0.5,face="bold"),
    legend.position = "bottom",
    plot.subtitle = element_text(hjust=0.5,face="bold"))+
  geom_text(aes(label=count),vjust=1.5,position=position_dodge(0.9),size=3.5)
## cramer's v test for Major issue type vs adherence to treatment given that the early stage is yes 
cramersV(conditional_data11$Major_Issue_Type,conditional_data11$Adherence_to_Treatment)

conditional_data12=data %>% filter(Early_Stage=="No")
chi_sq6=chisq.test(conditional_data12$Major_Issue_Type,conditional_data12$Adherence_to_Treatment)
chi_sq6
chi_sq6$expected
conditional_data12 %>% group_by(Major_Issue_Type,Adherence_to_Treatment) %>% summarise(count=n()) %>% 
  ggplot(aes(x=Major_Issue_Type,y=count,fill=Adherence_to_Treatment)) +
  geom_bar(stat="identity",position="dodge") + theme_minimal() +
  labs(title="Major Issue Type vs Adherence to Treatment  ",subtitle = "Filtered by Early stage is yes",x="Major Issue Type",y="Count",fill= "Adherence to Treatment") + 
  theme(
    plot.title = element_text(hjust=0.5,face="bold"),
    legend.position = "bottom",
    plot.subtitle = element_text(hjust=0.5,face="bold"))+
  geom_text(aes(label=count),vjust=1.5,position=position_dodge(0.9),size=3.5)
## cramer's v test for Major issue type vs adherence to treatment given that the early stage is no
cramersV(conditional_data12$Major_Issue_Type,conditional_data12$Adherence_to_Treatment)
## Result 
## 
## conditional chi square  for Died while treatment vs major issue type over Adherence to treatment
xtabs(data=data,~Died_while_Treatment+Major_Issue_Type+Adherence_to_Treatment) %>% addmargins()
chi_sq78=chisq.test(data$Died_while_Treatment,data$Major_Issue_Type)
chi_sq78
chi_sq78$expected
data %>% group_by(Died_while_Treatment,Major_Issue_Type) %>% summarise(count=n()) %>% 
  ggplot(aes(x=Died_while_Treatment,y=count,fill=Major_Issue_Type)) +
  geom_bar(stat="identity",position="dodge") + theme_minimal() +
  labs(title="Died while Treatment vs Major Issue Type",x="Died while Treatment",y="Count",fill= "Major Issue Type") + 
  theme(
    plot.title = element_text(hjust=0.5,face="bold"),
    legend.position = "bottom")+
  geom_text(aes(label=count),vjust=1.5,position=position_dodge(0.9),size=3.5)
## Cramer`s V test for Died While Treatment Vs Major Issue Type
cramersV(data$Died_while_Treatment,data$Major_Issue_Type)
## Result

## now we split the data based on Adherence to treatment to check the relationship between Died while treatment and Major issue type is differed by Adherence to treatment or not
conditional_data21=data %>% filter(Adherence_to_Treatment=="Yes")
chi_sq7=chisq.test(conditional_data21$Died_while_Treatment,conditional_data21$Major_Issue_Type)
chi_sq7
chi_sq7$expected
fisher.test(conditional_data21$Died_while_Treatment,conditional_data21$Major_Issue_Type)
conditional_data21 %>% group_by(Died_while_Treatment,Major_Issue_Type) %>% summarise(count=n()) %>% 
  ggplot(aes(x=Died_while_Treatment,y=count,fill=Major_Issue_Type)) +
  geom_bar(stat="identity",position="dodge") + theme_minimal() +
  labs(title="Died while Treatment vs Major Issue Type  ",subtitle = "Filtered by Adherence to Treatment is yes",x="Died while Treatment",y="Count",fill= "Major Issue Type") + 
  theme(
    plot.title = element_text(hjust=0.5,face="bold"),
    legend.position = "bottom",
    plot.subtitle = element_text(hjust=0.5,face="bold"))+
  geom_text(aes(label=count),vjust=1.5,position=position_dodge(0.9),size=3.5)
## cramer's v test for died while treatment vs major issue type given that the adherence to treatment is yes 
cramersV(conditional_data21$Died_while_Treatment,conditional_data21$Major_Issue_Type)

conditional_data22=data %>% filter(Adherence_to_Treatment=="No")
chi_sq8=chisq.test(conditional_data22$Died_while_Treatment,conditional_data22$Major_Issue_Type)
chi_sq8
chi_sq8$expected
conditional_data22 %>% group_by(Died_while_Treatment,Major_Issue_Type) %>% summarise(count=n()) %>% 
  ggplot(aes(x=Died_while_Treatment,y=count,fill=Major_Issue_Type)) +
  geom_bar(stat="identity",position="dodge") + theme_minimal() +
  labs(title="Died while Treatment vs Major Issue Type  ",subtitle = "Filtered by Adherence to Treatment is No",x="Died while Treatment",y="Count",fill= "Major Issue Type") + 
  theme(
    plot.title = element_text(hjust=0.5,face="bold"),
    legend.position = "bottom",
    plot.subtitle = element_text(hjust=0.5,face="bold"))+
  geom_text(aes(label=count),vjust=1.5,position=position_dodge(0.9),size=3.5)
## cramer's v test for died while treatment vs major issue type given that the adherence to treatment is no 
cramersV(conditional_data22$Died_while_Treatment,conditional_data22$Major_Issue_Type)

# Avova Test 
## Computing two way anova for major issue Type and severity score factor on muscle Strength

two_way_anova1= aov(Muscle_Strength~Major_Issue_Type*Severity_score_factor,data=data)
two_way_anova1
summary(two_way_anova1)

## using post hoc method
tukey_result <- TukeyHSD(two_way_anova1)
tukey_result
tukey_result$`Major_Issue_Type:Severity_score_factor`
tukey_result$`Major_Issue_Type`
tukey_result$`Severity_score_factor`

anova1_summary_Major_Issue_Type=data %>%
  group_by(Major_Issue_Type) %>%
  summarise(
    mean_Muscle_Strength = mean(Muscle_Strength),
    se_Muscle_Strength = sd(Muscle_Strength) / sqrt(n())  
  )
anova1_summary_Major_Issue_Type

# Bar plot with error bars for Major_Issue_Type and displaying mean values
ggplot(anova1_summary_Major_Issue_Type, aes(x = Major_Issue_Type, y = mean_Muscle_Strength, fill = Major_Issue_Type)) +
  geom_bar(stat = "identity", width = 0.6, color = "black") +  
  geom_errorbar(aes(ymin = mean_Muscle_Strength - se_Muscle_Strength, ymax = mean_Muscle_Strength  + se_Muscle_Strength), width = 0.2) +  # Error bars
  geom_text(aes(label = round(mean_Muscle_Strength , 2)), vjust = 2.8, color = "black", size = 4) +  # Mean labels
  labs( 
    x = "Major_Issue_Type", 
    y = "mean_Muscle_Strength ",
    title = "Comparing Means in Major issue Type  ") +
  theme_minimal()+theme(legend.position = "none",plot.title = element_text(hjust = 0.5, face = "bold"))

anova1_summary_Severity_score_factor=data %>%
  group_by(Severity_score_factor) %>%
  summarise(
    mean_Muscle_Strength = mean(Muscle_Strength),
    se_Muscle_Strength = sd(Muscle_Strength) / sqrt(n())  
  )
anova1_summary_Severity_score_factor


# Bar plot with error bars for Severity_score_factor and displaying mean values
ggplot(anova1_summary_Severity_score_factor, aes(x = Severity_score_factor, y = mean_Muscle_Strength, fill = Severity_score_factor)) +
  geom_bar(stat = "identity", width = 0.6, color = "black") +  
  geom_errorbar(aes(ymin = mean_Muscle_Strength - se_Muscle_Strength, ymax = mean_Muscle_Strength  + se_Muscle_Strength), width = 0.2) +  # Error bars
  geom_text(aes(label = round(mean_Muscle_Strength , 2)), vjust = 3.1, color = "black", size = 4) +  # Mean labels
  labs( 
    x = "Severity_score_factor", 
    y = "mean_Muscle_Strength ",
    title="Comparing Means in Severity Score") +
  theme_minimal()+theme(legend.position = "none",
                        plot.title = element_text(hjust = 0.5, face = "bold"))

## computing two way anova for Gender and Disorder type on Muscle Strength

two_way_anova2= aov(Muscle_Strength~Gender*Disorder,data=data)
two_way_anova2
summary(two_way_anova2)

anova2_summary_Gender=data %>%
  group_by(Gender) %>%
  summarise(
    mean_Muscle_Strength = mean(Muscle_Strength),
    se_Muscle_Strength = sd(Muscle_Strength) / sqrt(n())  
  )
anova2_summary_Gender

ggplot(anova2_summary_Gender, aes(x = Gender, y = mean_Muscle_Strength, fill = Gender)) +
  geom_bar(stat = "identity", width = 0.6, color = "black") +  
  geom_errorbar(aes(ymin = mean_Muscle_Strength - se_Muscle_Strength, ymax = mean_Muscle_Strength  + se_Muscle_Strength), width = 0.2) +  # Error bars
  geom_text(aes(label = round(mean_Muscle_Strength , 2)), vjust = 2.5, color = "black", size = 4) +  # Mean labels
  labs( 
    x = "Gender", 
    y = "mean_Muscle_Strength ",
    title="Comparing Means in Gender") +
  theme_minimal()+theme(legend.position = "none",
                        plot.title = element_text(hjust = 0.5, face = "bold"))

anova2_summary_Disorder=data %>%
  group_by(Disorder) %>%
  summarise(
    mean_Muscle_Strength = mean(Muscle_Strength),
    se_Muscle_Strength = sd(Muscle_Strength) / sqrt(n())  
  )
anova2_summary_Disorder

ggplot(anova2_summary_Disorder, aes(x = Disorder, y = mean_Muscle_Strength, fill = Disorder)) +
  geom_bar(stat = "identity", width = 0.6, color = "black") +  
  geom_errorbar(aes(ymin = mean_Muscle_Strength - se_Muscle_Strength, ymax = mean_Muscle_Strength  + se_Muscle_Strength), width = 0.2) +  # Error bars
  geom_text(aes(label = round(mean_Muscle_Strength , 2)), vjust = 4, color = "black", size = 4) +  # Mean labels
  labs( 
    x = "Disorder", 
    y = "mean_Muscle_Strength ",
    title="Comparing Means in Disorder") +
  theme_minimal()+theme(legend.position = "none",
                        plot.title = element_text(hjust = 0.5, face = "bold")) + theme(axis.text.x = element_text(size = 7,
                                                                                                                  angle = 90))

## two way Anova for Dosage and Adharance to treatment on Rate of Progression
shapiro.test(data$Rate_of_Progression)
two_way_anova3= aov(Rate_of_Progression~Dosage*Adherence_to_Treatment,data=data)
two_way_anova3
summary(two_way_anova3)

anova3_summary_Dosage=data %>%
  group_by(Dosage) %>%
  summarise(
    mean_Rate_of_Progression = mean(Rate_of_Progression),
    se_Rate_of_Progression = sd(Rate_of_Progression) / sqrt(n())  
  )
anova3_summary_Dosage
ggplot(anova3_summary_Dosage, aes(x = Dosage, y = mean_Rate_of_Progression, fill = Dosage)) +
  geom_bar(stat = "identity", width = 0.6, color = "black") +  
  geom_errorbar(aes(ymin = mean_Rate_of_Progression - se_Rate_of_Progression, ymax = mean_Rate_of_Progression  + se_Rate_of_Progression), width = 0.2) +  # Error bars
  geom_text(aes(label = round(mean_Rate_of_Progression , 2)), vjust = 2.8, color = "black", size = 4) +  # Mean labels
  labs( 
    x = "Dosage", 
    y = "mean_Rate_of_Progression ",
    title="Comparing Means in Dosage") +
  theme_minimal()+theme(legend.position = "none",
                        plot.title = element_text(hjust = 0.5, face = "bold"))

anova3_summary_Adherence_to_Treatment=data %>%
  group_by(Adherence_to_Treatment) %>%
  summarise(
    mean_Rate_of_Progression = mean(Rate_of_Progression),
    se_Rate_of_Progression = sd(Rate_of_Progression) / sqrt(n())  
  )
anova3_summary_Adherence_to_Treatment
ggplot(anova3_summary_Adherence_to_Treatment, aes(x = Adherence_to_Treatment, y = mean_Rate_of_Progression, fill = Adherence_to_Treatment)) +
  geom_bar(stat = "identity", width = 0.6, color = "black") +  
  geom_errorbar(aes(ymin = mean_Rate_of_Progression - se_Rate_of_Progression, ymax = mean_Rate_of_Progression  + se_Rate_of_Progression), width = 0.2) +  # Error bars
  geom_text(aes(label = round(mean_Rate_of_Progression , 2)), vjust = 2.8, color = "black", size = 4) +  # Mean labels
  labs( 
    x = "Adherence_to_Treatment", 
    y = "mean_Rate_of_Progression ",
    title="Comparing Means in Adherence to Treatment") +
  theme_minimal()+theme(legend.position = "none",
                        plot.title = element_text(hjust = 0.5, face = "bold"))

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


