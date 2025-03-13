# Libraries
library(dplyr)
library(tidyverse)
library(ggplot2)
library(lsr)
library(readr)

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




