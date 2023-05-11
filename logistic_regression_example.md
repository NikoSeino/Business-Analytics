Logistic Regression
================
Niko Seino
2023-05-11

- <a href="#business-task" id="toc-business-task">Business Task</a>
- <a href="#prep-the-data" id="toc-prep-the-data">Prep the Data</a>
- <a href="#logistic-regression-analysis"
  id="toc-logistic-regression-analysis">Logistic Regression Analysis</a>

## Business Task

Create a model that will predict whether or not a customer will default
on a loan, based on certain demographic categories.

## Prep the Data

Load libraries

``` r
rm(list = ls())

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(ROCR))
suppressPackageStartupMessages(library(ROSE))
```

Load data

``` r
#set working directory
setwd("~/DTSC560")

#read dataset into R
optivadf <- read.csv("optiva.csv")
#View(optivadf)
```

Convert Categorical variables to factors with levels and labels

``` r
optivadf$LoanDefault<-factor(optivadf$LoanDefault,levels = c(0,1),labels = c("No","Yes"))
optivadf$Entrepreneur<-factor(optivadf$Entrepreneur,levels = c(0,1),labels = c("No","Yes"))
optivadf$Unemployed<-factor(optivadf$Unemployed,levels = c(0,1),labels = c("No","Yes"))
optivadf$Married<-factor(optivadf$Married,levels = c(0,1),labels = c("No","Yes"))
optivadf$Divorced<-factor(optivadf$Divorced,levels = c(0,1),labels = c("No","Yes"))
optivadf$HighSchool<-factor(optivadf$HighSchool,levels = c(0,1),labels = c("No","Yes"))
optivadf$College<-factor(optivadf$College,levels = c(0,1),labels = c("No","Yes"))
```

Generate summary stats for the variables

``` r
#generate summary statistics for all variables in dataframe
summary(optivadf)
```

    ##    CustomerID    LoanDefault AverageBalance          Age        Entrepreneur
    ##  Min.   :    1   No :42411   Min.   :     1.3   Min.   :18.00   No :40242   
    ##  1st Qu.:10799   Yes:  782   1st Qu.:   176.8   1st Qu.:33.00   Yes: 2951   
    ##  Median :21597               Median :   625.3   Median :39.00               
    ##  Mean   :21597               Mean   :  1836.7   Mean   :40.76               
    ##  3rd Qu.:32395               3rd Qu.:  1849.9   3rd Qu.:48.00               
    ##  Max.   :43193               Max.   :132765.1   Max.   :95.00               
    ##  Unemployed  Married     Divorced    HighSchool  College    
    ##  No :41919   No :17247   No :38165   No :20062   No :29931  
    ##  Yes: 1274   Yes:25946   Yes: 5028   Yes:23131   Yes:13262  
    ##                                                             
    ##                                                             
    ##                                                             
    ## 

## Logistic Regression Analysis

1.  Partition the data into test, training, and validation sets

``` r
#set seed so the random sample is reproducible
set.seed(42)

#Partition the dataset into a training, validation and test set
Samples<-sample(seq(1,3),size=nrow(optivadf),replace=TRUE,prob=c(0.6,0.2,0.2))
Train<-optivadf[Samples==1,]
Validate<-optivadf[Samples==2,]
Test<-optivadf[Samples==3,]

#View descriptive statistics for each dataframe
# summary(Train)
# summary(Validate)
# summary(Test)
```

2.  Test for imbalance by creating undersample, oversample, and ROSE
    subsets from the training set

``` r
#Create a data frame with only the predictor variables by removing 
#column 2 (Loan Default)
xsdf<-Train[c(-2)]
# View(xsdf)

#Create an undersampled training subset
undersample<-downSample(x=xsdf, y=Train$LoanDefault, yname = "LoanDefault")

table(undersample$LoanDefault)
```

    ## 
    ##  No Yes 
    ## 491 491

``` r
#Create an oversampled training subset
oversample<-upSample(x=xsdf, y=Train$LoanDefault, yname = "LoanDefault")

table(oversample$LoanDefault)
```

    ## 
    ##    No   Yes 
    ## 25450 25450

``` r
#Create a training subset with ROSE
rose<-ROSE(LoanDefault ~ ., data  = Train)$data                         

table(rose$LoanDefault)
```

    ## 
    ##    No   Yes 
    ## 12969 12972

3.  Perform logistic regressions on each subset

``` r
options(scipen=999)
lrUnder <- glm(LoanDefault ~ . - CustomerID, data = undersample, 
               family = binomial(link = "logit"))

# undersample model summary
summary(lrUnder)
```

    ## 
    ## Call:
    ## glm(formula = LoanDefault ~ . - CustomerID, family = binomial(link = "logit"), 
    ##     data = undersample)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.8581  -1.1916   0.3068   1.0384   3.4218  
    ## 
    ## Coefficients:
    ##                    Estimate  Std. Error z value           Pr(>|z|)    
    ## (Intercept)      0.34002176  0.36456480   0.933             0.3510    
    ## AverageBalance  -0.00060786  0.00007925  -7.670 0.0000000000000172 ***
    ## Age             -0.00529762  0.00807387  -0.656             0.5117    
    ## EntrepreneurYes  0.59697356  0.26351246   2.265             0.0235 *  
    ## UnemployedYes    0.53765902  0.38584130   1.393             0.1635    
    ## MarriedYes       0.08629535  0.16985495   0.508             0.6114    
    ## DivorcedYes      0.55757129  0.24565578   2.270             0.0232 *  
    ## HighSchoolYes    0.28282501  0.19416178   1.457             0.1452    
    ## CollegeYes       0.16447219  0.22016243   0.747             0.4550    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1361.3  on 981  degrees of freedom
    ## Residual deviance: 1204.2  on 973  degrees of freedom
    ## AIC: 1222.2
    ## 
    ## Number of Fisher Scoring iterations: 6

``` r
# fit logistic regression model on the LoanDefault outcome variable
# using specified input variables with the oversample dataframe
lrOver <- glm(LoanDefault ~ . - CustomerID, data = oversample, 
              family = binomial(link = "logit"))

# oversample model summary
summary(lrOver)
```

    ## 
    ## Call:
    ## glm(formula = LoanDefault ~ . - CustomerID, family = binomial(link = "logit"), 
    ##     data = oversample)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.7770  -1.1989   0.3367   1.0221   3.6088  
    ## 
    ## Coefficients:
    ##                    Estimate  Std. Error z value             Pr(>|z|)    
    ## (Intercept)      0.65687135  0.04955864  13.254 < 0.0000000000000002 ***
    ## AverageBalance  -0.00068204  0.00001199 -56.897 < 0.0000000000000002 ***
    ## Age             -0.00568791  0.00108275  -5.253   0.0000001494784898 ***
    ## EntrepreneurYes  0.47890377  0.03511918  13.637 < 0.0000000000000002 ***
    ## UnemployedYes    0.40398245  0.05248362   7.697   0.0000000000000139 ***
    ## MarriedYes       0.00625761  0.02399575   0.261             0.794262    
    ## DivorcedYes      0.35968381  0.03366683  10.684 < 0.0000000000000002 ***
    ## HighSchoolYes    0.09094574  0.02753778   3.303             0.000958 ***
    ## CollegeYes      -0.17204639  0.03088280  -5.571   0.0000000253361255 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 70562  on 50899  degrees of freedom
    ## Residual deviance: 62915  on 50891  degrees of freedom
    ## AIC: 62933
    ## 
    ## Number of Fisher Scoring iterations: 6

``` r
# fit logistic regression model on the LoanDefault outcome variable
# using specified input variables with the rose dataframe

lrrose <- glm(LoanDefault ~ . - CustomerID, data = rose, 
              family = binomial(link = "logit"))

# ROSE model summary
summary(lrrose)
```

    ## 
    ## Call:
    ## glm(formula = LoanDefault ~ . - CustomerID, family = binomial(link = "logit"), 
    ##     data = rose)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.0704  -1.1495   0.6446   1.0762   2.9009  
    ## 
    ## Coefficients:
    ##                    Estimate  Std. Error z value             Pr(>|z|)    
    ## (Intercept)      0.50737034  0.06573694   7.718   0.0000000000000118 ***
    ## AverageBalance  -0.00042216  0.00001106 -38.157 < 0.0000000000000002 ***
    ## Age             -0.00450765  0.00139790  -3.225              0.00126 ** 
    ## EntrepreneurYes  0.50151948  0.04842316  10.357 < 0.0000000000000002 ***
    ## UnemployedYes    0.46822742  0.06950581   6.737   0.0000000000162223 ***
    ## MarriedYes      -0.05874034  0.03276963  -1.793              0.07305 .  
    ## DivorcedYes      0.41140381  0.04626243   8.893 < 0.0000000000000002 ***
    ## HighSchoolYes    0.05480798  0.03820934   1.434              0.15145    
    ## CollegeYes      -0.26780466  0.04281362  -6.255   0.0000000003971922 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 35962  on 25940  degrees of freedom
    ## Residual deviance: 32941  on 25932  degrees of freedom
    ## AIC: 32959
    ## 
    ## Number of Fisher Scoring iterations: 6

4.  Generate confusion matrices for each regression and attach
    probabilities to the validation set

- Using undersampled regression model

``` r
# obtain probability of positive class for each observation in validation set
lrprobsU <- predict(lrUnder, newdata = Validate, type = "response")

# obtain predicted class for each observation in validation set using threshold of 0.5
lrclassU <- as.factor(ifelse(lrprobsU > 0.5, "Yes","No"))

# output performance metrics using "Yes" as the positive class 
confusionMatrix(lrclassU, Validate$LoanDefault, positive = "Yes" )
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   No  Yes
    ##        No  3981   23
    ##        Yes 4600  114
    ##                                              
    ##                Accuracy : 0.4697             
    ##                  95% CI : (0.4592, 0.4803)   
    ##     No Information Rate : 0.9843             
    ##     P-Value [Acc > NIR] : 1                  
    ##                                              
    ##                   Kappa : 0.017              
    ##                                              
    ##  Mcnemar's Test P-Value : <0.0000000000000002
    ##                                              
    ##             Sensitivity : 0.83212            
    ##             Specificity : 0.46393            
    ##          Pos Pred Value : 0.02418            
    ##          Neg Pred Value : 0.99426            
    ##              Prevalence : 0.01571            
    ##          Detection Rate : 0.01308            
    ##    Detection Prevalence : 0.54072            
    ##       Balanced Accuracy : 0.64802            
    ##                                              
    ##        'Positive' Class : Yes                
    ## 

- Using oversampled regression model

``` r
# obtain probability of defaulting for each observation in validation set
lrprobsO <- predict(lrOver, newdata = Validate, type = "response")

#Attach probability scores to Validate dataframe
Validate <- cbind(Validate, Probabilities=lrprobsO)

# obtain predicted class for each observation in validation set using threshold of 0.5
lrclassO <- as.factor(ifelse(lrprobsO > 0.5, "Yes","No"))

#Attach predicted class to Validate dataframe
Validate <- cbind(Validate, PredClass=lrclassO)

#Create a confusion matrix using "Yes" as the positive class 
confusionMatrix(lrclassO, Validate$LoanDefault, positive = "Yes" )
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   No  Yes
    ##        No  4055   21
    ##        Yes 4526  116
    ##                                              
    ##                Accuracy : 0.4784             
    ##                  95% CI : (0.4679, 0.489)    
    ##     No Information Rate : 0.9843             
    ##     P-Value [Acc > NIR] : 1                  
    ##                                              
    ##                   Kappa : 0.0186             
    ##                                              
    ##  Mcnemar's Test P-Value : <0.0000000000000002
    ##                                              
    ##             Sensitivity : 0.84672            
    ##             Specificity : 0.47256            
    ##          Pos Pred Value : 0.02499            
    ##          Neg Pred Value : 0.99485            
    ##              Prevalence : 0.01571            
    ##          Detection Rate : 0.01331            
    ##    Detection Prevalence : 0.53246            
    ##       Balanced Accuracy : 0.65964            
    ##                                              
    ##        'Positive' Class : Yes                
    ## 

- Using ROSE regression model

``` r
# obtain probability of positive class for each observation in validation set
lrprobsR <- predict(lrrose, newdata = Validate, type = "response")

# obtain predicted class for each observation in validation set using threshold of 0.5
lrclassR <- as.factor(ifelse(lrprobsR > 0.5, "Yes","No"))

# output performance metrics using "Yes" as the positive class 
confusionMatrix(lrclassR, Validate$LoanDefault, positive = "Yes" )
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   No  Yes
    ##        No  4344   29
    ##        Yes 4237  108
    ##                                              
    ##                Accuracy : 0.5107             
    ##                  95% CI : (0.5001, 0.5212)   
    ##     No Information Rate : 0.9843             
    ##     P-Value [Acc > NIR] : 1                  
    ##                                              
    ##                   Kappa : 0.0183             
    ##                                              
    ##  Mcnemar's Test P-Value : <0.0000000000000002
    ##                                              
    ##             Sensitivity : 0.78832            
    ##             Specificity : 0.50623            
    ##          Pos Pred Value : 0.02486            
    ##          Neg Pred Value : 0.99337            
    ##              Prevalence : 0.01571            
    ##          Detection Rate : 0.01239            
    ##    Detection Prevalence : 0.49839            
    ##       Balanced Accuracy : 0.64728            
    ##                                              
    ##        'Positive' Class : Yes                
    ## 

The ROSE regression model has the highest accuracy out of the 3 models
tested.

5.  Plot an ROC curve and calculate area under the curve (AUC) for each
    model

- From the undersampled model

``` r
#create a prediction object to use for the ROC Curve
predROCU <- prediction(lrprobsU, Validate$LoanDefault)

#create a performance object to use for the ROC Curve
perfROCU <- performance(predROCU,"tpr", "fpr")

#plot the ROC Curve
plot(perfROCU)
abline(a=0, b= 1)
```

![unnamed-chunk-11-1](https://github.com/NikoSeino/Business-Analytics/assets/102825218/deb367c7-0ee4-4656-a351-bd644b222484)

``` r
# compute AUC 
performance(predROCU, measure="auc")@y.values[[1]]
```

    ## [1] 0.7089862

- From the oversampled model

``` r
#create a prediction object to use for the ROC Curve
predROC <- prediction(lrprobsO, Validate$LoanDefault)

#create a performance object to use for the ROC Curve
perfROC <- performance(predROC,"tpr", "fpr")

#plot the ROC Curve
plot(perfROC)
abline(a=0, b= 1)
```

![unnamed-chunk-12-1](https://github.com/NikoSeino/Business-Analytics/assets/102825218/d42a3d2c-4302-4821-b754-99fd21dce9df)

``` r
# compute AUC 
performance(predROC, measure="auc")@y.values[[1]]
```

    ## [1] 0.7100103

- From the ROSE model

``` r
#create a prediction object to use for the ROC Curve
predROCR <- prediction(lrprobsR, Validate$LoanDefault)

#create a performance object to use for the ROC Curve
perfROCR <- performance(predROCR,"tpr", "fpr")

#plot the ROC Curve
plot(perfROCR)
abline(a=0, b= 1)
```

![unnamed-chunk-13-1](https://github.com/NikoSeino/Business-Analytics/assets/102825218/f6010e87-495f-4a9d-ba0e-590611b5148f)

``` r
# compute AUC 
performance(predROCR, measure="auc")@y.values[[1]]
```

    ## [1] 0.6981542

The oversampled regression model actually has the highest AUC,
indicating it is the best model. So we will use this model to the test
set.

8.  Apply the best model to the test set

``` r
# obtain probability of defaulting for each observation in test set
lrprobstest <- predict(lrOver, newdata = Test, type = "response")

# obtain predicted class for each observation in test set using threshold of 0.5
lrclasstest <- as.factor(ifelse(lrprobstest > 0.5, "Yes","No"))

#Create a confusion matrix using "Yes" as the positive class 
confusionMatrix(lrclasstest, Test$LoanDefault, positive = "Yes" )
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   No  Yes
    ##        No  3970   33
    ##        Yes 4410  121
    ##                                              
    ##                Accuracy : 0.4794             
    ##                  95% CI : (0.4687, 0.49)     
    ##     No Information Rate : 0.982              
    ##     P-Value [Acc > NIR] : 1                  
    ##                                              
    ##                   Kappa : 0.0174             
    ##                                              
    ##  Mcnemar's Test P-Value : <0.0000000000000002
    ##                                              
    ##             Sensitivity : 0.78571            
    ##             Specificity : 0.47375            
    ##          Pos Pred Value : 0.02670            
    ##          Neg Pred Value : 0.99176            
    ##              Prevalence : 0.01805            
    ##          Detection Rate : 0.01418            
    ##    Detection Prevalence : 0.53094            
    ##       Balanced Accuracy : 0.62973            
    ##                                              
    ##        'Positive' Class : Yes                
    ## 

``` r
#Plot ROC Curve for model from oversampled training set using Test set

#create a prediction object to use for the ROC Curve
predROCtest <- prediction(lrprobstest, Test$LoanDefault)

#create a performance object to use for the ROC Curve
perfROCtest <- performance(predROCtest,"tpr", "fpr")

#plot the ROC Curve
plot(perfROCtest)
abline(a=0, b= 1)
```

![unnamed-chunk-14-1](https://github.com/NikoSeino/Business-Analytics/assets/102825218/fc1c484e-4031-446f-a3ac-c75f6bea3e76)

``` r
# compute AUC 
performance(predROCtest, measure="auc")@y.values[[1]]
```

    ## [1] 0.687949

7.  We can now use the best logistic regression model to predict the
    probability of defaulting for new customers.

Given a data set of new customers:

``` r
new_customers <- read.csv("OptivaNewData.csv")

#Convert categorical variables to factors with levels and labels
new_customers$Entrepreneur<-factor(new_customers$Entrepreneur,levels = c(0,1),labels = c("No","Yes"))
new_customers$Unemployed<-factor(new_customers$Unemployed,levels = c(0,1),labels = c("No","Yes"))
new_customers$Married<-factor(new_customers$Married,levels = c(0,1),labels = c("No","Yes"))
new_customers$Divorced<-factor(new_customers$Divorced,levels = c(0,1),labels = c("No","Yes"))
new_customers$HighSchool<-factor(new_customers$HighSchool,levels = c(0,1),labels = c("No","Yes"))
new_customers$College<-factor(new_customers$College,levels = c(0,1),labels = c("No","Yes"))
```

Make predictions for whether or not they will default:

``` r
lrprobsnew <- predict(lrOver, newdata = new_customers , type = "response")

#Attach probability scores to new_customers dataframe 
new_customers <- cbind(new_customers, Probabilities=lrprobsnew)
View(new_customers)
```
