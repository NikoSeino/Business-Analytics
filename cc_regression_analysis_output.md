Multiple Regression Analysis of Credit Card Data
================
Niko Seino
2023-03-13

### Business Question:

1.  What variables effectively contribute to predicting active
    cardholders’ credit card balances?
2.  What credit card balance might a new active cardholder hold
    depending on certain variables?

Install and load packages

Import and view data

``` r
carddf <- read_csv("Credit.csv")
```

    ## Rows: 310 Columns: 9
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## dbl (9): Income, Limit, Rating, Age, Education, Student, Gender, Married, Ba...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
View(carddf)
```

Check for any missing values

``` r
sum(is.na(carddf))
```

    ## [1] 0

Convert categorical variables to factors

``` r
carddf$Student <- factor(carddf$Student, levels = c(0, 1), labels = c("No","yes"))
carddf$Gender <- factor(carddf$Gender, levels = c(0, 1), labels = c("Male","Female"))
carddf$Married <- factor(carddf$Married, levels = c(0, 1), labels = c("No","yes"))
```

Generate summary statistics

``` r
summary(carddf)
```

    ##      Income           Limit           Rating           Age       
    ##  Min.   : 10354   Min.   : 1160   Min.   :126.0   Min.   :23.00  
    ##  1st Qu.: 23150   1st Qu.: 3976   1st Qu.:304.0   1st Qu.:42.00  
    ##  Median : 37141   Median : 5147   Median :380.0   Median :55.50  
    ##  Mean   : 49979   Mean   : 5485   Mean   :405.1   Mean   :55.61  
    ##  3rd Qu.: 63740   3rd Qu.: 6453   3rd Qu.:469.0   3rd Qu.:69.00  
    ##  Max.   :186634   Max.   :13913   Max.   :982.0   Max.   :98.00  
    ##    Education     Student      Gender    Married      Balance      
    ##  Min.   : 5.00   No :271   Male  :145   No :118   Min.   :   5.0  
    ##  1st Qu.:11.00   yes: 39   Female:165   yes:192   1st Qu.: 338.0  
    ##  Median :14.00                                    Median : 637.5  
    ##  Mean   :13.43                                    Mean   : 671.0  
    ##  3rd Qu.:16.00                                    3rd Qu.: 960.8  
    ##  Max.   :20.00                                    Max.   :1999.0

Partition the data into a training set and a validation set (Since it is
a small dataset, will divide 50-50)

``` r
set.seed(42)
sample <- sample(c(TRUE, FALSE), nrow(carddf), replace=TRUE, prob=c(0.5,0.5))
traincard  <- carddf[sample, ]
validatecard <- carddf[!sample, ]
```

Create a correlation matrix for the quantitative variables in the
training dataframe

``` r
cor(traincard[c(-6, -7, -8)])
```

    ##                 Income       Limit      Rating        Age    Education
    ## Income     1.000000000  0.86577459  0.86447853 0.26211804 -0.009954901
    ## Limit      0.865774593  1.00000000  0.99660300 0.17801382 -0.089089709
    ## Rating     0.864478533  0.99660300  1.00000000 0.18006901 -0.098724534
    ## Age        0.262118037  0.17801382  0.18006901 1.00000000  0.135112824
    ## Education -0.009954901 -0.08908971 -0.09872453 0.13511282  1.000000000
    ## Balance    0.476075747  0.79974843  0.80116447 0.03606945 -0.077228733
    ##               Balance
    ## Income     0.47607575
    ## Limit      0.79974843
    ## Rating     0.80116447
    ## Age        0.03606945
    ## Education -0.07722873
    ## Balance    1.00000000

Conduct multiple regression analysis using the training dataframe with
‘Balance’ as the outcome variable, and all others as predictor
variables. View the summary.

``` r
card_MR <- lm(Balance ~ Income + Limit + Rating + Age + Education + Student + Gender + Married, traincard)
summary(card_MR)
```

    ## 
    ## Call:
    ## lm(formula = Balance ~ Income + Limit + Rating + Age + Education + 
    ##     Student + Gender + Married, data = traincard)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -76.89 -21.05  -4.45  18.55  92.73 
    ## 
    ## Coefficients:
    ##                Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  -6.883e+02  1.931e+01 -35.642  < 2e-16 ***
    ## Income       -9.902e-03  1.372e-04 -72.156  < 2e-16 ***
    ## Limit         2.121e-01  1.562e-02  13.576  < 2e-16 ***
    ## Rating        1.686e+00  2.300e-01   7.333 1.57e-11 ***
    ## Age          -1.108e+00  1.532e-01  -7.234 2.68e-11 ***
    ## Education     6.766e-01  8.620e-01   0.785    0.434    
    ## Studentyes    4.879e+02  8.108e+00  60.171  < 2e-16 ***
    ## GenderFemale -5.082e+00  5.368e+00  -0.947    0.345    
    ## Marriedyes   -2.800e+00  5.674e+00  -0.494    0.622    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 32.37 on 142 degrees of freedom
    ## Multiple R-squared:  0.9938, Adjusted R-squared:  0.9935 
    ## F-statistic:  2860 on 8 and 142 DF,  p-value: < 2.2e-16

Calculate Variance Inflation Factor for each predictor variable to
assess multicollinearity

``` r
vif(card_MR)
```

    ##     Income      Limit     Rating        Age  Education    Student     Gender 
    ##   4.382514 156.423468 154.816559   1.110917   1.100389   1.088982   1.037292 
    ##    Married 
    ##   1.057214

Conduct a multiple regression analysis with ‘Balance’ as the outcome
variable and all other variables except ‘Limit’ as predictor variables

``` r
card_MR2 <- lm(Balance ~ Income + Rating + Age + Education + 
    Student + Gender + Married, data = traincard)
summary(card_MR2)
```

    ## 
    ## Call:
    ## lm(formula = Balance ~ Income + Rating + Age + Education + Student + 
    ##     Gender + Married, data = traincard)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -104.201  -33.308    0.435   32.766  128.697 
    ## 
    ## Coefficients:
    ##                Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  -7.859e+02  2.707e+01 -29.032  < 2e-16 ***
    ## Income       -9.677e-03  2.058e-04 -47.024  < 2e-16 ***
    ## Rating        4.766e+00  5.748e-02  82.915  < 2e-16 ***
    ## Age          -1.219e+00  2.311e-01  -5.276 4.79e-07 ***
    ## Education     2.110e+00  1.292e+00   1.633   0.1047    
    ## Studentyes    4.760e+02  1.218e+01  39.089  < 2e-16 ***
    ## GenderFemale -1.693e+00  8.100e+00  -0.209   0.8347    
    ## Marriedyes   -1.500e+01  8.463e+00  -1.773   0.0784 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 48.89 on 143 degrees of freedom
    ## Multiple R-squared:  0.9858, Adjusted R-squared:  0.9851 
    ## F-statistic:  1421 on 7 and 143 DF,  p-value: < 2.2e-16

Create a residual plot using these results

``` r
card_predict = predict(card_MR2)
card_resid = resid(card_MR2)
resid_df <- data.frame(card_predict, card_resid)

ggplot(resid_df, aes(x= card_predict, y = card_resid)) +
  geom_point() +
  labs(title = "Residual Plot", x = "Predicted Values", y = "Residuals")
```

![](credit_card_regression_analysis_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->
\*Residual plot points are evenly distributed, indicating they are
normally distributed and have constant variance

Create a probability plot

``` r
card_stdres <- rstandard(card_MR2)    #get standardized residuals
qqnorm(card_stdres, ylab = "Standardized residuals", xlab = "Normal scores")
```

![](credit_card_regression_analysis_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

\*q-q plot points are clustered along a straight line, indicating the
residuals are normally distributed

Create a new regression analysis using the training dataframe with
‘Balance’ as the outcome variable and only the statistically significant
variables as predictors

``` r
cardSS_MR <- lm(Balance ~ Income + Rating + Age + Student, traincard)
summary(cardSS_MR)
```

    ## 
    ## Call:
    ## lm(formula = Balance ~ Income + Rating + Age + Student, data = traincard)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -108.23  -35.78    3.47   35.28  115.49 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) -7.677e+02  2.006e+01 -38.278  < 2e-16 ***
    ## Income      -9.665e-03  2.059e-04 -46.938  < 2e-16 ***
    ## Rating       4.754e+00  5.741e-02  82.819  < 2e-16 ***
    ## Age         -1.164e+00  2.315e-01  -5.028 1.43e-06 ***
    ## Studentyes   4.794e+02  1.214e+01  39.506  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 49.34 on 146 degrees of freedom
    ## Multiple R-squared:  0.9853, Adjusted R-squared:  0.9849 
    ## F-statistic:  2440 on 4 and 146 DF,  p-value: < 2.2e-16

Run a standardized regression with standardized variables to compare
which make the strongest unique contribution to predicting credit card
balance

``` r
lm.beta(cardSS_MR)
```

    ## 
    ## Call:
    ## lm(formula = Balance ~ Income + Rating + Age + Student, data = traincard)
    ## 
    ## Standardized Coefficients::
    ## (Intercept)      Income      Rating         Age  Studentyes 
    ##          NA -0.97173221  1.69521016 -0.05277281  0.40666516

Conduct a final multiple regression analysis with the validation
dataframe with ‘Balance’ as the outcome variable and statistically
significant variables as the predictors

``` r
validcard_df <- lm(Balance ~ Income + Rating + Age + Student, validatecard)
summary(validcard_df)
```

    ## 
    ## Call:
    ## lm(formula = Balance ~ Income + Rating + Age + Student, data = validatecard)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -128.345  -30.438   -0.038   35.772  124.780 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) -7.875e+02  2.271e+01 -34.670  < 2e-16 ***
    ## Income      -9.805e-03  2.098e-04 -46.741  < 2e-16 ***
    ## Rating       4.806e+00  5.558e-02  86.481  < 2e-16 ***
    ## Age         -1.061e+00  2.818e-01  -3.767 0.000235 ***
    ## Studentyes   4.844e+02  1.406e+01  34.453  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 55.88 on 154 degrees of freedom
    ## Multiple R-squared:  0.9833, Adjusted R-squared:  0.9829 
    ## F-statistic:  2267 on 4 and 154 DF,  p-value: < 2.2e-16

Predict the credit card balances of the new cardholders in
credit_card_prediction.csv with 95% prediction intervals

``` r
newcard <- read_csv("credit_card_prediction.csv")
```

    ## Rows: 3 Columns: 4
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## dbl (4): Income, Rating, Age, Student
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
newcard$Student <- factor(newcard$Student, levels = c(0, 1), labels = c("No","yes"))

predict(validcard_df, newcard, interval = "prediction", level = .95)
```

    ##         fit       lwr       upr
    ## 1  380.8456  266.3149  495.3763
    ## 2 1510.8839 1395.8143 1625.9534
    ## 3 1481.4392 1369.1817 1593.6968
