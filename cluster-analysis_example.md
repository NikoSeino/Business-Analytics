Cluster Analysis
================
Niko Seino
2023-05-10

**Business Question:**

What is the best way to group magazine subscribers by key
characteristics?

1.  Install packages and load data set

``` r
rm(list = ls())

library("janitor")
library("tidyverse")
library("fpc")
library("cluster")

magdf <- read.csv("young_professional_magazine.csv")
```

2.  View and Prepare the data for analysis

``` r
# View(magdf)

#count observations in the dataset
summarize(magdf, n())
```

    ##   n()
    ## 1 410

``` r
#create histograms for each continuous variables
options(scipen=+500000) #tells program not to use scientific notation unless over 500000

ggplot(magdf) +
  geom_histogram(aes(x=Age), binwidth = 1)
```
![](cluster-analysis_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->
``` r
ggplot(magdf) +
  geom_histogram(aes(x=Value.of.Investments), binwidth = 5000)
```
![](cluster-analysis_files/figure-gfm/unnamed-chunk-2-2.png)<!-- -->
``` r
ggplot(magdf) +
  geom_histogram(aes(x=Number.of.Transactions), binwidth = 1)
```
![](cluster-analysis_files/figure-gfm/unnamed-chunk-2-3.png)<!-- -->
``` r
ggplot(magdf) +
  geom_histogram(aes(x=Household.Income), binwidth = 5000)
```
![](cluster-analysis_files/figure-gfm/unnamed-chunk-2-4.png)<!-- -->
``` r
#add labels to binary variables
magdf$Female <- factor(magdf$Female, levels = c(0, 1), labels = c("No", "Yes"))
magdf$Real.Estate.Purchases <- factor(magdf$Real.Estate.Purchases, levels = c(0, 1), labels = c("No", "Yes"))
magdf$Graduate.Degree <- factor(magdf$Graduate.Degree, levels = c(0, 1), labels = c("No", "Yes"))
magdf$Have.Children <- factor(magdf$Have.Children, levels = c(0, 1), labels = c("No", "Yes"))
```

``` r
#create new dataframe with only quantitative variables
quantdf <- magdf[c(-2, -3, -6, -8)]
# View(quantdf)

#normalize each variable (find z-scores)
quantdfn <- scale(quantdf)
head(quantdfn)
```

    ##              Age Value.of.Investments Number.of.Transactions Household.Income
    ## [1,]  1.96017891           -1.0333608           -0.636327424       0.02126726
    ## [2,] -0.02788133           -1.0207112           -0.636327424      -0.11946370
    ## [3,]  2.70570151           -0.1099432           -0.313837629      -0.75418902
    ## [4,] -0.52489639           -0.5653272            0.008652165       0.59855137
    ## [5,]  0.22062620           -0.8499422           -0.313837629      -0.03330189
    ## [6,]  0.46913373            0.7059532           -0.958817218       1.40560031

**K-means cluster analysis**

3.  Prepare and create an elbow plot to determine number of clusters

``` r
#set random seed in order to replicate the analysis
set.seed(42)

#create function to calculate total within-cluster sum of squared deviations to use in elbow plot
wss <- function(k){kmeans(quantdfn, k, nstart = 10)} $tot.withinss

#specify range of k values
k_values <- 1:10

#run the function to create range of values for the elbow plot
wss_values <- map_dbl(k_values, wss)

#create a new dataframe containing both k values and wss
elbowdf <- data.frame(k_values, wss_values)
#View(elbowdf)
```

Graph the elbow plot

``` r
ggplot(elbowdf, aes(x=k_values, y=wss_values)) +
  geom_line() +
  geom_point()
```

![unnamed-chunk-6-1](https://github.com/NikoSeino/Business-Analytics/assets/102825218/6e29d687-0a6b-4bbc-9879-b16d7a417f9f)

4.  Run analysis with appropriate number of clusters. Based on the plot,
    we will use k=4 for number of clusters.

``` r
#run k-means clustering with 4 clusters and 1000 random restarts
k4 <- kmeans(quantdfn, 4, nstart=1000)

#display the structure k4
str(k4)
```

    ## List of 9
    ##  $ cluster     : int [1:410] 3 3 3 4 3 2 3 4 4 3 ...
    ##  $ centers     : num [1:4, 1:4] 0.3942 0.0261 0.5955 -1.0337 1.5746 ...
    ##   ..- attr(*, "dimnames")=List of 2
    ##   .. ..$ : chr [1:4] "1" "2" "3" "4"
    ##   .. ..$ : chr [1:4] "Age" "Value.of.Investments" "Number.of.Transactions" "Household.Income"
    ##  $ totss       : num 1636
    ##  $ withinss    : num [1:4] 282 143 292 240
    ##  $ tot.withinss: num 956
    ##  $ betweenss   : num 680
    ##  $ size        : int [1:4] 63 46 175 126
    ##  $ iter        : int 3
    ##  $ ifault      : int 0
    ##  - attr(*, "class")= chr "kmeans"

``` r
#look at cluster statistics
cluster.stats(dist(quantdfn, method = "euclidean"), k4$cluster)
```

    ## $n
    ## [1] 410
    ## 
    ## $cluster.number
    ## [1] 4
    ## 
    ## $cluster.size
    ## [1]  63  46 175 126
    ## 
    ## $min.cluster.size
    ## [1] 46
    ## 
    ## $noisen
    ## [1] 0
    ## 
    ## $diameter
    ## [1] 8.731119 6.459981 5.047733 4.948840
    ## 
    ## $average.distance
    ## [1] 2.712944 2.289854 1.702407 1.821043
    ## 
    ## $median.distance
    ## [1] 2.503044 2.096996 1.633555 1.748860
    ## 
    ## $separation
    ## [1] 0.3072036 0.4262397 0.3094109 0.3072036
    ## 
    ## $average.toother
    ## [1] 3.260644 3.249676 2.735785 2.717608
    ## 
    ## $separation.matrix
    ##           [,1]      [,2]      [,3]      [,4]
    ## [1,] 0.0000000 0.4262397 0.4010144 0.3072036
    ## [2,] 0.4262397 0.0000000 0.4721009 0.6744707
    ## [3,] 0.4010144 0.4721009 0.0000000 0.3094109
    ## [4,] 0.3072036 0.6744707 0.3094109 0.0000000
    ## 
    ## $ave.between.matrix
    ##          [,1]     [,2]     [,3]     [,4]
    ## [1,] 0.000000 3.717902 3.154172 3.241587
    ## [2,] 3.717902 0.000000 3.096728 3.227990
    ## [3,] 3.154172 3.096728 0.000000 2.394819
    ## [4,] 3.241587 3.227990 2.394819 0.000000
    ## 
    ## $average.between
    ## [1] 2.903973
    ## 
    ## $average.within
    ## [1] 1.960052
    ## 
    ## $n.between
    ## [1] 57757
    ## 
    ## $n.within
    ## [1] 26088
    ## 
    ## $max.diameter
    ## [1] 8.731119
    ## 
    ## $min.separation
    ## [1] 0.3072036
    ## 
    ## $within.cluster.ss
    ## [1] 956.4456
    ## 
    ## $clus.avg.silwidths
    ##          1          2          3          4 
    ## 0.07470313 0.18916895 0.26707357 0.21641395 
    ## 
    ## $avg.silwidth
    ## [1] 0.2132051
    ## 
    ## $g2
    ## NULL
    ## 
    ## $g3
    ## NULL
    ## 
    ## $pearsongamma
    ## [1] 0.4197272
    ## 
    ## $dunn
    ## [1] 0.0351849
    ## 
    ## $dunn2
    ## [1] 0.882738
    ## 
    ## $entropy
    ## [1] 1.25922
    ## 
    ## $wb.ratio
    ## [1] 0.6749552
    ## 
    ## $ch
    ## [1] 96.15431
    ## 
    ## $cwidegap
    ## [1] 3.413033 3.493278 1.616582 1.376527
    ## 
    ## $widestgap
    ## [1] 3.493278
    ## 
    ## $sindex
    ## [1] 0.4509849
    ## 
    ## $corrected.rand
    ## NULL
    ## 
    ## $vi
    ## NULL

``` r
#look at ratios of between cluster average ($ave.between.matrix) and within cluster ave ($average.distance)

#ratio for cluster distances is >1, indicating reasonable cluster number

#combine each observation's cluster assignment with unscaled data frame
quantdfk4 <- cbind(quantdf, clusterID = k4$cluster)

#View(quantdfk4)

#calculate variable averages for all non-normalized observations
summarize_all(quantdf, mean)
```

    ##       Age Value.of.Investments Number.of.Transactions Household.Income
    ## 1 30.1122             28538.29               5.973171         74459.51

``` r
#calculate variable averages for each cluster and compare
quantdfk4 %>%
  group_by(clusterID) %>%
  summarize_all(mean)
```

    ## # A tibble: 4 × 5
    ##   clusterID   Age Value.of.Investments Number.of.Transactions Household.Income
    ##       <int> <dbl>                <dbl>                  <dbl>            <dbl>
    ## 1         1  31.7               53433.                   8.27           72208.
    ## 2         2  30.2               28417.                   5.39          146491.
    ## 3         3  32.5               21406.                   5.05           63217.
    ## 4         4  26.0               26041.                   6.32           64903.

5.  Make insights based on the cluster analysis results

By looking at the averages for each variable in each of the four
clusters, we can see that customers have been grouped into categories
based on certain characteristics:

- Cluster 1 - The “Investors”: These customers have large investments,
  and average income
- Cluster 2 - The “financial conservatives”: These customers have
  average investments despite having higher income
- Cluster 3 - The “less resourced: These customers have less income and
  lower investments
- Cluster 4 - The “up-and-comers”: These are younger subscribers that
  are investing as much as the older ones, despite having less income
