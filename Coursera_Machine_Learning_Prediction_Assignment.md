Loading packages and data
=========================

Loading the required packages

``` r
library(caret)
library(randomForest)
library(rpart)
library(rattle)
library(knitr)
library(rpart.plot)
```

load data, remove columns, set seed

``` r
trainingData <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!", ""))
testingData <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!", ""))


# remove all the columns that are empty
trainingData <- trainingData[, colSums(is.na(trainingData)) == 0]
testingData <- testingData[, colSums(is.na(testingData)) == 0]

# remove the first columns that can't be consired as predictors (name, timestamps, etc...) 
trainingData <- trainingData[, -c(1:7)]
testingData <- testingData[, -c(1:7)]

#random number to seed for later reproducibility
set.seed(1806)
```

Cross-Validation: In order to allow cross-validation, it is necessary to partion the training set into two different datasets

``` r
traningPartitionData <- createDataPartition(trainingData$classe,  p = 0.7, list = F)
trainingDataSet <- trainingData[traningPartitionData, ]
testingDataSet <- trainingData[-traningPartitionData, ]
```

Prediction model 1 - decision tree
==================================

``` r
decisionTreeModel <- rpart(classe ~ ., data = trainingDataSet, method = "class")
decisionTreePrediction <- predict(decisionTreeModel, testingDataSet, type = "class")
```

Plot Decision Tree

``` r
rpart.plot(decisionTreeModel, main = "Decision Tree", under = T, faclen = 0)
```

![](README_figs/README-unnamed-chunk-6-1.png)

Using confusion matrix to test results, print accuracy value

``` r
cmtree <- confusionMatrix(decisionTreePrediction, testingDataSet$classe)
overall <- cmtree$overall
overall.accuracy <- overall['Accuracy'] 
overall.accuracy 
```

    ##  Accuracy 
    ## 0.7497026

Prediction model 2 - random forest
==================================

``` r
randomForestModel <- randomForest(classe ~. , data = trainingDataSet, method = "class")
randomForestPrediction <- predict(randomForestModel, testingDataSet, type = "class")
```

``` r
cmforest <- confusionMatrix(randomForestPrediction, testingDataSet$classe)
overall <- cmforest$overall
overall.accuracyforest <- overall['Accuracy'] 
overall.accuracyforest 
```

    ## Accuracy 
    ## 0.995582

Final Model Definition
======================

Prediction model 2 - random forest

``` r
predictionFinal <- predict(randomForestModel, testingDataSet, type = "class")
```

The training dataset was predicted by two Machine Learning algorithims (Decision Tree and Random Forest). Based on the acurracy of each model, the best option is to keep the Random Forest choice since the accuracy of the random forest predication model is higher than the accuracy of the decision tree model.
