########################
# In this script, we intend to analyse the well known spam dataset from the HP labs.
# We will try to predict whether a mail should be classified as spam based with 
# the help of various decision tree libraries and a support vector machine.
# The models will be compared based on their performance with a test data set, notably
# accuracy, specificity and sensitivity.
# Note: For an introduction into this topic, I recommend the "Introduction to Statistical Learning"
# book by James et al. (http://www-bcf.usc.edu/~gareth/ISL/)
########################


########################
# Variable describtion (taken from https://archive.ics.uci.edu/ml/datasets/Spambase)
# The dataset consists of 58 variables and 4601 observations:
# Variables 1 to 48: "word_freq_WORD" -  percentage of words in the e-mail that are equal to WORD
# Variables 49 to 54: "char_freq_CHARACTER" -  percentage of characters in the mail equal to CHARACTER. Note that 
# the character are "spelled out" because the use of symbols in column names causes trouble.
# capital_run_length_average: average length of uninterrupted sequences of capital letters 
# capital_run_length_longest: length of longest uninterrupted sequence of capital letters 
# capital_run_length_total: total number of capital letters in the e-mail 
# Target variable: "Spam" - indicates whether a given mail is spam or not
########################

library(RCurl)
library(tree) 
library(randomForest)
library(adabag)
library(e1071)


# Load the data
Data <- getURL("https://raw.githubusercontent.com/FM2805/SpamIdentification_MachineLearning/Development/Spambase_HP.csv")
Data <- read.csv(text=Data)
Data$X <- NULL

# Set seed for reproductible results
set.seed(12345)
# Shuffle DF rows
Data <-Data[sample(nrow(Data)),]
rownames(Data) <- NULL

# Get overview of distribution
table(Data$Spam)
# 1813 spam mails are in the dataset (39%), whereas 2788 (61%) are classified as non-spam.



########################
# The first step of our analysis is the implementation of a simple decision tree.
# Initially, we of course split our data into a test and a training set for validation purposes.
# For the test data, we will compute the accuracy, specificity and sensitivity
########################

# We start by randomly dividing the data into a test (20%) and training data set (80% - 3681 observations)
train_Index <-sample(1:4601,3681,replace=FALSE)
Data$Indicator <- ifelse(rownames(Data) %in% as.character(train_Index),"TRAIN","TEST")
Data$Spam <- as.factor(Data$Spam)
Data_Test <- subset(Data,Indicator=="TEST")
Data_Train <- subset(Data,Indicator=="TRAIN")
Data_TestIndic <- Data_Test$Spam

# Get the model formula (Spam as a function of all other variables)
rhs <-paste(colnames(Data)[1:(length(Data)-2)],collapse="+")
lhs <- 'Spam'
Equation <- as.formula(paste(lhs,rhs,sep="~"))


# Apply a "simple" classification tree on the training data
First_Tree <-tree(Equation,data=Data_Train) 
summary(First_Tree)
# The algorithm constructed a tree with 12 terminal nodes and kept only 7 variables

# Apply to the test data
Pred_Tree <- predict(First_Tree, Data_Test, type="class")
table(Pred_Tree,Data_TestIndic)
# Accuracy (515+316/(515+316+42+47): 90%
# Specificity 515/(515+42): 92%
# Sensitivity 316/(316+47): 87%

# Apply a cross validation algorithm to "prune" the tree
# It will give us the "best" number of terminal nodes for all considered trees ("size")
Tree_CV <-cv.tree(First_Tree)
Tree_CV
# We see that our tree with 12 nodes performs actually best. 
# To demonstrate this, lets take a tree with fewer nodes (5)
First_Tree_opt <-prune.tree(First_Tree ,best=5)
Pred_Tree_opt <- predict(First_Tree_opt, Data_Test, type="class")
table(Pred_Tree_opt,Data_TestIndic)
# We immediately see that our performance worsened

# Lets plot our tree
plot(First_Tree)
text(First_Tree)
# We notice that the number of exclamation marks is identified as the primiary indicator for 
# determining if a mail is classified as spam or not. For instance, a mail with less than 7.9% of
# characters are "!" and then roughly more than 5% of the words are "remove" will be classified as spam.

########################
# Now we implement a decision tree with the "bagging" methodology (which is akin to bootstrapping).
# For the test data, we will again compute the accuracy, specificity and sensitivity
########################

# Estimate the model with 200 trees
Second_Tree <-bagging(Equation,data=Data_Train,mfinal=200) 
# This already takes a while because 200 trees are used 

# Lets take a look at the relative importance of the variables
sort(Second_Tree$importance)
# We see that the number of exclamation marks appears to be by
# far the most influential factors

# Apply to the test data
Pred_Tree <- predict(Second_Tree, Data_Test, type="class")
# Note: The confusion matrix may also be accessed via Pred_Tree$confusion in this case
Pred <-Pred_Tree$class
table(Pred,Data_TestIndic)
# Accuracy (523+311)/(523+311+34+52): 91%
# Specificity: 523/(523+34) = 94%
# Sensitivity: 311/(311+52) = 86%
# Our numbers actually do improve a bit, with the exception of the sensitivity measure.


########################
# Lets estimate a random forest model.
# For the test data, we will again compute the accuracy, specificity and sensitivity
########################

# Initially we want to set the best number of variables for the splitting of our tree (the 'mtry' in randomForest())
bestmtry <- tuneRF(Data_Train[,1:57],Data_Train$Spam)
# It gives us the number 7

# Lets estimate the tree (with ntree=200, as in the example above)
Third_Tree <-randomForest(Equation,data=Data_Train, ntree=200,mtry=7,importance=TRUE)
# Once again exclamation marks and dollar symbols seem to be crucial

# Apply to the test data
Pred_Tree <- predict(Third_Tree, Data_Test, type="class")
table(Pred_Tree,Data_TestIndic)
# Accuracy (545+338)/(545+338+12+25): 96%
# Specificity: 545/(545+12) = 98%
# Sensitivity: 338/(338+25) = 93%
# With the randomForest algorithm, we achieve by far our best result.

# Lets plot out result: investigate the 10 most important variables
varImpPlot(Third_Tree, n.var=10,main="Random Forest Model")
# The plot on the left hand side(Mean Decrease Accuracy) shows how much the accuracy
# of our model decreases when a variable is removed. For instance, of the indicator 
# for the exclamation marks is removed, this results in (on average) 32 additional false classifications.
# The Mean Decrease GINI plot roughly indicates how important a predictor is when splitting the data.

# Overall, the most important indicator to identify a spam mail is the numbers of exclamation marks.


########################
# Finally, we implement a support vecorm achine model and compare it to the tree based approaches.
# For the test data, we will once again compute the accuracy, specificity and sensitivity
########################


#Basic SVM
SVM_Basis <- svm(Equation, data=Data_Train,kernel="radial")
summary(SVM_Basis)
Pred_SVM <- predict(SVM_Basis,newdata=Data_Test,type="class")
table(Pred_SVM,Data_TestIndic)
# Accuracy (533+328)/(533+328+24+35): 94%
# Specificity: 533/(533+24) = 96%
# Sensitivity: 328/(328+35) = 90%
# The model does a slightly worse job than the random forest

# Let's tune the SVM for optimal parameters (gamma and cost) in a given range
# Note: This is computationally very intensive
# svm_tune <- tune.svm(Equation,data=Data_Train, gamma = 10^(-4:-1), cost = 10^(-1:1))
# summary(svm_tune)
# Result: gamma: 0.01, cost: 10

# Estimate the model with the chosen parameters
SVM_Optim <- svm(Equation, data=Data_Train,kernel="radial",gamma= 0.01 ,cost=10)
Pred_SVM <- predict(SVM_Optim,newdata=Data_Test,type="class")
table(Pred_SVM,Data_TestIndic)
# Accuracy (532+333)/(532+333+25+30): 94%
# Specificity: 532/(532+25) = 96%
# Sensitivity: 333/(333+30) = 92%
# The results improve marginally

########################
# To conclude: the random forest model outperforms all other implemented approaches rather substantially.
# Beware, however, that there are weaknesses associated with this method (such as overfitting).
# With respect to the data, we found the the number of exclamation marks in a mail is by far the most
# important predictor when it comes to classify a mail (spam/no spam) in this data set
########################


