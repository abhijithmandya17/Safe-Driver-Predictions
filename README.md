# SYS 6018 Competition 4 team 2
NOTE: Our name on Kaggle is "Competition 4 team 2" - we let Professor Gerber know and he told us this was ok

Tyler Lewris - tal3fj
Abhijith Mandya - am6ku



# Best Kaggle Scores
Linear Model (Parametric Approach): 0.166

RandomForest (Non-Parametric Approach): 

# Team Roles
Tyler: Data cleaning and Transformations, parametric approach

Abhijith: Data cleaning and Transformations, non-parametric approach

James: "open track", reflection questions

# Data Cleaning
This data set was previously engineered  by Kaggle before opening it up for the competition. All, unknown or missing values were codified as -1. This helped us asses missing values easily and move forward with very few hiccups. Train and test data were first merged to perform all cleaning and transformations in a unified fashion. We dropped the columns which had more than 50% unknown values. Further, we correctly classified the categorical and numerical columns so that we could impute the mode and mean respectively for them. We then reset the levels to reflect the true distribution within the columns. For the Random forest, we took a few more steps where we reversed the one hot encoded columns to reduce the dimensionality of the data set and dropped a single column which had over 100 levels.   

# Linear Method
After extensive and thorough data cleaning and exploration, I started the parametric approach by subsetting the data. This was an important step as the full data set has nearly 600,000 observations and 59 variables - far more computationally demanding than my computer can handle. After subsetting the data I ran a linear model with all variables to get a base level model. Then, I utilized R's vif function to identify and remove any insignficant variables that were amplifying the noise of my model. Next, I analyzed plots of variables to try and understand interactions among variables. At this point, after removing five variables that clearly should not belong in my predictive models, I ran a logisitic regression on the entire dataset with the remaining variables. I then chose only the variables that had p-values < 0.05 and created a new model. From there, I further analyzed this new model and removed two variables that had many factor levels that were not signficant at the 0.05 level. 

At this point, I have identified a list of variables that are all significant at the 0.05 level. I then performed stepwise regression using R's stepAIC function on the same models (using the data subset - stepwise is computationally expensive and unrealistic to do on the full dataset) and evaluated three different models based on their relative AIC score. I then compared each of these models using anova tables, summary functions, plots, implemented an ROC Curve to visualize each model and their predictive power, and then performed K-fold cross validation. I performed 3 different K-fold cross validations setting K = 3, 5, and 10. For the purpose of this assignemtn, K=5 made the most sense both computationally and logically. After performing K-fold cross validation on all three models using the full dataset I chose the one that returned the lowest train MSE. Knowing that the predicted output is a probability, I used K-fold cross validation on the final model to identify the optimal threshold level to maximize accuracy(i.e. at what level should these probabilities be assigned a 0 or 1). For my predictions, a threshold level of 0.036 was selected. I then used an unnormalized.gini.index function to determine what type of score I would receive on Kaggle using the training data set. Finally, I wrote my predictions to a csv in the correct format so as to be accepted by Kaggle for a score. 


# Grading Criteria
Data exploration

Data cleaning (missing values, outliers, etc.)

Rationale for the selected statistical modeling methods

Correct implementation and use of statistical modeling methods

Appropriate model selection approach (train/test, cross-validation, etc.)

Thoroughly documented code (R comments)

Appropriate use of functions to reduce code complexity and redundancy

Clear thinking about the reflection questions

# Reflection Questions: 
Who might care about this problem and why?

Why might this problem be challenging?

What other problems resemble this problem?
