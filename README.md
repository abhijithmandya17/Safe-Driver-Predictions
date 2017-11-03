# SYS 6018 Competition 4 team 2
NOTE: Our name on Kaggle is "Competition 4 team 2" - we let Professor Gerber know and he told us this was ok

Tyler Lewris - tal3fj
Abhijith Mandya - am6ku



# Best Kaggle Scores
Linear Model (Parametric Approach): 0.166

RandomForest (Non-Parametric Approach): 0.245

Gradient Boost (Open Track): 0.245

# Team Roles
Tyler: Data cleaning and Transformations, parametric approach

Abhijith: Data cleaning and Transformations, non-parametric approach, open track, Reflection questions

James: "open track", Reflection questions

# Data Cleaning and Exploration
This data set was previously engineered  by Kaggle before opening it up for the competition. All, unknown or missing values were codified as -1. This helped us asses missing values easily and move forward with very few hiccups. Train and test data were first merged to perform all cleaning and transformations in a unified fashion. We dropped the columns which had more than 50% unknown values. Further, we correctly classified the categorical and numerical columns so that we could impute the mode and mean respectively for them. We then reset the levels to reflect the true distribution within the columns. For the Random forest, we took a few more steps where we reversed the one hot encoded columns to reduce the dimensionality of the data set and dropped a single column which had over 100 levels.   
# Linear Method
After extensive and thorough data cleaning and exploration, I started the parametric approach by subsetting the data. This was an important step as the full data set has nearly 600,000 observations and 59 variables. I ran a linear model with all variables to get a base level model. Then, I utilized vif function to identify and remove any variables that were colinear. Next, I analyzed plots of variables to try and understand interactions among variables. At this point, after removing five variables that clearly should not belong in my predictive models, I ran a logisitic regression on the entire dataset with the remaining variables. I then chose only the variables that had p-values < 0.05 and created a new model. From there, I further analyzed this new model and removed two variables that had many factor levels that were not signficant at the 0.05 level. 

At this point, I have identified a list of variables that are all significant at the 0.05 level. I then performed stepwise regression using R's stepAIC function on the same models (using the data subset - stepwise is computationally expensive and unrealistic to do on the full dataset) and evaluated three different models based on their relative AIC score. I then compared each of these models using anova tables, summary functions, plots, implemented an ROC Curve to visualize each model and their predictive power, and then performed K-fold cross validation. I performed 3 different K-fold cross validations setting K = 3, 5, and 10. For the purpose of this assignemtn, K=5 made the most sense both computationally and logically. After performing K-fold cross validation on all three models using the full dataset I chose the one that returned the lowest train MSE. Knowing that the predicted output is a probability, I used K-fold cross validation on the final model to identify the optimal threshold level to maximize accuracy(i.e. at what level should these probabilities be assigned a 0 or 1). For my predictions, a threshold level of 0.036 was selected. I then used an normalized.gini.index function to determine what type of score I would receive on Kaggle using the training data set. Finally, I wrote my predictions to a csv in the correct format so as to be accepted by Kaggle for a score. 

# Non Linear Method (Random Forest)
The choice of random forest was mandatory for grading in this competition. After specific data cleaning for random forest mentioned above, a subset of data was chosen (n = 100,000) from the train data set to tune the hyper parameters of the model like m, number of trees and sampling size. To cross validate these values, we chose to parallelize our runs using the foreach package that let us run multiple forests at the same time. Based on the outcomes, we decided to use m = 3 and 1000 trees. Another unique situation with this dataset was imbalance in the response variable. One class dominated the other roughly in the proportion of 24:1. Random forest classifiers do really poorly with such imbalance since predictions are based on majority vote. To combat this, we downsampled our data to balance it. We kept all the minority class observations (n = 3615, 21694 in the subset and train data respectively) while randomly selecting an equal amount of majority class observations. This model was then validated on held out train data using the Gini function producing a highly overextimated score of 0.394. This model was then used to run on the whole train data and make predictions generating a kaggle score of 0.245.  

# Open Track (Gradient Boost)
Gradient Boost was attempted using the caret package on cleaned data. The caret package allows the model to be cross validated internally. We chose a value of 5 to save computational memory. Model was first trained on a subset of 100,000 downsampled observations and then finally on the complete downsampled data set. Normalized Gini scores predicted 0.289 while Kaggle gave out a score of 0.245, same as the random forest. This suggests that for any further improvement, further feature engineering and or feature selection is necessary.   

# Reflection Questions: 
Who might care about this problem and why?

Large transportation companies like Uber and Lyft benefit directly from solving such problems. Predicting safety scores of their drivers can in turn help them remodel their pricing and compensations to the drivers. Furthermore, they can implement training systems for those with lower scores to improve thier services. 

Inaccuracies in a car insurance company’s claim predictions most directly concerns drivers. Charging higher premiums to drivers with lower safety scores can improve the return on investment for insurance companies. Thus, from a business standpoint, it is in the best interest for such types of companies to provide the most accurate and reliable claim predictions.

Why might this problem be challenging?

Safety scores can depend on a multitude of predictors, most of which cannot be collected or codified for use in statistical learning. Data collection, as seen in this data, is extremely cumbersome and a huge percentage of the data was missing, rendering those predictors useless. Dealing with big data, in general, can be very time consuming. To add to the complications, our data includes mixed data types which tend to increase the model complexity. Furthermore, safety usually isn't linearly dependent on a set of variables; thus we end up using high end learning algorithms to improve predictions reducing interpretibility. 

What other problems resemble this problem?

Credit score predictions, human behavior predictions like brand preference, ad click through are all similar examples which rely on predicting human behavior in a multitude of ways.
