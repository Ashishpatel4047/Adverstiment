Project - Marketing
Infer relationship between sales and the three media budgets: TV, Radio and Newspaper.

Project Steps Followed
Define Project Goals/Objective
Data Retrieval
Data Cleansing
Exploratory Data Analysis
Data Modeling
Result Analysis
Project Objective
Suppose you have been assigned as Data Scientist to advice on how to improve the sales of a particular product of a company.

The company provided you with sales data from  different markets.

The data also contains the advertising budgets for the product in each of those markets for three different media: TV, radio, and newspaper.

The client cannot directly increase the sales of the product.

But they can adjust the advertisement budget for each of the three media. As a data scientist, if you can establish the relationship between advertisement expenditure and sales, you can provide your feedback on how to adjust the budgets so that sales can increase.

So, the objective is to develop a model that you can use to predict the sales on the basis of the three media budgets.



Define Research Goals
Infer relationship between sales and three media budgets: TV, Radio, and Newspaper

Data Set
The Data set can be downloaded from the link

The dataset contains sales data from 200 markets

By the end of the project, the learners will be able to learn the approaches required for Multiple Linear Regression


Import the Libraries
In [1]: import pandas as pd

In [2]: import numpy as np                      
In [3]: import matplotlib.pyplot as plt
In [4]: import seaborn as sns   
Load the data



The most important step of model development is understanding the dataset. Generally, we follow the following steps to understand the data:

View the raw data
Dimensions of the dataset
Data Types of the attributes
Presence of Null Values in the dataset
Statistical Analysis
Data Errors (zero values)
View Raw Data
In [6]: advertising.head()            
Out[6]: 
      TV  Radio  Newspaper  Sales
0  230.1   37.8       69.2   22.1
1   44.5   39.3       45.1   10.4
2   17.2   45.9       69.3    9.3
3  151.5   41.3       58.5   18.5
4  180.8   10.8       58.4   12.9
Dimensions of the dataset
In [7]: advertising.shape
Out[7]: (200, 4)
We get the dimension of the dataset. The dataset has 200 rows and 4 columns.

Data Type
In [8]: advertising.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200 entries, 0 to 199
Data columns (total 4 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   TV         200 non-null    float64
 1   Radio      200 non-null    float64
 2   Newspaper  200 non-null    float64
 3   Sales      200 non-null    float64
dtypes: float64(4)
memory usage: 6.4 KB
Observations
Our observations are as follows

NaN values do not present in the data set. Because of the Non-Null Count and number of rows in the dataset match.

There are 3 Input Variables and 1 Output Variable (Sales)

The data type of all the input variables is float64. The data type of out variable (Sales) is float64.

Shows that all the input as well as output variables are continuous (quantitative) data types.

None of the columns contain the Null Values

Null Values
In [9]: advertising.isnull().sum()

Out[9]: 
TV           0
Radio        0
Newspaper    0
Sales        0
dtype: int64


In [10]: pd.set_option('precision', 2)                                          
In [11]: advertising.describe()                                                 
Out[11]: 
           TV   Radio  Newspaper   Sales
count  200.00  200.00     200.00  200.00
mean   147.04   23.26      30.55   14.02
std     85.85   14.85      21.78    5.22
min      0.70    0.00       0.30    1.60
25%     74.38    9.97      12.75   10.38
50%    149.75   22.90      25.75   12.90
75%    218.82   36.52      45.10   17.40
max    296.40   49.60     114.00   27.00
We can see that the min value of Radio is zero. We need to confirm how many zero values existing in the dataset.

For all other columns, the data cleaning is not required. However the data scaling is required.

Analysis of Zero Values in Predictors
In [12]: (advertising == 0).sum(axis=0)
Out[12]: 
TV           0
Radio        1
Newspaper    0
Sales        0
dtype: int64





Relationship between Sales and TV
In [14]: sns.regplot(advertising.TV, advertising.Sales, order=1, ci=None, scatte
    ...: r_kws={'color':'r', 's':9}) 
    ...: plt.xlim(-10,310)
    ...: plt.ylim(bottom=0)
    ...: plt.show() 

# order 1 for linear model
# ci - confidence interval
#scatter_kws Color - red size - 9
Advertisement

Relationship between Sales and Radio
In [15]: sns.regplot(advertising.Radio, advertising.Sales, order=1, ci=None, sca
    ...: tter_kws={'color':'r', 's':9}) 
    ...: plt.xlim(0,55) 
    ...: plt.ylim(bottom=0)
    ...: plt.show() 
Advertisement

Relationship between Sales and Newspaper
In [16]: sns.regplot(advertising.Newspaper, advertising.Sales, order=1, ci=None,
    ...:  scatter_kws={'color':'r', 's':9}) 
    ...: plt.xlim(-10,115) 
    ...: plt.ylim(bottom=0) 
    ...: plt.show() 
Advertisement




In [17]: from sklearn.preprocessing import scale
In [18]: X = scale(advertising.TV, with_mean=True, with_std=False).reshape(-1,1)
In [19]: y = advertising.Sales     

# scale - standardize the data set along any axis
# with_mean = True If True, center the data before scaling
# with_std = If True, scale the data to unit variance
# reshape (-1,1) one of new shape parameter as -1. It is an unknown dimension and we want
In [20]: X[0:5]                                                                 
Out[20]: 
array([[  83.0575],
       [-102.5425],
       [-129.8425],
       [   4.4575],
       [  33.7575]])
In [21]: X.mean()                                                               
Out[21]: 1.0089706847793422e-14

In [22]: X.std()                                                                
Out[22]: 85.63933175679269




Linear Regression for Scaled Data using sklearn
In [23]: import sklearn.linear_model as skl_lm

In [24]: regr = skl_lm.LinearRegression()

In [25]: regr.fit(X,y)

Out[25]: LinearRegression()

In [26]: regr.intercept_ 

Out[26]: 14.0225

In [27]: regr.coef_ 

Out[27]: array([0.04753664])
Calculate RSS
In [28]: min_rss = np.sum((regr.intercept_+regr.coef_*X - y.values.reshape(-1,1))**2)

In [29]: min_rss
Out[29]: 2102.5305831313514



Using Sklearn
In [30]: regr = skl_lm.LinearRegression()

In [31]: X = advertising.TV.values.reshape(-1,1)

In [32]: y = advertising.Sales

In [33]: regr.fit(X,y)
Out[33]: LinearRegression()

In [34]: regr.intercept_
Out[34]: 7.032593549127693

In [35]: regr.coef_
Out[35]: array([0.04753664])
RSS
In [36]: min_rss = np.sum((regr.intercept_+regr.coef_*X - y.values.reshape(-1,1))**2)

In [37]: min_rss
Out[37]: 2102.5305831313514
MSE
In [38]: mse = min_rss/len(y) 

In [39]: mse
Out[39]: 10.512652915656757
R-Sq using Sklearn
In [40]: from sklearn.metrics import mean_squared_error, r2_score

In [41]: Sales_pred = regr.predict(X)

In [42]: r2_score(y, Sales_pred) 
Out[42]: 0.611875050850071
MSE using SKLearn
In [43]: mean_squared_error(y, Sales_pred)
Out[43]: 10.512652915656757



In [44]: import statsmodels.formula.api as smf

In [45]: est = smf.ols('Sales ~ TV', advertising).fit()

In [46]: est.summary()
Out[46]:
<class 'statsmodels.iolib.summary.Summary'>
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  Sales   R-squared:                       0.612
Model:                            OLS   Adj. R-squared:                  0.610
Method:                 Least Squares   F-statistic:                     312.1
Date:                Mon, XX Apr XXXX   Prob (F-statistic):           1.47e-42
Time:                        11:49:06   Log-Likelihood:                -519.05
No. Observations:                 200   AIC:                             1042.
Df Residuals:                     198   BIC:                             1049.
Df Model:                           1                                         
Covariance Type:            nonrobust   
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      7.0326      0.458     15.360      0.000       6.130       7.935
TV             0.0475      0.003     17.668      0.000       0.042       0.053
==============================================================================
Omnibus:                        0.531   Durbin-Watson:                   1.935
Prob(Omnibus):                  0.767   Jarque-Bera (JB):                0.669
Skew:                          -0.089   Prob(JB):                        0.716
Kurtosis:                       2.779   Cond. No.                         338.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""
 
Regression RSS and MSE
In [47]: est.params                         
Out[47]: 
Intercept    7.03
TV           0.05
dtype: float64
RSS
In [48]: ((advertising.Sales - (est.params[0] + est.params[1] * advertising.TV))** 2).sum()

Out[48]: 2102.5305831313512
MSE
In [49]: ((advertising.Sales - (est.params[0] + est.params[1]*advertising.TV))** 2).sum()/len(advertising.Sales)
    
Out[49]: 10.512652915656757



near Regression Sales and Radio
In [50]: est = smf.ols('Sales ~ Radio', advertising).fit()

In [51]: print(est.summary().tables[1])

==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      9.3116      0.563     16.542      0.000       8.202      10.422
Radio          0.2025      0.020      9.921      0.000       0.162       0.243
==============================================================================
Check the p-value of Intercept and Radio.

It shows that there is a relationship between Sales and Radio

Linear Regression Sales and Newspaper
In [52]: est = smf.ols('Sales ~ Newspaper', advertising).fit()
In [53]: print(est.summary().tables[1])
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     12.3514      0.621     19.876      0.000      11.126      13.577
Newspaper      0.0547      0.017      3.300      0.001       0.022       0.087
==============================================================================
Check the p value of Intercept and Newspaper.

It shows that there is a relationship between Sales and Newspaper



Multiple Linear Regression
In a three-dimensional setting, with two predictors and one response, the least squares regression line becomes a plane. The plane is chosen to minimize the sum of the squared vertical distances between each observation (shown in red) and the plane.

Regression

The multiple regression model

In this project the model is

In [54]: est = smf.ols('Sales ~ TV + Radio + Newspaper', advertising).fit()
In [55]: est.summary()
Out[55]:
<class 'statsmodels.iolib.summary.Summary'
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  Sales   R-squared:                       0.897
Model:                            OLS   Adj. R-squared:                  0.896
Method:                 Least Squares   F-statistic:                     570.3
Date:                Tue, xx Apr xxxx   Prob (F-statistic):           1.58e-96
Time:                        17:30:45   Log-Likelihood:                -386.18
No. Observations:                 200   AIC:                             780.4
Df Residuals:                     196   BIC:                             793.6
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      2.9389      0.312      9.422      0.000       2.324       3.554
TV             0.0458      0.001     32.809      0.000       0.043       0.049
Radio          0.1885      0.009     21.893      0.000       0.172       0.206
Newspaper     -0.0010      0.006     -0.177      0.860      -0.013       0.011






n [56]: advertising.corr()
Out[56]: 
             TV  Radio  Newspaper  Sales
TV         1.00   0.05       0.06   0.78
Radio      0.05   1.00       0.35   0.58
Newspaper  0.06   0.35       1.00   0.23
Sales      0.78   0.58       0.23   1.00







In many situations the response variable  is qualitative. Example is eye color, which can be blue, brown, or green. We also refer the qualitative variables as categorical too. The process or approach for predicting qualitative response is known as classification.

As we have studied in the case of regression, we have a set of training observation . We can use the training set to build a classifier that can perform well not only on training data but also on the test data.

Many possible classification techniques or classifiers to predict a qualitative response are available. The three of the most widely used classifiers are logistic regression, linear discriminant analysis, and -nearest neighbors.



Classification problems occur more often than regression problems. Some of the examples are

A patient reaches the emergency ward of a hospital. He has a set of symptoms. These symptoms could be attributed to one of three medical conditions. Which of the three medical conditions the patient may have?
A payment gateway service wants to determine whether a particular transaction that has been performed on the website is fraudulent on the basis of user’s IP address, past transaction history, so forth.






We will try to understand the concepts of classification using a case study. The objective of the case study is to predict whether an individual will default on his or her credit card balance, on the basis of annual income and monthly credit card balance.

Credit Card Fraud

Figure 1

The diagram above shows plot pf the annual income and monthly credit card balance for a subset of  individual. In the left-hand panel, the data for individuals who defaulted in a given month has been displayed in orange and who did not in blue. From the plot we may infer that individuals who defaulted tended to have higher credit card balances than those who did not. In the right-hand side panel, we have shown two pairs of boxplots. The first box plot shows the distribution of balance that has been split by the binary default variable. The second plot shows the similar distribution for income.

In this case study, we shall learn, how can we build a model to predict default  for a given value of balance  and income . Since the response variable, default, is not quantitative, the use of simple linear regression will not be appropriate.




Why the linear regression is not an appropriate model for the qualitative response?

Suppose the research goal of a study is to predict the medical condition of a patient in the emergency ward based on the symptoms. In this example, three possible diagnoses would be, stroke, drug overdose, and epileptic seizure. If we encode these values as the quantitative response variable  we get

This coding means that there is an ordering on the outcome whereas drug overdose is placed between stroke and epileptic seizure. It also means that the difference between stroke and drug overdose is the same as the difference between drug overdose and epileptic seizure. In the practical scenario, this is not the case. On the other hand, we may also choose some other coding

This coding suggests a totally different relationship among the three conditions and would produce a different linear model. This will lead to a different set of predictions on test observations. In general, it is not possible to convert a qualitative response variable with more than two levels into a quantitative response for linear regression.

Suppose we consider binary (two-level) qualitative response. For the example given above, suppose there are only two possibilities for the patient’s medical condition: stroke and drug overdose. In such a situation, we can use the dummy variable approach to code the response as follows:

For this binary response, we can fit linear regression and predict drug overdose if  and stroke otherwise. Even if we flip the above coding, the linear regression will produce the same results.

Linear Regression may give

The above equation can provide estimate of 
However, if we use the linear regression, some of the estimates might be outside the  interval. For a predicted value that is close to zero, we may predict a negative probability for default. If we predict a very large value, the probability could be bigger than . This is not sensible as probability should fall between  and . So, we should model  using a function that gives output between  and .

Credit Card Default





We can use logistic regression to model the probability that  belongs to a particular category. For our case study, logistic regression models the probability of default. The probability of default given balance is

The value of  can also be written as , will range between  and . Hence, we can predict  for any individual for whom .


In the previous section, we talked about how the linear regression may result in probabilities that may be lesser than  or bigger than . To avoid such problems, we need to model  such that the outputs fall between  and  for all values of . We can use the logistic function.

The figure below illustrates the fit of the logistic regression model to the data used for the credit card default case study.

Credit Card Default

Figure - 3

Here for low balance, we predict the probability of the default as close to zero but never below zero. Similarly for high balance, we predict the default probability close to, but never one. The logistic function will always produce -Shaped curve. Hence, regardless of the value of , we will always obtain a sensible prediction. We also notice that the logistic model can capture the range of probabilities more effectively than the linear regression model.

We can also write the logistic equation as

We can call the quantity  as odds which can take any value between  and . Values that are close to  and  indicate very low and very high probabilities of default. For example, if  out of  people defaults, , odds would be . Similarly, on average nine out of every ten people with an adds of  will default because  means an odds of .

We can take the logarithm of both sides to get

The left-hand side is called the log-odds or logit. So we can see that logit is linear in .

In logistic regression model, the log odds change by , if increase  by one unit. The relationship between  and  is not a straight line, so  does not corresponds to change in  with respect to one-unit increase in . The amount that  changes due to one-unit increase in  depends on the current value of . However, we can say that if  is positive, an increase in  will be associated with increase in . Similarly, if  is negative, an increase in  will be associated with decrease in .



How to estimate the logistic regression coefficients?

For logistic regression, the coefficients  and  are unknown. We need to estimate them based on the available training data. For non-linear models, we prefer the maximum likelihood approach.

In the approach, we try to estimate  and  such that the predicted probability  of default of each individual is as close as possible to the individual’s observed default status.

In other words, we can say that we try to find  and  such that if we plug the estimates into the model for , we get a number that is close to  for all those individuals who defaulted and a number that is close to zero for all the individuals who did not.

We can easily fit the logistic regression using statistical software packages such as R and Python, so in this book, we need not focus on the details of maximum likelihood fitting procedure.






Table 1 exhibits the coefficient estimates and related information that we got after fitting the logistics regression model on the Default data to predict the probability of default=Yes using balance. From the results we can see that . This indicates that the increase in balance and increase in the probability of default is associated. A one-unit increase in balance results in an increase in the log odds of default by  units.

Table - 1

             Coef.  Std.Err.          z          P>|z|
const   -10.651331  0.361169 -29.491287  3.723665e-191
balance   0.005499  0.000220  24.952404  2.010855e-137
Table 1 also exhibits the similarity between linear regression output and logistic regression output. The z-statistics in the Table 1 plays the same role as the -statistics in linear regression output.

The -statistics associated with  in this case is equal to . A large value of -statistics is an indication of evidence against the null hypothesis . Moreover, due to a very small -value, we can reject  and we can conclude that there is an association between balance and probability of default. At this point, we are not interested in the estimated intercept value as shown in Table – 1.

Video
