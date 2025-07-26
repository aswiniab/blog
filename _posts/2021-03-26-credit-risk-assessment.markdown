---
layout: post
title: 
date: 2021-03-26 00:00:00 +0300
description: Credit Risk Assessment
img: cred.jpg # Add image post (optional)
fig-caption: Photo from Unsplash by CardMapr.nl
tags: [Credit Risk, Machine Learning] # add tag
---

[View Credit Risk Assessment Project](../assets/german-credit-risk-blog.html)

Loans form an integral part of banking operations. However, not all the loans are promptly returned and hence it is important for a bank to closely monitter its loan applications. This project is an analysis of the German credit dataset from the UCI Machine Learning repository. The dataset contains details of 1000 loan applicants with 21 attributes along with the classification as good or bad credit.

In this project, the relationship between the credit risk and various attribues will be explored through basic statistical techniques, and presented through visualizations.

**Contents**

1. Import data
2. Data preparation, cleaning
3. Exploratory data analysis
4. Feature engineering
5. Models
6. Summary
7. References, future work

## 1. Import data

Let's begin by downloading the data from the [UCI Machine Learning repository](http://archive.ics.uci.edu/ml/about.html).

{% highlight ruby %}
from urllib.request import urlretrieve
urlretrieve('http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data', 'german.data')
{% endhighlight %}
```output
    ('german.data', <http.client.HTTPMessage at 0x7effdfbb5550>)
```
The dataset has been downloaded and extracted.

## 2. Data Preparation and Cleaning

In this step, we do data preparation and cleaning, making the data suitable for subsequent analysis.

**2.1 Load data into dataframe**

The datafile is in `.data` format, delimited with space, and has no headers.

{% highlight ruby %}
import pandas as pd
german_df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
delimiter=' ',header=None)
{% endhighlight %}

Now, let's have an over-view of the dataset.

{% highlight ruby %}
german_df.info()
{% endhighlight %}
```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 21 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   0       1000 non-null   object
     1   1       1000 non-null   int64
     2   2       1000 non-null   object
     3   3       1000 non-null   object
     4   4       1000 non-null   int64
     5   5       1000 non-null   object
     6   6       1000 non-null   object
     7   7       1000 non-null   int64
     8   8       1000 non-null   object
     9   9       1000 non-null   object
     10  10      1000 non-null   int64
     11  11      1000 non-null   object
     12  12      1000 non-null   int64
     13  13      1000 non-null   object
     14  14      1000 non-null   object
     15  15      1000 non-null   int64
     16  16      1000 non-null   object
     17  17      1000 non-null   int64
     18  18      1000 non-null   object
     19  19      1000 non-null   object
     20  20      1000 non-null   int64
    dtypes: int64(8), object(13)
    memory usage: 164.2+ KB
```
The dataset contains 21 variables and 1000 observatios. 8 variables are of numeric type and 13 of object type. As the object type variables do not have any null values, we can conclude that they are of categorical type.

**2.2 Label the columns**

Next, let's label the variables for ease of use. The [document](<(http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc)>) describing the dataset may be referred for this.

{% highlight ruby %}
urlretrieve('http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc', 'german.doc')
f = open('german.doc')
german_doc= f.read()
print(german_doc)
{% endhighlight %}
```output
    Description of the German credit dataset.

    1. Title: German Credit data

    2. Source Information

    Professor Dr. Hans Hofmann
    Institut f"ur Statistik und "Okonometrie
    Universit"at Hamburg
    FB Wirtschaftswissenschaften
    Von-Melle-Park 5
    2000 Hamburg 13

    3. Number of Instances:  1000

    Two datasets are provided.  the original dataset, in the form provided
    by Prof. Hofmann, contains categorical/symbolic attributes and
    is in the file "german.data".

    For algorithms that need numerical attributes, Strathclyde University
    produced the file "german.data-numeric".  This file has been edited
    and several indicator variables added to make it suitable for
    algorithms which cannot cope with categorical variables.   Several
    attributes that are ordered categorical (such as attribute 17) have
    been coded as integer.    This was the form used by StatLog.


    6. Number of Attributes german: 20 (7 numerical, 13 categorical)
       Number of Attributes german.numer: 24 (24 numerical)


    7.  Attribute description for german

    Attribute 1:  (qualitative)
    	       Status of existing checking account
                   A11 :      ... <    0 DM
    	       A12 : 0 <= ... <  200 DM
    	       A13 :      ... >= 200 DM /
    		     salary assignments for at least 1 year
                   A14 : no checking account

    Attribute 2:  (numerical)
    	      Duration in month

    Attribute 3:  (qualitative)
    	      Credit history
    	      A30 : no credits taken/
    		    all credits paid back duly
                  A31 : all credits at this bank paid back duly
    	      A32 : existing credits paid back duly till now
                  A33 : delay in paying off in the past
    	      A34 : critical account/
    		    other credits existing (not at this bank)

    Attribute 4:  (qualitative)
    	      Purpose
    	      A40 : car (new)
    	      A41 : car (used)
    	      A42 : furniture/equipment
    	      A43 : radio/television
    	      A44 : domestic appliances
    	      A45 : repairs
    	      A46 : education
    	      A47 : (vacation - does not exist?)
    	      A48 : retraining
    	      A49 : business
    	      A410 : others

    Attribute 5:  (numerical)
    	      Credit amount

    Attibute 6:  (qualitative)
    	      Savings account/bonds
    	      A61 :          ... <  100 DM
    	      A62 :   100 <= ... <  500 DM
    	      A63 :   500 <= ... < 1000 DM
    	      A64 :          .. >= 1000 DM
                  A65 :   unknown/ no savings account

    Attribute 7:  (qualitative)
    	      Present employment since
    	      A71 : unemployed
    	      A72 :       ... < 1 year
    	      A73 : 1  <= ... < 4 years
    	      A74 : 4  <= ... < 7 years
    	      A75 :       .. >= 7 years

    Attribute 8:  (numerical)
    	      Installment rate in percentage of disposable income

    Attribute 9:  (qualitative)
    	      Personal status and sex
    	      A91 : male   : divorced/separated
    	      A92 : female : divorced/separated/married
                  A93 : male   : single
    	      A94 : male   : married/widowed
    	      A95 : female : single

    Attribute 10: (qualitative)
    	      Other debtors / guarantors
    	      A101 : none
    	      A102 : co-applicant
    	      A103 : guarantor

    Attribute 11: (numerical)
    	      Present residence since

    Attribute 12: (qualitative)
    	      Property
    	      A121 : real estate
    	      A122 : if not A121 : building society savings agreement/
    				   life insurance
                  A123 : if not A121/A122 : car or other, not in attribute 6
    	      A124 : unknown / no property

    Attribute 13: (numerical)
    	      Age in years

    Attribute 14: (qualitative)
    	      Other installment plans
    	      A141 : bank
    	      A142 : stores
    	      A143 : none

    Attribute 15: (qualitative)
    	      Housing
    	      A151 : rent
    	      A152 : own
    	      A153 : for free

    Attribute 16: (numerical)
                  Number of existing credits at this bank

    Attribute 17: (qualitative)
    	      Job
    	      A171 : unemployed/ unskilled  - non-resident
    	      A172 : unskilled - resident
    	      A173 : skilled employee / official
    	      A174 : management/ self-employed/
    		     highly qualified employee/ officer

    Attribute 18: (numerical)
    	      Number of people being liable to provide maintenance for

    Attribute 19: (qualitative)
    	      Telephone
    	      A191 : none
    	      A192 : yes, registered under the customers name

    Attribute 20: (qualitative)
    	      foreign worker
    	      A201 : yes
    	      A202 : no



    8.  Cost Matrix

    This dataset requires use of a cost matrix (see below)


          1        2
    ----------------------------
      1   0        1
    -----------------------
      2   5        0

    (1 = Good,  2 = Bad)

    the rows represent the actual classification and the columns
    the predicted classification.

    It is worse to class a customer as good when they are bad (5),
    than it is to class a customer as bad when they are good (1).
```
Based on the description, we name the columns.

{% highlight ruby %}
german_df.columns=['account_bal','duration','payment_status','purpose',
'credit_amount','savings_bond_value','employed_since',
'intallment_rate','sex_marital','guarantor','residence_since',
'most_valuable_asset','age','concurrent_credits','type_of_housing',
'number_of_existcr','job','number_of_dependents','telephon',
'foreign','target']
{% endhighlight %}

{% highlight ruby %}
german_df= german_df.replace(['A11','A12','A13','A14', 'A171','A172','A173','A174','A121','A122','A123','A124'],
['neg_bal','positive_bal','positive_bal','no_acc','unskilled','unskilled','skilled','highly_skilled',
'none','car','life_insurance','real_estate'])
{% endhighlight %}


## 3. Exploratory Data Analysis and Visualization

{% highlight ruby %}

# import libraries for visualizations

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
{% endhighlight %}

**3.1 Examine missing values**

{% highlight ruby %}

# check for missing values

german_df.isna().any().any()
{% endhighlight %}
```output
    False
```
**3.1 Examine distribution of target column**

{% highlight ruby %}
german_df.target.unique()
{% endhighlight %}
```output
    array([1, 2])
```
The `target` column has two values:

- 1: representing a good loan
- 2: representing a bad (defaulted) loan.

The usual convention is to use '1' for bad loans and '0' for good loans. Let's replace the values to comply to the convention.

{% endhighlight %}
from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()
le.fit(german_df.target)
german_df.target=le.transform(german_df.target)
german_df.target.head(5)
{% endhighlight %}
```output
    0    0
    1    1
    2    0
    3    0
    4    1
    Name: target, dtype: int64
```
An understanding of the percentage of good and bad loans would be useful for the further analysis. A pie chart would be best tool to help with this.

{% highlight ruby %}
good_bad_per=round(((german_df.target.value_counts()/german_df.target.count())\*100))
good_bad_per
plt.pie(good_bad_per,labels=['Good loans', 'Bad loans'], autopct='%1.0f%%', startangle=90)
plt.title('Percentage of good and bad loans');
{% endhighlight %}

![png](german-credit-risk-blog_files/german-credit-risk-blog_24_0.png)

The pie chart shows that 30% of the loan applicants defaulted. From this information, we see that this is an imbalanced class problem. Hence, we will have to weigh the classes by their representation in the data to reflect this imbalance.

**4.2 Exploration of continues variables**

- Summary statistics
- Histograms
- Box-plots

**Observations**

- A glance of the distribution of the continues variables shows that there is a wide gap in the range of variables. The credit amount variable might have to be transformed to bring all variables to similar range.
- The histogram suggests that the credit amount is approximatly normally distributed. However, age and duration have a skewed distribution.
- The box plots show that most of the credits amounts are between 1000 to 4500 dollars. The credit amount is positively skewed. Most of the loan duration is from 15 to 30 months. Majority of the loan applicants have age between 28 - 43.

{% highlight ruby %}
german_df[['credit_amount','duration','age']].describe()
{% endhighlight %}

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>credit_amount</th>
      <th>duration</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3271.258000</td>
      <td>20.903000</td>
      <td>35.546000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2822.736876</td>
      <td>12.058814</td>
      <td>11.375469</td>
    </tr>
    <tr>
      <th>min</th>
      <td>250.000000</td>
      <td>4.000000</td>
      <td>19.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1365.500000</td>
      <td>12.000000</td>
      <td>27.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2319.500000</td>
      <td>18.000000</td>
      <td>33.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3972.250000</td>
      <td>24.000000</td>
      <td>42.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>18424.000000</td>
      <td>72.000000</td>
      <td>75.000000</td>
    </tr>
  </tbody>
</table>
</div>

{% highlight ruby %}
german_df['credit_amount']=np.log(german_df['credit_amount'])
{% endhighlight %}

{% highlight ruby %}
german_df[['credit_amount','duration','age']].describe()
{% endhighlight %}

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
    

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>credit_amount</th>
      <th>duration</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.788691</td>
      <td>20.903000</td>
      <td>35.546000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.776474</td>
      <td>12.058814</td>
      <td>11.375469</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.521461</td>
      <td>4.000000</td>
      <td>19.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.219276</td>
      <td>12.000000</td>
      <td>27.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.749107</td>
      <td>18.000000</td>
      <td>33.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.287088</td>
      <td>24.000000</td>
      <td>42.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.821409</td>
      <td>72.000000</td>
      <td>75.000000</td>
    </tr>
  </tbody>
</table>
</div>

{% highlight ruby %}

# histograms of continues variables

fig, axes = plt.subplots(1,3, figsize=(16,8))
plt.suptitle('Histogram of continuous variables')
axes[0].hist(german_df['duration'])
axes[0].set_xlabel('No. of observations')
axes[0].set_ylabel('Years')
axes[0].set_title('Histogram of loan duration');

axes[1].hist(german_df['credit_amount'])
axes[1].set_xlabel('No. of observations')
axes[1].set_ylabel('Credit amount (dollars)')
axes[1].set_title('Histogram of Credit amount');

axes[2].hist(german_df['age'])
axes[2].set_xlabel('No. of observations')
axes[2].set_ylabel('Age')
axes[2].set_title('Histogram of Age');
{% endhighlight %}

![png](german-credit-risk-blog_files/german-credit-risk-blog_30_0.png)

{% highlight ruby %}

# box-plots of continues variables

fig, ax = plt.subplots(1,3,figsize=(20,5))
plt.suptitle('BOX PLOTS')
sns.boxplot(german_df['credit_amount'], ax=ax[0]);
sns.boxplot(german_df['duration'], ax=ax[1], color='salmon');
sns.boxplot(german_df['age'], ax=ax[2], color='darkviolet');
{% endhighlight %}
```output
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
```
![png](german-credit-risk-blog_files/german-credit-risk-blog_31_1.png)

**4.2 Relationship between the credit amount and repayment duration**

- scatter plot

**Observations**

The scatter plot shows that in general, larger loans have longer duration of repayment. Cases where large loans are given with short repayment period have turned out to be bad loans.

{% highlight ruby %}
sns.scatterplot(y=german_df.credit_amount,
x=german_df.duration,
hue=german_df.target,
s=100,
);
{% endhighlight %}

![png](german-credit-risk-blog_files/german-credit-risk-blog_33_0.png)

