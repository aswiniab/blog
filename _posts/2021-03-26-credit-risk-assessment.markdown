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









