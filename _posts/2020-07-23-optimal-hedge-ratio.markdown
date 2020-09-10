---
layout: post
title: Estimation of an optimal hedge ratio
date: 2020-07-23 13:32:20 +0300
description: Estimation of an optimal hedge ratio # Add post description (optional)
img: hedge.png # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [Hedge ratio, Linear regression, STATA]
---
## What is a Regression Model?
Regression is an attempt to explain movements in a variable by reference to movements in one or more variables. An example is how asset prices vary with their level of market risk. 

## Problem statement
An investor wishes to hedge a long position in the S&P500 (or its constituent stocks) using a short position in futures contracts. The investor is interested in the optimal hedge ratio, i.e., the number of units of the futures asset to sell per unit of the spot assets held.

## Methodology
The changes in the log of the spot and futures prices are known as spot and futures returns. In a regression model estimated for the return of spot and future price series, the slope parameter measures the optimal hedge ratio as well as the short run relationship between the two series. 

## Data
We have the spot and future prices of the S&P500 indices from Sept 1997 to March 2018 collected on a monthly basis.  

## STATA code
{% highlight ruby %}
import excel "G:\S&P_hedge.xls", sheet("s&p500") firstrow

# open a log file to records what appears in the result window
log using "G:\S&P_hedge.smcl"

# codebook helps to check the unit of variables and missing values
codebook 
{% endhighlight %}
The output shows that variable- 'Date' is recorded to have daily units whereas in actual it is monthly
![codebook_date]({{site.baseurl}}/assets/img/hedge/img1Date.png)

Hence we generate a new variable 'date' which contains the monthly equivalent of 'Date'
{% highlight ruby %}
generate date= mofd(Date)

# change display format of 'date' from daily to monthly
format %tm date

codebook date
{% endhighlight %}
The output shows that 'date' variable has monthly unit
![codebook_date]({{site.baseurl}}/assets/img/hedge/img2Daterevised.png)

{% highlight ruby %}
# as we are to perform time series analysis, indicate that date is the time series variable and it has monthly data
tsset date, monthly
{% endhighlight %}
We want to run ur analysis based on retuns of S&P500 index, instead of price levels. So the next step is to transform the spot and future prices into percentage returns. As commonly followed in academic finance research, we use continuously compounded retruns (ie, logarithmic returns)for our analysis instead of simple returns
simple return Ft (%)= 100* (F‐L.F)/L.F‐L
continuesly compunded return ft (%)= 100* ln(F/L.F)

{% highlight ruby %}
generate rfutures= 100*(ln(Futures)-(ln(L.Futures)))
generate rspot= 100*(ln(Spot)-(ln(L.Spot)))
{% endhighlight %}
Now, let us look at the summary statistics of 'rfututes' and 'rspot'
![codebook_date]({{site.baseurl}}/assets/img/hedge/img3_summarise.png)

The summary statistics of futures returns ('rfutures') and spot returns ('rspot') shows that the two return series are very similar in terms of mean, standard deviation, minimum and maximum, as expected from the economic theory.
Let us now estimate a regression to explain variation in the spot rates with variationin the futures rates.
{% highlight ruby %}
regress rspot rfutures
{% endhighlight %}
The regression result is given below:
![codebook_date]({{site.baseurl}}/assets/img/hedge/img4RegressRspotRfutures.png)

The parameter estimates for the intercept (αˆ) and slope (βˆ) are 0.013 and 0.975, respectively. The estimated return regression slope parameter measures the optimal hedge ratio as well as the short run relationship between the two series. Hence, the optimal hedge ratio is 0.975 here.

Regression of raw spot and futures indices can be interpreted as measuring the long run relationship between them. The code and output of regression of raw spot and futures indices is given below:
{% highlight ruby %}
regress Spot Futures
{% endhighlight %}
The regression result is given below:
![codebook_date]({{site.baseurl}}/assets/img/hedge/img5RegressSpotFutures.png)

Looking at the results, we find that the long-term relationship between spot and futures prices is almost 1:1 (as expected). The intercept of the price level regression can be considered to approximate the cost of carry.

### References
1. Book- 'Introductory Econometrics for Finance' (fourth edition) by Chris Brooks 
2. Book- 'STATA guide to accompany Introductory Econometrics for Finance' (fourth edition) by Lisa Schopohl, Robert Wichmann, Chris Brooks
3. [https://www.investopedia.com](https://www.investopedia.com)
