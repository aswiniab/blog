---
layout: post
title: Linear Regression using STATA
date: 2020-07-23 13:32:20 +0300
description: Estimation of an optimal hedge ratio # Add post description (optional)
img: hedge.png # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [Linear regression, STATA]
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
![codebook_date]({{site.baseurl}}/assets/img/codebook_Date.png)

Hence we generate a new variable 'date' which contains the monthly equivalent of 'Date'
```
generate date= mofd(Date)

# change display format of 'date' from daily to monthly
format %tm date

codebook date
```
The output shows that 'date' variable has monthly unit
![codebook_date]({{site.baseurl}}/assets/img/codebook_date2.png)

```
# as we are to perform time series analysis, indiacte that date is the time series time indicating
  variable and it has monthly data
tsset date, monthly
```
We want to run ur analysis based on retuns of S&P500 index, instead of price levels. So the next step is to transform the spot and future prices into percentage returns. As commonly followed in academic finance reserch, we use continuesly compounded retruns (ie, logarithmic returns)for our analysis instead of simple returns

```
simple return Ft (%)= 100* (F‐L.F)/L.F‐L
continuesly compunded return ft (%)= 100* ln(F/L.F)
```
The summary statistics of futures and spot returns shows that the two return series are very similar in terms of mean, std dev, min and max, as expected from the economic theory.

```
regress rspot rfutures
```
```
log close
save "G:\ECON803\STATA_selfstudy\Regression_SandP.dta", replace
```
