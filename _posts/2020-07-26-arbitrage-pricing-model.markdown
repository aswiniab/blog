---
layout: post
title: Arbitrage Pricing Model
date: 2020-07-26 00:00:00 +0300
description: Arbitrage Pricing Model captures the sensitivity of an asset's returns to changes in certain macroeconomic variables that affects it
img: apt2.jpeg # Add image post (optional)
tags: [Arbitrage Pricing Theory, APT, STATA] # add tag
---

### 1. Why do we need an arbitrage pricing model? 
Occasionaly, assets gets mispriced in the market – either overvalued or undervalued – for a brief period of time. Market action would eventually correct the situation, moving price back to its fair market value. However, to an arbitrageur, temporarily mispriced securities represent a short-term opportunity to profit, virtually without risk. 
Arbitrage pricing model helps us to estimate the fair market price of a stock. It is based on Arbitrage Pricing Theory (APT) that holds that an asset’s returns can be forecasted with the linear relationship of an asset’s expected returns and the macroeconomic factors that affect the asset’s risk.  The expected return is the profit or loss an investor anticipates on an investment, usually expressed in percentage.

Let us try to develop an arbitrage pricing model for returns on Microsoft stock.

### 2. What factors affect the Microsoft stock price?

Several studies have been conducted on factors that could affect stock returns. The following factors have been identified to have a potential impact on stock price:

*Table1:Factors affecting stock price and their potential impact*
| Factor | Impact on stock price |
|:---:|:---:|
|S&P500 index|Positive|
|Consumer price index|Negative|
|Industrial production index|Positive|
|Treasury bill yields|Negative|
|Consumer credit|Positive|
|Credit spread|Negative|
|Money supply|Positive|

*	**S&P 500** is a stock market index that measures the stock performance of 500 large companies listed on stock exchanges in the United States. Many consider it to be one of the best representations of the U.S. stock market.
*	**Consumer Price Index (CPI)** is a measure of the average change over time in the prices paid by urban consumers for a market basket of consumer goods and services. If the CPI index increases, the return from the stock looks less attractive. It is used for calculating inflation.
*	**Industrial Production Index (IPI)** is a monthly economic indicator measuring real output in the manufacturing, mining, electric and gas industries, relative to a base year. A increased IPI indicate increased industrial activity and is expected to have a positive impact on the return of stocks.
*	**Treasury yield** is the effective interest rate (expressed as percentage) that the U.S. government pays to borrow money for different lengths of time. As the treasury bill yield increases, stock as an investment vehicle becomes unattractive relative to treasury bonds.
*	**Consumer credit** is personal debt taken on to purchase goods and services. A credit card is one form of consumer credit. Although any type of personal loan could be labelled consumer credit, the term is usually used to describe unsecured debt that is taken on to buy everyday goods and services. Increased consumer credit indicate increased consumer spending, leading to increased production and transactions in the market. Hence, it has a positive impact on the stock returns.
*	**Credit spread** measures the difference in yield between U.S. Treasury bonds and other debt securities of lesser quality, such as corporate bonds.Credit spreads widen when U.S. Treasury markets are favored over corporate bonds, typically in times of uncertainty or when economic conditions are expected to deteriorate.
*	**Money supply** is all the currency and other liquid instruments in a country's economy on the date measured that can be used almost as easily as cash. The more the money supply, the higher price of stock.
 

### 3. How can we model the stock return?
The total return for a stock is the sum of capital gains/losses and dividend income.
To model the stock return, we first gather the input data and apply data transformations. With the transformed data, we use a linear regression model to explain the relationship between the predictors and the stock return. 

#### 3.1 Data gathering
To prepare for analysis and model development, the following data for the period March 1986 to June 2020 were collected and cleaned:
*Table2:Data sources*
|Data|Variable|Source|
|:---:|:---:|:---:|
|Microsoft stock return|MICROSOFT|[finance.yahoo.com](https://finance.yahoo.com)|
|S&P 500 index|SANDP|[finance.yahoo.com](https://finance.yahoo.com)|
|Consumer Price Index|CPI|[US Inflation Calculator](https://www.usinflationcalculator.com)|
|Industrial Production Index|INDPRO|[Fred Economic Data](https://fred.stlouisfed.org)|
|Treasury bill yields |USTB3M, USTB10Y|[Fred Economic Data](https://fred.stlouisfed.org)|
|Consumer Credit|CCREDIT|[Fred Economic Data](https://fred.stlouisfed.org)|
|Credit Spread|BMINUSA|[Fred Economic Data](https://fred.stlouisfed.org)|
|Money Supply|M1SUPPLY|[Fred Economic Data](https://fred.stlouisfed.org)|

* SANDP: captures the closing value of S&P 500 stock index in USD as on the first day of every month for the period. 
*	CPI: captures the consumer price index data series for the period mentioned. The data is based upon base year 1982 which has CPI 100. What does this mean? A CPI of 195.3, as an example from year 2005, indicates 95.3% inflation since 1982.
* INDPRO: captures the seasonally adjusted monthly value of the index. The data has 2012 as the base year with an index value of 100. 
* USTB3M and USTB10Y:  captures the average monthly value of US treasury bill yield with a maturity period of 3 months and 10 years respectively.
* CCREDIT: captures total consumer credit in billions of dollars (not seasonally adjusted) outstanding at the end of a month for period March 1986 to June 2020. 
* BMINUSA: captures Moody's Seasoned Baa Corporate Bond Minus Federal Funds Rate monthly data on percent units not seasonally adjusted. BMINUSA is calculated as the spread  between [Moody's Seasoned Baa Corporate Bond](https://fred.stlouisfed.org/series/BAA) and [Effective Federal Funds Rate](https://fred.stlouisfed.org/series/EFFRM)
* M1SUPPLY: captures the monthly average seasonally adjusted money supply value in billions of dollars. It includes funds that are readily accessible for spending such as currency, traveler's checks, demand deposits, and Other Checkable Deposits (OCDs). 

#### 3.2 Data Transformation
Now that we have prepared our dataset we can start with the actual analysis. 
Lets import the data file and check the variables using STATA.

{% highlight ruby %}
import excel "G:\Portfolio\data.xlsx", sheet("MSFT") firstrow
# display details of variable 'Date'
codebook Date
{% endhighlight %}

It can be observed that the 'Date' variable has units 'days' while actually it should be 'months'.
![Date_days]({{site.baseurl}}/assets/img/APT_output/Date_days.png)

{% highlight ruby %}
# replace Date unit with month
replace Date = mofd(Date)
format %tm Date
tsset Date, monthly
codebook Date
{% endhighlight %}

The 'Date' variable now has units 'months' which is correct.
![Date_month]({{site.baseurl}}/assets/img/APT_output/Date_months.png)

#### 3.2.1 Log-Transformation
We apply log transformation to the following data series since all the values are positive and exhibit high positive skewness. This helps in linearizing the model. 
* MICROSOFT: Microsoft stock price series 
* SANDP: S&P stock index
* CPI: Consumer Price Index

##### 3.2.2 Transformation from levels to first differences
The APT posits that the stock returns can be explained by reference to the unexpected changes in the macroeconomic variables rather than their levels. This is required as we are dealing with a dynamic model here. The current value of the stock return depends on previous values of stock along with other variables. 

> Dynamic model is a model where the current value of yt depends on previous values of y or on previous values of one or more of the variables, eg., ![img](http://www.sciweavers.org/tex2img.php?eq=y_%7Bt%7D%3D%20%5Cbeta%20_%7B1%7D%2B%5Cbeta_%7B2%7D%20x_%7B2t%7D%2B%5Cbeta_%7B3%7D%20x_%7B3t%7D%2B%5Cbeta_%7B4%7D%20x_%7B4t%7D%2B...%2B%20%5Cgamma_%7B1%7D%20y_%7Bt-1%7D%2B%5Cgamma_%7B2%7D%20x_%7B2t-1%7D%2B%E2%8B%AF%2B%5Cgamma_%7Bk%7D%20x_%7Bkt-1%7D%2Bu_%7Bt%7D&bc=White&fc=Gray&im=jpg&fs=12&ff=arev&edit=0[/img])

This makes the errors to be correlated with one another, violating one of the assumption of Classical Linear Regression Model (CLRM). The errors are said to be ‘autocorrelated’ in this case.  A potential remedy for autocorrelated residuals would be to switch to a model in first differences rather than in levels. 
What are unexpected changes in the macroeconomic variables? The unexpected value of a variable can be defined as the difference between the actual (realised) value of the variable and its expected value. The question then arises that what is the expected value of the variables? It can be assumed that the investors have naive expectations that the next period value of the variable is equal to the current value. This being the case, the entire change in the variable from one period to the next is the unexpected change (because investors are assumed to expect no change).
Hence, the first stage is to generate a set of changes or differences for each of the variables.

**Target variable**
The variable whose behaviour we seek to explain is the excess return of the Microsoft stock in percentage.
To find this, first we find the first difference of the variable:
rmsoft = 100 * (ln (MICROSOFT / L.MICROSOFT) )
The excess return of the stock is given by subtracting the risk free rate from the rate of the stock. The US treasury bill yield of 3 months maturity (USTB3M) is usually taken as the risk free rate. The monthly risk free rate is given by
mustb3m = USTB3M / 12 
the excess return of the Microsoft stock is given by:
ermsoft = rmsoft – mustb3m

The STATA codes are:
{% highlight ruby %}
generate rmsoft = 100*(ln(MICROSOFT/L.MICROSOFT))
generate mustb3m=USTB3M/12
generate ermsoft=rmsoft-mustb3m
{% endhighlight %}
**Predictors**
There are seven macroeconomic variables which act as preditors in this regression model. Their first differences is given by:
*	dspread = BMINUSA - L.BMINUSA
*	dcredit = CCREDIT - L.CCREDIT
*	dprod = INDPRO - L.INDPRO
*	rsandp = 100 * (ln (MICROSOFT/L.MICROSOFT) )
*	dmoney = M1SUPPLY - L.M1SUPPLY
*	inflation = 100 * (ln (CPI/L.CPI) )
*	term = USTB10Y - USTB3M

STATA code:
{% highlight ruby %}
generate dspread = BMINUSA-L.BMINUSA
generate dcredit = CCREDIT-L.CCREDIT
generate dprod = INDPRO-L.INDPRO
generate rsandp = 100*(ln(SANDP/L.SANDP))
generate dmoney = M1SUPPLY-L.M1SUPPLY
generate inflation = 100*(ln(CPI/L.CPI))
generate term = USTB10Y-L.USTB3M
{% endhighlight %}
Next we need to apply further transformations to some of the transformed series, so we generate another set of variables:
*	dinflation = inflation - L.inflation
*	rterm = term - L.term
*	ersandp = rsandp - mustb3m 

{% highlight ruby %}
generate dinflation = inflation-L.inflation
generate rterm=term-L.term
generate ersandp=rsandp-mustb3m
{% endhighlight %}

#### 3.3 Regression Analysis
We can now ready run the regression. 
{% highlight ruby %}
regress ermsoft ersandp rterm dpro dcredit dinflation dmoney  dspread
{% endhighlight %}
![Date_month]({{site.baseurl}}/assets/img/APT_output/regression.png)

{% highlight ruby %}
test (dpro dcredit dmoney dspread)
{% endhighlight %}

![Date_month]({{site.baseurl}}/assets/img/APT_output/Ftest.png)

Note that the signs of the coefficients estimates match with the expected impact of the variable given in table 1. The regression output shows that only ersandp, dinflation and rterm factors have significant impact on ermsoft. Hence, the model can be reduced as:
![img](http://www.sciweavers.org/tex2img.php?eq=ermsoft_%7Bt%7D%20%3D%201.33%20%2B%201.28%20ersandp_%7Bt%7D%20%2B%202.19%20dinflation_%7Bt%7D%20%2B%204.73%20rterm_%7Bt%7D%20%2B%20u_%7Bt%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0[/img])

Using this model the fair market price of Microsoft stock can be estimated. In the event the market misprices the stock, the arbitrage opportunity can be harnessed to gain.
