---
layout: post
title: Arbitrage Pricing Model
date: 2020-07-26 00:00:00 +0300
description: Arbitrage Pricing Model captures the sensitivity of an asset's returns to changes in certain macroeconomic variables that affects it
img: apt2.jpeg # Add image post (optional)
tags: [Arbitrage Pricing Theory, APT, STATA] # add tag
---

### 1. Why do we need an arbitrage pricing model? 
Occasionaly, assets gets mispriced in the market – either overvalued or undervalued – for a brief period of time. Market action would eventually correct the situation, moving price back to its fair market value. However, to an arbitrageur, temporarily mispriced securities represent a short-term opportunity to profit without any risk. 
Arbitrage pricing model helps us to estimate the fair market price of a stock. It is based on Arbitrage Pricing Theory (APT) that holds that an asset’s returns can be forecasted with the linear relationship of an asset’s expected returns and the macroeconomic factors that affect the asset’s risk.  The APT aims to pinpoint the fair market price of a security that may be temporarily incorrectly priced. 

Let us try to develop an arbitrage pricing model for returns on Microsoft stock.

### 2. What factors affect the stock price?

 Through various studies, the following factors have been identified to impact Microsoft stock price:

| Factor | Impact on Microsoft stock price |
|:---:|:---:|
|S&P500 index|Positive|
|Consumer price index|Negative|
|Industrial production index|Positive|
|Treasury bill yields|Negative|
|Consumer credit|Positive|
|Credit spread|Negative|

The S&P 500 is largely considered an essential benchmark index for the U.S. stock market. Composed of 500 large-cap companies across a breadth of industry sectors, the index captures the pulse of the American corporate economy. A high value of the the index indicate that the stock market is optimistic. 
The Consumer Price Index (CPI) is a measure of the average change over time in the prices paid by urban consumers for a market basket of consumer goods and services. If the CPI index increases, the return from the stock looks less attractive. 
The industrial production index (IPI) is a monthly economic indicator measuring real output in the manufacturing, mining, electric and gas industries, relative to a base year. A increased IPI indiacte increased industrial activity and is expected to have a postive impac on the reurn of stocks.
Treasury bill yields is a means of 'narrow' money supply. As the treasury bill yeilds increases, stock as an investment vehicle becomes unattractive relative to treasury bonds.
Consumer credit is personal debt taken on to purchase goods and services. A credit card is one form of consumer credit. Although any type of personal loan could be labeled consumer credit, the term is usually used to describe unsecured debt that is taken on to buy everyday goods and services. Increased consumer credit indicate increased consumer spending, leading to increased production and transctions in the market. Hence, it has a positive impact on the stock returns.
Credit spread measures the difference in yield between U.S. Treasury bonds and other debt securities of lesser quality, such as corporate bonds.Credit spreads widen when U.S. Treasury markets are favored over corporate bonds, typically in times of uncertainty or when economic conditions are expected to deteriorate.
M1 includes funds that are readily accessible for spending such as currency, traveler's checks, demand deposits, and Other Checkable Deposits (OCDs). 

### 3. How can we model the stock return?
The total return for a stock is the sum of capital gains/losses and dividend income.
To model the stock return, we first gather the input data and apply data transformations. With the transformed data, we use a linear regression model to explain the relationship between the predictors and the stock return. 

#### 3.1 Data gathering
To prepare for analysis and model development, the following data since March 1986 were collected and cleaned:
* MICROSOFT: closing value (currency in USD) of Microsoft soft on 1st day of the month.
* SANDP: closing value of S&P500 (currency in USD) on 1st day of the month.
* CPI: monthly data on consumer price index  
* INDPRO:The Industrial Production Index (INDPRO) is an economic indicator that measures real output for all facilities located in the United States manufacturing,  
  mining, and electric, and gas utilities.
* USTB3M: the monthly average of US Treasury Bill Yield (in percentage) for 3 months maturity 
* USTB10Y: the monthly average of US Treasury Bill Yield (in percentage) for 10 year maturity in percentage
* M1SUPPLY: the monthly average seasonally adjusted M1 money stock value in billions of dollars 
* CCREDIT: Total Consumer Credit Owned and Securitized, Outstanding monthly end of period data in billions of dollars not seasonally adjusted 
* BMINUSA: Moody's Seasoned Baa Corporate Bond Minus Federal Funds Rate monthly data on percent units not seasonally adjusted. BMINUSA is calculated as the spread between [Moody's Seasoned Baa Corporate Bond](https://fred.stlouisfed.org/series/BAA) and [Effective Federal Funds Rate](https://fred.stlouisfed.org/series/EFFRM)

|Data|Abbreviation|Source|
|:---:|:---:|:---:|
|Microsoft stock return|MICROSOFT|[finance.yahoo.com](https://finance.yahoo.com)|
|S&P Price index|SANDP|[finance.yahoo.com](https://finance.yahoo.com)|
|Consumer Price Index|CPI|[US Inflation Calculator](https://www.usinflationcalculator.com)|
|Industrial Production Index|INDPRO|[Fred Economic Data](https://fred.stlouisfed.org)|
|US Treasury bill yield for 3 months|USTB3M|[Fred Economic Data](https://fred.stlouisfed.org)|
|US Treasury bill yield for 10 years|USTB10Y|[Fred Economic Data](https://fred.stlouisfed.org)|
|M1 money stock|M1SUPPLY|[Fred Economic Data](https://fred.stlouisfed.org)|
|Consumer Credit|CCREDIT|[Fred Economic Data](https://fred.stlouisfed.org)|
|Credit Spread|BMINUSA|[Fred Economic Data](https://fred.stlouisfed.org)|


#### 3.2 Data Transformation
Now that we have prepared our dataset we can start with the actual analysis. 
Lets import the data file and check the variables using STATA.
```
import excel "G:\Portfolio\data.xlsx", sheet("MSFT") firstrow
# To output details of variable 'Date'
codebook Date
```
It can be observed that the 'Date' variable has units 'days' while actually it should be 'months'.
![Date_days]({{site.baseurl}}/assets/img/APT_output/Date_unit_days.png)

```
# replace Date unit with month
replace Date = mofd(Date)
format %tm Date
tsset Date, monthly
codebook Date
```
The 'Date'variable now has units 'months' which is correct.
![Date_month](https://github.com/aswiniab/blog/blob/gh-pages/assets/img/APT_output/Date_unit_months.png)

#### 3.2.1 Log-Transformation
We apply log transformation to the following data series since all the values are positive and exhibit high positive skewness. This helps in linearizing the model. 
* MICROSOFT: Microsoft stock price series 
* SANDP: S&P stock index
* CPI: Consumer Price Index

##### 3.2.2 Transformation from levels to first differences
The APT posits that the stock returns can be explained by reference to the unexpected changes in the macroeconomic variables rather than their levels. This is required as we are dealing with a dynamic model here. The current value of the stock return depends on previous values of stock along with other variables. 

> Dynamic model is a model where the current value of yt depends on previous values of y or on previous values of one or more of the variables, eg.,
y_t=β_1+β_2 x_1t+β_3 x_2t+β_4 x_3t+β_5 x_4t+⋯+γ_1 y_(t-1)+γ_2 x_(1t-1)+⋯+γ_k x_(kt-1)+u_t
![equation](http://www.sciweavers.org/download/Tex2Img_1598726436.jpg)

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
```
generate rmsoft = 100*(ln(MICROSOFT/L.MICROSOFT))
generate mustb3m=USTB3M/12
generate ermsoft=rmsoft-mustb3m
```
**Predictors**
There are seven macroeconomic variables which act as preditors in this regression model. Their first differences is given by:
•	dspread = BMINUSA - L.BMINUSA
•	dcredit = CCREDIT - L.CCREDIT
•	dprod = INDPRO - L.INDPRO
•	rsandp = 100 * (ln (MICROSOFT/L.MICROSOFT) )
•	dmoney = M1SUPPLY - L.M1SUPPLY
•	inflation = 100 * (ln (CPI/L.CPI) )
•	term = USTB10Y - USTB3M

STATA code:
```
generate dspread = BMINUSA-L.BMINUSA
generate dcredit = CCREDIT-L.CCREDIT
generate dprod = INDPRO-L.INDPRO
generate rsandp = 100*(ln(SANDP/L.SANDP))
generate dmoney = M1SUPPLY-L.M1SUPPLY
generate inflation = 100*(ln(CPI/L.CPI))
generate term = USTB10Y-L.USTB3M
```
Next we need to apply further transformations to some of the transformed series, so we generate another set of variables:
•	dinflation = inflation - L.inflation
•	rterm = term - L.term
•	ersandp = rsandp - mustb3m 

```
generate dinflation = inflation-L.inflation
generate rterm=term-L.term
generate ersandp=rsandp-mustb3m
```

#### 3.3 Regression Analysis
We can now ready run the regression. 
```
regress ermsoft ersandp rterm dpro dcredit dinflation dmoney  dspread
```
Output

```
test (dpro dcredit dmoney dspread)
```
output

Note that the signs of the coefficients estimates match with the expected impact of the variable given in table 1. The regression output shows that only ersandp, dinflation and rterm factors have significant impact on ermsoft. Hence, the model can be reduced as:

〖ermsoft〗_t=1.33+1.28〖 ersandp〗_t+2.19〖 dinflation〗_t+4.73 〖rterm〗_t+u_t

Using this model the fair market price of Microsoft stock can be estimated. In the event the market misprices the stock, the arbitrage opportunity can be harnessed to gain.
