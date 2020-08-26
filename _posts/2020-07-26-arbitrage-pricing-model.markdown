---
layout: post
title: Arbitrage Pricing Model
date: 2020-07-26 00:00:00 +0300
description: Arbitrage pricing model captures the sensitivity of an asset's returns to changes in certain macroeconomic variables that affect it.# Add post description (optional)
img: apt2.jpeg # Add image post (optional)
tags: [Arbitrage Pricing Theory, APT, STATA] # add tag
---

### 1.Why do we need an arbitrage pricing model? 
Occasionaly, assets gets mispriced in the market – either overvalued or undervalued – for a brief period of time. Market action would eventually correct the situation, moving price back to its fair market value. However, to an arbitrageur, temporarily mispriced securities represent a short-term opportunity to profit without any risk. 
Arbitrage pricing model helps us to estimate the fair market price of a stock. It is based on Arbitrage Pricing Theory (APT) that holds that an asset’s returns can be forecasted with the linear relationship of an asset’s expected returns and the macroeconomic factors that affect the asset’s risk.  The APT aims to pinpoint the fair market price of a security that may be temporarily incorrectly priced. 

Let us try to develop an arbitrage pricing model for returns on Microsoft stock.

### 2.What factors affect the stock price?

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

### 3.How can we model stock return?
The total return for a stock is the sum of capital gains/losses and dividend income.
To model the stock return, we first gather the input data and apply data transformations. With the transformed data, we use a linear regression model to explain the relationship between the predictors and the stock return. 

#### 3.1 Data gathering
To prepare for analysis and model development, the following data from March 1986 till March 2018 were collected and cleaned:

|Data|Abbreviation|Source|
|:---:|:---:|:---:|
|Microsoft stock return|MICROSOFT|[finance.yahoo.com](https://finance.yahoo.com)|
|S&P Price index|SANDP||
|Consumer Price Index|CPI||
|Industrial Production Index|IP||
|US Treasury bill yield for 3 months|USTB3M||
|US Treasury bill yield for 10 years|USTB10Y||
||M1SUPPLY|
|Consumer Credit|CCREDIT|
||BMINUSA||

