---
layout: post
title: Arbitrage Pricing Model
date: 2020-07-26 00:00:00 +0300
description: Arbitrage pricing model captures the sensitivity of an asset's returns to changes in certain macroeconomic variables that affect it.# Add post description (optional)
img: apt2.jpeg # Add image post (optional)
tags: [Arbitrage Pricing Theory, APT, STATA] # add tag
---

Arbitrage pricing model is based on Arbitrage Pricing Theory (APT) which captures the sensitivity of the asset's returns to changes in certain macroeconomic variables that affect it. Occasionaly, assets gets mispriced in the market – either overvalued or undervalued – for a brief period of time. However, market action should eventually correct the situation, moving price back to its fair market value. To an arbitrageur, temporarily mispriced securities represent a short-term opportunity to profit virtually risk-free. The APT aims to pinpoint the fair market price of a security that may be temporarily incorrectly priced. For example, the returns on Microsoft’s stock can be explained can be explained based on a set of macroeconomic and financial variables.

### Problem Statement

We wish to develop a model to predict the fair price of Microsoft's stock using Arbitrage Pricing Theory (APT) 

### Development of Model

**Step1.Identification of factors affecting the stock**
The first step is to identify factors that we believe affects the return on the asset. 
In the case presented here, the following factors have been identified to impact the return of Microsoft stock.

| Factor | Impact on Microsoft's stock return |
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
