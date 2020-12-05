---
layout: post
title: Retail Business Analytics
date: 2020-12-04 00:00:00 +0300
description: Analysis of grocery purchase data of 2000 customers to identify groups of customers with similar preferences
img: apt2.jpeg # Add image post (optional)
tags: [Machine Learning, Unsupervised Learning, Clustering, K-means clustering] # add tag
---

The goal of this project is to segment the customers of a grocery store based on their purchase data. The data set lists the quantities of 42 different items purchased by 2000 customers in individual shopping trips. Hence, the similarity on items purchased is used as the metric for grouping customers. A classification of this type is useful for targeted marketing and improving/tweaking products to meet customer requirements. 

Here, all we have is the set of attributes (42 grocery items) measured on a set of observations (2000 customers). There is no response variable. Hence, usual procedures of building a model that explains/predicts the response variable based on the values of the input attributes is ruled out. The class of algorithms that analyzes this type of unlabeled data is called unsupervised learning, where the aim is to discover hidden patterns in the data. Clustering is an unsupervised learning technique used to uncover distinct clusters of observations within a set of data. Clusters are groups of data-points such that members of the same group have similar features while members of different groups have dissimilar features. Here, the similarity/ dissimilarity between observations or clusters is measured in terms of distance between them. At the start of the analysis we do not know how many clusters are there, or whether they even exist. 

There are different methods for measuring the distance between observations such as *euclidean, manhattan, minkowski* etc. Which distance measure to choose will depend on the type of data and the nature of question that is being asked. Euclidean distance is the most common distance measure. It measures the straight-line distance between the data points. 
In this project, we choose the Euclidean method to measure the distance. Once the distance method is selected, the next question is, how do we measure the distance between two groups of observations, or between a group and a single observation? One method is to compute all pairwise dissimilarities between observations in cluster A and the observations in cluster B and use the average of them. 

The K-means clustering algorithm is the most common and computationally efficient clustering technique available. It works with continues variables. However, the disadvantage is that we have specify in advance the number of clusters to be created.

The objective of K-means clustering is to partition the data {1,2,.,n} into k-clusters {C1, C2,..,Ck} such that:

	1. C_1⋃C_2  …⋃C_k={1,2,..,n}	
	2. C_k∩ C_(k^' )=0 for all k≠k'	
	3. within-cluster variation is minimum
	
The following steps summarizes the k-means clustering algorithm:

	1. Specify the number of clusters (k) to be created. 	
	2. Select randomly k objects from the data-set as the initial cluster centroids.
	3. Re-assign each observation to the cluster whose centroid is closest to that observation (defined using Euclidean distance)
	4. For each of the K clusters, update the cluster centroid (the vector of the p feature means for the observations in that cluster)
	5. Repeat the last two steps until there is only minimal improvement in the minimization function of total within sum of squares.
	
The analysis is carried out using R software, in the following steps. 

