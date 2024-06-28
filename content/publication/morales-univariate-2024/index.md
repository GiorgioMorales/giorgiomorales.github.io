---
title: Univariate Skeleton Prediction in Multivariate Systems Using Transformers
authors:
- Giorgio Morales
- John W. Sheppard
date: '2024-06-01'
publishDate: '2024-06-28T19:23:08.317271Z'
publication_types:
- paper-conference
publication: '*Machine Learning and Knowledge Discovery in Databases*'
abstract: Symbolic regression (SR) methods attempt to learn mathematical expressions
  that approximate the behavior of an observed system. However, when dealing with
  multivariate systems, they often fail to identify the functional form that explains
  the relationship between each variable and the system's response. To begin to address
  this, we propose an explainable neural SR method that generates univariate symbolic
  skeletons that aim to explain how each variable influences the system's response.
  By analyzing multiple sets of data generated artificially, where one input variable
  varies while others are fixed, relationships are modeled separately for each input
  variable. The response of such artificial data sets is estimated using a regression
  neural network (NN). Finally, the multiple sets of input-response pairs are processed
  by a pre-trained Multi-Set Transformer that solves a problem we termed Multi-Set
  Skeleton Prediction and outputs a univariate symbolic skeleton. Thus, such skeletons
  represent explanations of the function approximated by the regression NN. Experimental
  results demonstrate that this method learns skeleton expressions matching the underlying
  functions and outperforms two GP-based and two neural SR methods.
tags:
- Symbolic regression
- Transformers
- Knowledge discovery
links:
- name: URL
  url: http://arxiv.org/abs/2406.17834
---
