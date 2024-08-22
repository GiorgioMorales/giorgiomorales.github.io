---
title: Blog post - Unraveling the Complexity of Multivariate Systems with Symbolic Regression
summary: Explanation of our ECML-PKDD paper "Univariate Skeleton Prediction in Multivariate Systems Using Transformers"
date: 2024-08-22
authors:
  - admin
tags:
  - Blog
  - Symbolic Regression
  - XAI

featured: true

image:
  caption: 'Generated with Meta AI'
  focal_point: ""
  preview_only: false
---

In the realm of data-driven modeling, symbolic regression (SR) stands out as a powerful tool for discovering mathematical equations that describe underlying system behaviors.
Unlike traditional methods, SR doesn't assume a specific model structure; instead, it explores a wide range of possible equations, making it both flexible and interpretable.
But here's the catch: when dealing with complex, multivariate systems, SR often struggles to pinpoint the exact relationships between each independent variable and the system's response.

That's where our research comes in.
We developed a new approach to SR that generates univariate symbolic skeletons—abstract representations of mathematical expressions that capture the structural relationship between variables and responses.
Our method focuses on isolating and modeling these relationships individually, which allows for a more accurate understanding of each variable's influence.

By leveraging a regression neural network and a novel Multi-Set Transformer model, we process synthetic data to identify these univariate skeletons. Our experimental results demonstrate that this approach not only improves the accuracy of the learned skeletons but also outperforms existing SR methods, including those based on genetic programming and deep learning.

In this blog post, I'll walk you through the key ideas behind our method, its potential implications for SR, and how it pushes the boundaries of what we can achieve in explainable AI.

# Multivariate Skeleton Prediction

Let’s consider a system where the output, $y$,depends on $t$ variables $\textbf{x}=\{ x_1, \dots, x_t \}$.
Essentially, there’s an underlying function $f$ that maps these variables to the output, so we have $y = f(\mathbf{x}) = f(x_1, \dots, x_t)$. 

Now, imagine we want to understand how each individual variable $x_i$ (like $x_1$, $x_2$, etc.) contributes to the output $y$.
Instead of just looking at the whole function $f$, we break it down and focus on what we call "skeleton functions."
These are simplified versions of the function where we replace specific numbers with placeholders.
These skeletons can be obtained using the skeleton function $\kappa(\cdot)$, which replaces the numerical constants of a given symbolic expression by placeholders $c_i$; e.g., $\kappa (3x^2 +e^{2x} -4) = c_1\,x^2 + e^{c_2\, x} + c_3$. 

Our goal is to figure out these skeletons for each variable, $\hat{\mathbf{e}}(x_1),\dots, \hat{\mathbf{e}}(x_t)$, to better understand how each one affects the system’s response. 
Next, we’ll walk through the steps our method uses to achieve this.

## Neural Network Training

To approximate the function $f$ based on observed data, we use a regression model. Suppose we have a dataset $\textbf{X}= \{ \textbf{x}_1, \dots , \textbf{x}_{N_R} \}$ with $N_R$ samples, where each sample is represented as $\textbf{x}_j = \{ x_{j,1}, \dots, x_{j,t} \}$, and the corresponding target values are $\textbf{y}= \{ y_1, \dots , y_{N_R} \}$.

We build a neural network (NN) regression model, denoted as $\hat{f}(\cdot; \boldsymbol{\theta}_{NN})$, where $\boldsymbol{\theta}_{NN}$ represents the network’s weights. The network learns to capture the relationship between the input $\textbf{X}$ and the targets $\textbf{y}$.

For any given input $\textbf{x}_j$, the model estimates the target as $ \hat{y}_j = \hat{f}(\textbf{x}_j) $.
The network’s parameters $\boldsymbol{\theta}_{NN}$ are optimized by minimizing the mean squared error (MSE) between the predicted values and the actual target values:
$ \boldsymbol{\theta}_{NN}^* = \; \text{argmin}_{\boldsymbol{\theta}_{NN}} \ \frac{1}{{N_R}} \sum_{j=1}^{N_R} (\hat{y}_{j} - y_{j})^2 $.

We chose a neural network for this task because of its ease of training and high accuracy, though other regression methods could also be used.







[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NISL-MSU/MultiSetSR/blob/master/DemoMSSP.ipynb)
