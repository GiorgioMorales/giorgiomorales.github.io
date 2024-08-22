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
These are simplified versions of the function where we replace specific numbers with placeholders (e.g. turning $3x^2 +e^{2x} -4)$ into something like $c_1\,x^2 + e^{c_2\, x} + c_3$). 
In particular, $\kappa(\cdot)$ represent a skeleton function that replaces the numerical constants of a given symbolic expression by placeholders $c_i$.

Our goal is to figure out these skeletons for each variable, $\hat{\mathbf{e}}(x_1),\dots, \hat{\mathbf{e}}(x_t)$, to better understand how each one affects the system’s response. 
Next, we’ll walk through the steps our method uses to achieve this.