---
title: Blog post - Unraveling the Complexity of Multivariate Systems with Symbolic Regression
summary: Explanation of our ECML-PKDD paper "Univariate Skeleton Prediction in Multivariate Systems Using Transformers"
date: 2024-08-23
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

You can find our paper "Univariate Skeleton Prediction in Multivariate Systems Using Transformers" [here](https://giorgiomorales.github.io/publication/morales-univariate-2024/).
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

<div style="display: flex; justify-content: center;">
  <figure style="text-align: center;">
    <img src="step1.jpg" alt="figure" width="100%">
    <figcaption>Figure 1: Neural network training.</figcaption>
  </figure>
</div>


## Multi-Set Symbolic Skeleton Prediction

In tackling the symbolic regression (SR) problem, we break it down into simpler, single-variable sub-problems. This approach is a twist on the Symbolic Skeleton Prediction (SSP) problem explored in other research. To illustrate why we deviate from traditional SSP, let's consider an example:

Imagine a function $y = \frac{x_1}{\log (x_1^2 + x_2)}$. If we focus on the relationship between $x_1$ and $y$, while keeping $x_2$ constant, the behavior of the function can vary depending on the value of $x_2$. 
As shown in Fig. 2, different fixed values of $x_2$ lead to different function behaviors. 
This variability can make it tricky for SSP solvers to generate a consistent functional form.

<div style="display: flex; justify-content: center;">
  <figure style="text-align: center;">
    <img src="curves.jpg" alt="figure" width="100%">
    <figcaption>Figure 2: $x_1$ vs. $y$ curves when $x_2=4.45$, $0.2$, and $1.13$.</figcaption>
  </figure>
</div>

Moreover, fixing some variables might push the function into a space where its form is hard to recognize, especially if the range of the variable we're analyzing is limited. To improve SSP, we can introduce additional context by using multiple sets of input-response pairs, each created by fixing the non-analyzed variables to different values.

The idea here is to process all these sets together to generate a skeleton that captures the common structure across all input sets. 
We call this problem **Multi-Set Symbolic Skeleton Prediction (MSSP)** and it's depicted in Fig. 3.

<div style="display: flex; justify-content: center;">
  <figure style="text-align: center;">
    <img src=MSSP.jpg alt="figure" width="100%">
    <figcaption>Figure 3: An example of an MSSP problem.</figcaption>
  </figure>
</div>

More formally, let's say we have a dataset with $N_R$ input-response pairs $( \mathbf{X}, \mathbf{y})$, where $\mathbf{X}$ represents the inputs, and $\mathbf{y}$ represents the responses. If we're interested in how the $v$-th variable $x_v$ relates to the response $y$, we create a collection of $N_S$ sets, denoted as $\mathbf{D} = {\mathbf{D}^{(1)}, \dots, \mathbf{D}^{(N_S)} }$. Each set $\mathbf{D}^{(s)}$ contains $n$ pairs $( \mathbf{X}_v^{(s)}, \mathbf{y}^{(s)} )$, where the non-analyzed variables are fixed at different values.

These sets can be created either by sampling from the dataset or by generating new data points using a regression model if the dataset isn't large enough. Each set represents the relationship between $x_v$ and $y$ under different conditions. Although these relationships, denoted as $f^{(1)}(x_v),\dots, f^{(N_S)}(x_v)$, are derived from the same overall function $f(\mathbf{x})$, they differ due to the varying fixed values.

The key is that these functions should share a common symbolic skeleton, even if their coefficients differ. Applying a skeleton function $\kappa(\cdot)$ to each $f^{(s)}(x_v)$ should give us the same target skeleton $\mathbf{e}(x_v)$, with placeholders for the constants.

So, the MSSP problem involves processing the collection $\mathbf{D}$ to generate a skeleton $\hat{\mathbf{e}}(x_v)$ that approximates the true skeleton $\mathbf{e}(x_v)$. 


### Multi-Set Transformer

Our approach to solving the MSSP problem is inspired by the [Set Transformer](https://proceedings.mlr.press/v97/lee19d/lee19d.pdf), an attention-based neural network derived from the transformer model. 
The Set Transformer is designed for handling set-input problems, making it capable of processing input sets of different sizes while maintaining permutation invariance. 
We've adapted this model into a **Multi-Set Transformer**, tailoring it to the specific needs of our research.

Let the \( s \)-th input set be \(\mathbf{D}^{(s)} = ( \mathbf{X}_v^{(s)}, \mathbf{y}^{(s)} ) = \{ (x_{v, i}^{(s)}, y_i^{(s)} ) \}_{i=1}^n\).
Our Multi-Set Transformer has two main parts: an encoder and a decoder. The encoder converts all input sets into a single latent representation \(\mathbf{Z}\). It does this by using an encoder stack \(\phi\) to transform each input set \(\mathbf{S}^{(s)}\) into its own latent representation \(\mathbf{z}^{(s)} \in \mathbb{R}^{d}\) (where \(d\) is the embedding size).

The full encoder, \(\Phi\), generates \(N_S\) individual encodings \(\mathbf{z}^{(1)}, \dots, \mathbf{z}^{(N_S)}\), which are then combined into the final latent representation:

\[
\mathbf{Z} = \Phi ( \mathbf{S}^{(1)}, \dots, \mathbf{S}^{(N_S)}, \boldsymbol{\theta}_e ) = \rho ( \phi ( \mathbf{S}^{(1)}, \boldsymbol{\theta}_e  ) , \dots, \phi ( \mathbf{S}^{(N_S)}, \boldsymbol{\theta}_e  ) )
\]

Here, \(\rho(\cdot)\) is a pooling function, \(\boldsymbol{\theta}_e\) are the trainable weights, and \(\phi\) is a stack of \(\ell\) induced set attention blocks (ISABs) that encode the interactions within each input set in a permutation-invariant way.

Figure 4 illustrates the simplified architecture of the Multi-Set Transformer. The decoder \(\psi\) generates sequences based on the representation \(\mathbf{Z}\) produced by \(\Phi\). The output sequence \(\hat{\mathbf{e}} = \{ \hat{e}_1, \dots, \hat{e}_{N_{out}} \}\) represents the skeleton as a sequence of indexed tokens in prefix notation. Each token is mapped to a numerical index using a predefined vocabulary of unique symbols.

For instance, the expression \(\frac{c}{x} e^{\frac{c}{\sqrt{x}}}\) is represented as `{mul, div, c, x, exp, div, c, square, x}` in prefix notation, which is then converted to a sequence of indices like `{"0, 14, 11, 2, 3, 12, 11, 2, 18, 3, 1"}` using the vocabulary.

<div style="display: flex; justify-content: center;">
  <figure style="text-align: center;">
    <img src=Multi-settransformer.jpg alt="figure" width="100%">
    <figcaption>Figure 4: An example of a MSSP problem using the Multi-Set Transformer.</figcaption>
  </figure>
</div>


...post in construction...


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NISL-MSU/MultiSetSR/blob/master/DemoMSSP.ipynb)
