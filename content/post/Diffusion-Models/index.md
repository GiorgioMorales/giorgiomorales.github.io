---
title: Blog post - Diffusion Models Background
summary: Deriving the standard loss function of a difussion model
date: 2025-01-07
authors:
  - admin
tags:
  - Blog
  - Generative modeling
  - GenAI

featured: true

image:
  caption: 'Image Credits: Meta AI'
  focal_point: ""
  preview_only: true
---

In my previous post, ["Score matching for score estimation"](/post/score-matching/), I promised to analyze the "[Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)" paper [1] by Ho, Jain, and Abbeel (2020).
The paper's main contribution is the formalization of the connection between diffusion probabilistic models and denoising score matching with Langevin dynamics.
Based on this strategy, they presented pivotal experiments for high-quality image synthesis.

After reading it carefully, I decided to focus mainly on its Background section; that is, to talk about the basics of **Difussion Models**.
The remainder of the paper, which is definitely interesting and important, is well explained and relatively straightforward.
So, in fairness, I'll effectively be reviewing the ["Deep Unsupervised Learning using Nonequilibrium Thermodynamics"](https://proceedings.mlr.press/v37/sohl-dickstein15.html) paper [2] by Sohl-Dickstein et al. (2015). 

## Diffusion Model

For the sake of completeness, in this section, we just reproduce the definitions and notations given by Ho, Jain, and Abbeel (2020).
A diffusion model is a latent variable model represented by [Eq. 1](#EQ1):

<a name="EQ1"></a>

$$\begin{equation}
p_{\theta} (\mathbf{x}_0) := \int p_{\theta}(\mathbf{x}_{0:T}) d\, \mathbf{x}_{1:T},
\end{equation}$$

where $\mathbf{x}_0 \sim q(\mathbf{x}_0)$ represents the data and $\mathbf{x}_1, \dots, \mathbf{x}_T$ is a sequence of $T$ latent variables.
$p_{\theta}(\mathbf{x}_{0:T})$, **reverse process**, is a Markov chain whose transitions are Gaussian and have to be learned 
($\theta$ represents the set of learning parameters):

<a name="EQ2"></a>

$$\begin{equation}
p_{\theta}(\mathbf{x}_{0:T}) := p(\mathbf{x}_T) \prod_{t=1}^T p_{\theta} (\mathbf{x}_{t-1} | \mathbf{x}_t),
\end{equation}$$

where $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I})$ and:

<a name="EQ3"></a>

$$\begin{equation}
p_{\theta} (\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_{\theta}(\mathbf{x}_t, t), \Sigma_{\theta}(\mathbf{x}_t, t)).
\end{equation}$$

On the other hand, the **forward (or difussion) process**, $q(\mathbf{x}_{1:T} | \mathbf{x}_0)$, consists of a known Markov chain 
that gradually adds Gaussian noise according to a schedule $\beta_1, \dots, \beta_T$:

<a name="EQ4"></a>

$$\begin{equation}
q(\mathbf{x}_{1:T} | \mathbf{x}_0) := \prod_{t=1}^T q(\mathbf{x}_t | \mathbf{x}_{t-1}),
\end{equation}$$

and

<a name="EQ5"></a>

$$\begin{equation}
q(\mathbf{x}_t | \mathbf{x}_{t-1}) := \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}),
\end{equation}$$

The framework is depicted in the figure below. 

<div style="display: flex; justify-content: center;">
  <figure style="text-align: center;">
    <img src=graph.jpg alt="figure" width="100%">
    <figcaption>Directed graphical model [1].</figcaption>
  </figure>
</div>

## Evidence Lower Bound (ELBO)

The training of a diffusion model consists of optimizing the evidence (or variational) lower bound as follows:

<a name="EQ6"></a>

$$\begin{equation}
\begin{align*}
\mathbb{E}[- \log p_{\theta} (\mathbf{x}_0)] &\leq \mathbb{E}_q \left[- \log \frac{p_{\theta} (\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \right], \\
& = \mathbb{E}_q \left[- \log p(\mathbf{x}_T) - \sum_{t \geq 1} \log \frac{p_{\theta} (\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_t | \mathbf{x}_{t-1})} \right] =: L. 
\end{align*}
\end{equation}$$

But of course, the question is "Why?".
So, in this section, we'll derive these expressions ourselves.
We begin by applying the logarithm to [Eq. 1](#EQ1).
Then, multiply and divide the right-hand side by $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$:

<a name="EQ7"></a>

$$\begin{equation}
\begin{align*}
\log p_{\theta} (\mathbf{x}_0) &= \log \int p_{\theta}(\mathbf{x}_{0:T}) d\, \mathbf{x}_{1:T}, \\
 &= \log \int \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} q(\mathbf{x}_{1:T}|\mathbf{x}_0) d\, \mathbf{x}_{1:T}. 
\end{align*}
\end{equation}$$

{{% callout note %}}
 Recall that we can transform the expectation of function $f(\mathbf{x})$ into an integral expression as follows:

 $$\mathbb{E}_{p(\mathbf{x})} [f(\mathbf{x})] = \int f(\mathbf{x})p(\mathbf{x}) d\mathbf{x}.$$
{{% /callout %}}

Thus, we can take [Eq. 7](#EQ7) and write it as:

<a name="EQ8"></a>

$$\begin{equation}
\log p_{\theta} (\mathbf{x}_0) = \log \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[\frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\right].
\end{equation}$$

{{% callout note %}}
 We'll use Jensen's inequality, which states that given a concave function $f$ and a random variable $x$, the following holds:

 $$f(\mathbb{E}[x]) \geq \mathbb{E}[f(x)].$$
{{% /callout %}}

Applying Jensen's inequality to the right-hand side of [Eq. 8](#EQ8), we have:

$$\begin{equation*}
\log \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[\frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\right] 
\geq \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[\log \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\right].
\end{equation*}$$

Therefore:

<a name="EQ9"></a>

$$\begin{equation*}
-\log p_{\theta} (\mathbf{x}_0)
\leq \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[\log \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\right].
\end{equation*}$$

This is close to first part of the expression we want to demonstrate. 
Now let's apply the expectation over the true data distribution $p_{data}(\mathbf{x}_0)$ to both sides of [Eq. 9](#EQ9):

<a name="EQ10"></a>

$$\begin{equation}
\mathbb{E}[-\log p_{\theta} (\mathbf{x}_0)]
\leq \mathbb{E}_{p_{data}(\mathbf{x}_0)} \left[\mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[\log \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\right]\right].
\end{equation}$$

{{% callout note %}}
 **Law of total expectation (LTE)**:

 $$\mathbb{E}_{p(B)}[\mathbb{E}_{q(A|B)}[f(A, B)]] \geq \mathbb{E}_{q(A,B)}[f(A, B)].$$
{{% /callout %}}

Applying LTE to [Eq. 10](#EQ10) and denoting $q(\mathbf{x}_0, \mathbf{x}_{1:T}) = q(\mathbf{x}_{0:T})$ and $\mathbb{E}_{q(\mathbf{x}_{0:T}|\mathbf{x}_0)} [\dots] = \mathbb{E}_{q}[\dots]$, we obtain:

$$\begin{equation}
\mathbb{E}[-\log p_{\theta} (\mathbf{x}_0)]
\leq \mathbb{E}_q \left[\log \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\right],
\end{equation}$$

which demonstrates the first inequality of [Eq. 6](#EQ6) $\square$.

---

Now, we need to demonstrate the second part of [Eq. 6](#EQ6).
To do this, we replace $p_{\theta} (\mathbf{x}_{0:T})$ and $q(\mathbf{x}_{1:T} | \mathbf{x}_0)$ using [Eq. 2](#EQ2) and [Eq. 4](#EQ4), respectively:

$$\begin{equation*}
\mathbb{E}_q \left[- \log \frac{p_{\theta} (\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \right]
= \mathbb{E}_q \left[- \log \left( \frac{ p(\mathbf{x}_T) \prod_{t=1}^T p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_t)}{ \prod_{t=1}^T q( \mathbf{x}_t | \mathbf{x}_{t-1}) } \right) \right].
\end{equation*}$$

Then, by splitting the log of divisions into subtractions and writing the log of multiplications as a sum, we get:

<a name="EQ11"></a>

$$\begin{equation}
\begin{align*}
\mathbb{E}_q [- \log &\frac{p_{\theta} (\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} ] \\ 
&= \mathbb{E}_q [- \log p(\mathbf{x}_T) - \sum_{t \geq 1} \log \frac{p_{\theta} (\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_t | \mathbf{x}_{t-1})} ],
\end{align*}
\end{equation}$$

which demonstrates the second part of [Eq. 6](#EQ6) $\square$.


## Loss Function Expansion

In Eq. 5 of [1], the loss function $L$ is further reduced as:

<a name="EQ12"></a>

$$\begin{equation}
\begin{align*}
L = \mathbb{E}_q [ &\underbrace{ D_{\text{KL}} (q(\mathbf{x}_T | \mathbf{x}_0)) || p(\mathbf{x}_T))}_{L_T} + \\
&\sum_{t>1} \underbrace{D_{\text{KL}} ( q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) || p_{\theta} (\mathbf{x}_{t-1} | \mathbf{x}_t) )}_{L_{t-1}} - \\
&\underbrace{\log p_{\theta}(\mathbf{x}_0 | \mathbf{x}_1)}_{L_0} ].
\end{align*}
\end{equation}$$

In this section, we'll derive the expression given in [Eq. 12](#EQ12).
It turns out that the proof of this is given in [2] and in Appendix A of [1].
However, I'll take a more detailed approach.

We start from [Eq. 11](#EQ11).
We isolate the case when $t=1$ so that:

<a name="EQ13"></a>

$$\begin{equation}
L = \mathbb{E}_q \left[- \log p(\mathbf{x}_T) - \sum_{t > 1} \log \frac{p_{\theta} (\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_t | \mathbf{x}_{t-1})} 
 -\log \frac{p_{\theta} (\mathbf{x}_0 | \mathbf{x}_1)}{q(\mathbf{x}_1 | \mathbf{x}_0)} \right].
\end{equation}$$

We want to reverse the terms in $q(\mathbf{x}_t | \mathbf{x}_{t-1})$; that is, instead of having a function of $\mathbf{x}_t$ conditioned on $\mathbf{x}_{t-1}$, we'd like 
to have a function of $\mathbf{x}_{t-1}$ conditioned on $\mathbf{x}_t$.
So, let's note that, because the forward process is a Markov chain, $\mathbf{x}_t$ depends only on $\mathbf{x}_{t-1}$.
Thus, we can express the following:

<a name="EQ14"></a>

$$\begin{equation}
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0).
\end{equation}$$

{{% callout note %}}
 Bayes' theorem using three variables:

 $$P(A | B, C) = \frac{P(A, B, C)}{P(B, C)} = \frac{P(B | A, C) P(A,C)}{P(B, C)}$$
 $$= \frac{P(B | A, C) P(A|C) P(C)}{P(B, C) P(C)} = \frac{P(B | A, C) P(A|C)}{P(B, C)}.$$
{{% /callout %}}

Then, rewriting [Eq. 14](#EQ14) using Bayes' theorem, we get:

<a name="EQ15"></a>

$$\begin{equation}
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) = \frac{q(\mathbf{x}_{t-1} | \mathbf{x}_{t}, \mathbf{x}_0) q(\mathbf{x}_{t} | \mathbf{x}_0) }{ q(\mathbf{x}_{t - 1}| \mathbf{x}_0)}.
\end{equation}$$

Combining [Eq. 15](#EQ15) and [Eq. 13](#EQ13), we get:

$$\begin{equation*}
\begin{align*}
L = &\mathbb{E}_q [- \log p(\mathbf{x}_T) - \\
&\underbrace{\sum_{t > 1} \log \frac{p_{\theta} (\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_{t-1} | \mathbf{x}_{t}, \mathbf{x}_0)} \frac{q(\mathbf{x}_{t - 1}| \mathbf{x}_0)}{q(\mathbf{x}_{t} | \mathbf{x}_0)}}_M \\
&-\log \frac{p_{\theta} (\mathbf{x}_0 | \mathbf{x}_1)}{q(\mathbf{x}_1 | \mathbf{x}_0)} ].
\end{align*}
\end{equation*}$$

From the previous, we simplify $M$:

$$\begin{equation*}
\begin{align*}
M &= \sum_{t > 1} \log \frac{p_{\theta} (\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_{t-1} | \mathbf{x}_{t}, \mathbf{x}_0)} \frac{q(\mathbf{x}_{t - 1}| \mathbf{x}_0)}{q(\mathbf{x}_{t} | \mathbf{x}_0)} \\
&= \sum_{t > 1} \left( \log \frac{p_{\theta} (\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_{t-1} | \mathbf{x}_{t}, \mathbf{x}_0)} + \log q(\mathbf{x}_{t - 1}| \mathbf{x}_0) - \log q(\mathbf{x}_{t} | \mathbf{x}_0) \right) \\
&= \sum_{t > 1} \left( \log \frac{p_{\theta} (\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_{t-1} | \mathbf{x}_{t}, \mathbf{x}_0)} \right) + (\log q(\mathbf{x}_{1}| \mathbf{x}_0) - \log q(\mathbf{x}_{2} | \mathbf{x}_0)) \\
&\;\;\;\;+ (\log q(\mathbf{x}_{2}| \mathbf{x}_0) - \log q(\mathbf{x}_{3} | \mathbf{x}_0)) \\
&\;\;\;\;+ \dots + (\log q(\mathbf{x}_{T-1}| \mathbf{x}_0) - \log q(\mathbf{x}_{T} | \mathbf{x}_0)) \\
&= \sum_{t > 1} \left( \log \frac{p_{\theta} (\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_{t-1} | \mathbf{x}_{t}, \mathbf{x}_0)} \right) + \log q(\mathbf{x}_{1}| \mathbf{x}_0) - \log q(\mathbf{x}_{T} | \mathbf{x}_0)
\end{align*}
\end{equation*}$$

Replacing $M$ in $L$ and rearranging the log operations:

<a name="EQ16"></a>

$$\begin{equation}
\begin{align*}
L = &\underbrace{\mathbb{E}_q [- \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_{T} | \mathbf{x}_0)}}_S - \\
&\sum_{t > 1} \log \frac{p_{\theta} (\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_{t-1} | \mathbf{x}_{t}, \mathbf{x}_0)} \\
&-\log p_{\theta} (\mathbf{x}_0 | \mathbf{x}_1) ].
\end{align*}
\end{equation}$$

Considering the LTE, we can rewrite the expectation in $S$ as:

$$\begin{equation*}
\begin{align*}
S &= \mathbb{E}_q \left[ \mathbb{E}_{q(\mathbf{x}_T | \mathbf{x}_0)} \left[- \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_{T} | \mathbf{x}_0)} \right]\right]\\
& = \mathbb{E}_q \left[ \mathbb{E}_{q(\mathbf{x}_T | \mathbf{x}_0)} \left[ \log \frac{q(\mathbf{x}_{T} | \mathbf{x}_0)}{p(\mathbf{x}_T)} \right]\right]
\end{align*}
\end{equation*}$$

{{% callout note %}}
 **Kullback-Leibler (KL) divergence**: For distributions $A$ and $B$.

 $$D_{\text{KL}}(A||B) = \mathbb{E}_A \left[ \log \frac{A}{B} \right].$$
{{% /callout %}}

Therefore, $S$ is expressed in terms of a KL divergence:

$$\begin{equation*}
S = \mathbb{E}_q \left[ D_{\text{KL}}(q(\mathbf{x}_T | \mathbf{x}_0) ||p(\mathbf{x}_T) ) \right].
\end{equation*}$$

We can apply the same idea to the remaining elements in [Eq. 16](#EQ16) to obtain:

$$\begin{equation*}
\begin{align*}
L = \mathbb{E}_q [ &\underbrace{ D_{\text{KL}} (q(\mathbf{x}_T | \mathbf{x}_0)) || p(\mathbf{x}_T))}_{L_T} + \\
&\sum_{t>1} \underbrace{D_{\text{KL}} ( q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) || p_{\theta} (\mathbf{x}_{t-1} | \mathbf{x}_t) )}_{L_{t-1}} - \\
&\underbrace{\log p_{\theta}(\mathbf{x}_0 | \mathbf{x}_1)}_{L_0} ] \;\;\;\;\; \square.
\end{align*}
\end{equation*}$$

## Mean and Variance of the Forward Process

Now, we will derive the expressions for $q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \boldsymbol{\mu}_t, \mathbf{\Sigma}_t)$ 
and $q(\mathbf{x}_{t-1} |\mathbf{x}_t, \mathbf{x}_0)= \mathcal{N}(\mathbf{x}_t; \tilde{\boldsymbol{\mu}}_t, \tilde{\mathbf{\beta}}_t \mathbf{I})$.

We have parameterized $q(\mathbf{x}_t | \mathbf{x}_{t-1})$ in [Eq. 5](#EQ5) as a normal distribution dependent on $\beta_t$.
Using the Markov property, the mean at $\mathbf{x}_t$ is a product of all the scaling terms up to $t$:

<a name="EQ17"></a>

$$\begin{equation}
\begin{align*}
\boldsymbol{\mu}_t &= \sqrt{(1-\beta_t)(1-\beta_{t-1})\dots (1-\beta_{1})} \mathbf{x}_0 \\
&= \sqrt{\prod_{s=1}^t(1-\beta_s)} \mathbf{x}_0 = \sqrt{\bar{\alpha}_t} \mathbf{x}_0.
\end{align*}
\end{equation}$$

{{% callout note %}}
Note that the variance is unaffected by constant scaling factors and that independent noise terms add their variances

 $Var(c\,A) = c^2 Var(A)$ and 
 
$Var(A+B) = Var(A) + Var(B)$.
{{% /callout %}}

Considering that, due to the assumed normal distribution, $\mathbf{x}_t$ can be expressed as 
$\mathbf{x}_t = \sqrt{1-\beta_t} \mathbf{x}_{t-1} + \epsilon_t$, where $\epsilon_t \sim \mathcal{N}(0, \beta_t \mathbf{I})$. 
Therefore:

$$\begin{equation*}
\begin{align*}
\mathbf{\Sigma}_t &= Var(\sqrt{1 - \beta_t} \mathbf{x}_{t-1} + \epsilon_t | \mathbf{x}_0) \\
&= (1 - \beta_t) Var(\mathbf{x}_{t-1} | \mathbf{x}_{0}) + \beta_t \mathbf{I}.
\end{align*}
\end{equation*}$$

Using recursion and expanding the expression:
$$\begin{equation*}
\begin{align*}
\mathbf{\Sigma}_t &= (1 - \beta_t) [ (1 - \beta_{t-1}) \mathbf{\Sigma}_{t-2} + \beta_t \mathbf{I}. ] + \beta_t \mathbf{I}, \\
&= (1 - \beta_t)(1 - \beta_{t-1}) \mathbf{\Sigma}_{t-2} + \underbrace{\beta_{t-1} (1 - \beta_t) - (1 - \beta_t) \mathbf{I} + \mathbf{I}}_{\mathbf{I} - (1 - \beta_t)(1 - \beta_{t-1})},
\end{align*}
\end{equation*}$$
    
Take into account that the generative process starts from $\mathbf{x}_{0}$, which is assumed to be a known (fixed) data point from the dataset.
Thus, its variance is zero ($ \mathbf{\Sigma}_0 = 0$).
Then, in general, we have:

<a name="EQ18"></a>

$$\begin{equation}
\begin{align*}
\mathbf{\Sigma}_t &= (\prod_{s=1}^t(1 - \beta_s)) \mathbf{\Sigma}_{0} + (1 - \prod_{s=1}^t(1 - \beta_s))\mathbf{I},\\
\mathbf{\Sigma}_t &= (1 - \bar{\alpha}_t)\mathbf{I}.
\end{align*}
\end{equation}$$

From [Eq. 17](#EQ17) and [Eq. 18](#EQ18), we obtain a new expression for $q(\mathbf{x}_t | \mathbf{x}_0)$:

<a name="EQ19"></a>

$$\begin{equation}
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}\left (\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I} \right).
\end{equation}$$

---

Now, we can continue deriving an expression for $q(\mathbf{x}_{t-1} |\mathbf{x}_t, \mathbf{x}_0)$.
To do this, we'll use $q(\mathbf{x}_t | \mathbf{x}_{t-1})$ and the expression we derived for $q(\mathbf{x}_t | \mathbf{x}_0)$ ([Eq. 19](#EQ19)).
We know that both terms are Gaussian; thus, their joint distribution must be Gaussian too:

<a name="EQ20"></a>

$$\begin{equation}
\begin{align*}
&q(\mathbf{x}_t, \mathbf{x}_{t-1} | \mathbf{x}_0) = \\
&\mathcal{N} \left( 
\begin{bmatrix} 
\mathbf{x}_t \\ 
\mathbf{x}_{t-1} 
\end{bmatrix}; 
\begin{bmatrix} 
\sqrt{\bar{\alpha}_t} \mathbf{x}_0 \\ 
\sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0 
\end{bmatrix}, 
\begin{bmatrix} 
(1 - \bar{\alpha}_t) \mathbf{I} & C \\ 
C & (1 - \bar{\alpha}_{t-1}) \mathbf{I} 
\end{bmatrix} 
\right),
\end{align*}
\end{equation}$$

where the covariance term $C$ would be determined by:

<a name="EQ21"></a>

$$\begin{equation}
\begin{align*}
C = Cov(\mathbf{x}_t, \mathbf{x}_{t-1}) &=  \mathbb{E} [(\mathbf{x}_t - \underbrace{\mathbb{E}[\mathbf{x}_t]}_0)(\mathbf{x}_{t-1} - \underbrace{\mathbb{E}[\mathbf{x}_{t-1}]}_0)^{\top}] ,\\
&= \mathbb{E} [ (\sqrt{\alpha_t} \mathbf{x}_{t-1} + \sqrt{\beta_t} \epsilon_t)(\mathbf{x}_{t-1})^{\top}],\\
&= \mathbb{E} [ \sqrt{\alpha_t} \mathbf{x}_{t-1}\mathbf{x}_{t-1}^{\top}] + \underbrace{\mathbb{E} [\sqrt{\beta_t} \epsilon_t \mathbf{x}_{t-1})^{\top}]}_0,\\
&= \sqrt{\alpha_t} \mathbb{E} [ \mathbf{x}_{t-1}\mathbf{x}_{t-1}^{\top}],\\
&= \sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})\mathbf{I}.
\end{align*}
\end{equation}$$

{{% callout note %}}
General formula for the conditional distribution of a multivariate Gaussian:
 
$p(x_A|x_B) = \mathcal{N}(x_B; \mu_B + C_{BA} C_{AA}^{-1} (x_A - \mu_A), \Sigma_B - C_{BA}C_{AA}^{-1}C_{AB})$.
{{% /callout %}}

Combining the previous with [Eq. 20](#EQ20) and [Eq. 21](#EQ21), we have:

$$\begin{equation*}
\begin{align*}
\tilde{\boldsymbol{\mu}}_t &= \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})\mathbf{I}}{(1 - \bar{\alpha}_{t})\mathbf{I}} (\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0),\\
&= \underbrace{\frac{\sqrt{\bar{\alpha}_{t-1}} (1 - \bar{\alpha}_{t}) - \sqrt{\alpha_{t}}(1 - \bar{\alpha}_{t-1}) \sqrt{\bar{\alpha}_t}}{1 - \bar{\alpha}_t} \mathbf{x}_0}_H + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_{t})} \mathbf{x}_t.
\end{align*}
\end{equation*}$$

Let's simplify $H$:

$$\begin{equation*}
\begin{align*}
H &= \sqrt{\bar{\alpha}_{t-1}} (\frac{1 - \bar{\alpha}_t - \sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1}) \sqrt{\bar{\alpha}_t} }{1 - \bar{\alpha}_t}) \mathbf{x}_0,\\
&= \sqrt{\bar{\alpha}_{t-1}} \frac{1 - \bar{\alpha}_t - \alpha_{t} + \bar{\alpha}_t}{1 - \bar{\alpha}_t} \mathbf{x}_0,\\
&= \sqrt{\bar{\alpha}_{t-1}} \frac{1 - \alpha_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 = \sqrt{\bar{\alpha}_{t-1}} \frac{\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0.
\end{align*}
\end{equation*}$$

Replacing $H$ in $\tilde{\boldsymbol{\mu}}_t$, we obtain the mean expression we were looking for:

$$\begin{equation*}
\tilde{\boldsymbol{\mu}}_t = \sqrt{\bar{\alpha}_{t-1}} \frac{\beta_{t}}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_{t})} \mathbf{x}_t. 
\end{equation*}$$

Finally, we need to the determine the variance expression for $q(\mathbf{x}_{t-1} |\mathbf{x}_t, \mathbf{x}_0)$; i.e., $\tilde{\mathbf{\beta}}_t \mathbf{I}$.
Replacing in [Eq. 20](#EQ20) what we know so far, we get the following:

$$\begin{equation*}
\begin{align*}
\tilde{\mathbf{\beta}}_t \mathbf{I} &= (1 - \bar{\alpha}_{t-1}) \mathbf{I} - \sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})\mathbf{I} \frac{1}{(1 - \bar{\alpha}_t) \mathbf{I}} \sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})\mathbf{I}, \\
&= \frac{(1 - \bar{\alpha}_{t-1})(1 - \bar{\alpha}_{t}) - \alpha_t(1 - 2 \bar{\alpha}_{t-1} + \alpha^2_{t-1}) }{(1 - \bar{\alpha}_{t})}\mathbf{I}, \\
&= \frac{ 1 - (1 + \alpha_t) \bar{\alpha}_{t-1} + \alpha_t \bar{\alpha}_{t-1}^2 - \alpha_t + 2\alpha_t \bar{\alpha}_{t-1} - \alpha_t \bar{\alpha}_{t-1} ^2 }{(1 - \bar{\alpha}_{t})}\mathbf{I}, \\
&= \frac{1 - \alpha_t - (1 - \alpha_t) \bar{\alpha}_{t-1}}{(1 - \bar{\alpha}_{t})}\mathbf{I}, \\
\end{align*}
\end{equation*}$$

In conclusion:

$$\begin{equation}
\tilde{\mathbf{\beta}}_t \mathbf{I} = \frac{(1 - \bar{\alpha}_{t-1})} {(1 - \bar{\alpha}_{t})} \beta_t \mathbf{I}.
\end{equation}$$


The next paper I'll analyze is "[Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS)" by Yang et al. (2021).  


## References

[1] J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” in Advances in Neural Information Processing Systems, 2020, pp. 6840–6851. 

[2] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep Unsupervised Learning using Nonequilibrium Thermodynamics,” in Proceedings of the 32nd International Conference on Machine Learning, PMLR, Jun. 2015, pp. 2256–2265. 


