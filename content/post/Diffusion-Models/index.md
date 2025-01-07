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
p_{\theta} (\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mathbf{\mu}_{\theta}(\mathbf{x}_t), \Sigma_{\theta}(\mathbf{x}_t), t).
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
\mathbb{E}[- \log p_{\theta} (\mathbf{x}_0)] \leq &\mathbb{E}_q[- \log \frac{p_{\theta} (\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)}], \\
& = \mathbb{E}_q[- \log p(\mathbf{x}_T) - \sum_{t \geq 1} \log \frac{p_{\theta} (\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_t | \mathbf{x}_{t-1})}] =: L. 
\end{align*}
\end{equation}$$

But of course, the question is "Why?".
So, in this section, we'll derive these expressions ourselves.
We begin by applying the logarithm to [Eq. 1](#EQ1), and multiplying and dividing the right side by $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$:

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
\log \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[\frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\right] = 
\geq \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[\log \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\right].
. 
\end{equation*}$$

Therefore:

<a name="EQ9"></a>

$$\begin{equation*}
-\log p_{\theta} (\mathbf{x}_0)
\leq \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[\log \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\right].
. 
\end{equation*}$$


Post in progress

## References

[1] J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” in Advances in Neural Information Processing Systems, 2020, pp. 6840–6851. 

[2] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep Unsupervised Learning using Nonequilibrium Thermodynamics,” in Proceedings of the 32nd International Conference on Machine Learning, PMLR, Jun. 2015, pp. 2256–2265. 


