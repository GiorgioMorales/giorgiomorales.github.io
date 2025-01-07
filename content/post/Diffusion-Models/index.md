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
    <figcaption>Directed graphical model.</figcaption>
  </figure>
</div>

## Evidence Lower Bound (ELBO)


## References

[1] J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” in Advances in Neural Information Processing Systems, 2020, pp. 6840–6851. 

[2] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep Unsupervised Learning using Nonequilibrium Thermodynamics,” in Proceedings of the 32nd International Conference on Machine Learning, PMLR, Jun. 2015, pp. 2256–2265. 


