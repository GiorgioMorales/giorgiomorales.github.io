---
title: Blog post - Score matching for score estimation
summary: Obtaining the proof for the score matching objective function 
date: 2024-12-09
authors:
  - admin
tags:
  - Blog
  - Generative modeling
  - GenAI
---

I have recently started reading some **seminal generative modeling papers** thanks to the recommendation of Mehdy Bennani (PhD student at Université de Caen Normandie).
Since this is a new research area for me, I am committed to studying these papers thoroughly to develop a solid understanding of the subject. 

However, as is the case with any area in Machine Learning that has evolved progressively over the years, some equations, theorems, or concepts are presented with the assumption that the reader is already familiar with them. 
A new reader may accept them as having an inherent value of truth and move on, or they may ask "Why" and start going down the rabbit hole.
While trying to escape the rabbit hole, I'll share some of the eye-opening things I encountered along the way, and explain them in a way that makes sense to me
(so that I'll be able to remember when I read these posts in the future).


In this post, I'll start with the "[Generative Modeling by Estimating Gradients of the Data Distribution](https://dl.acm.org/doi/10.5555/3454287.3455354)" paper by Song & Ermon (2019) [1].
As indicated in the paper's abstract, this work presents a "generative model where samples are produced via Langevin dynamics using gradients of the data distribution estimated with score matching."
Rather than explaining the paper itself, I'll limit myself to describe what I didn't know and/or catched my attention.

## (Stein) Score

Let's start with some notation. 
We're given a dataset $\{ \mathbf{x}_i \in \mathbb{R}^D \}_{i=1}^N$ whose samples come from an _unknown_ data distribution $p_{\text{data}}(\mathbf{x})$.

Now, consider a general distribution $p(\mathbf{x})$.
If this distribution was parametrized by a set of parameters $\beta$, we would express it as $p(\mathbf{x}; \beta)$.
We could try to use Maximum Likelihood Estimation (MLE) to find the parameters $\beta$ so that $p(\mathbf{x}; \beta)$ approximates $p_{\text{data}}(\mathbf{x})$.
The challenge is that, in many settings, we know the un-normalized distribution $\tilde{p}(\mathbf{x}; \beta)$ but not $p(\mathbf{x}; \beta)$.
To calculate it, we need to normalize $\tilde{p}(\mathbf{x}; \beta)$ as follows:

$$p(\mathbf{x}; \beta) = \frac{\tilde{p}(\mathbf{x}; \beta)}{\int_{\mathcal{X}} \tilde{p}(\mathbf{x}; \beta)dx} = \frac{\tilde{p}(\mathbf{x}; \beta)}{Z_{\beta}}.$$

Note that $Z_{\beta}$ may be simple to calculate when dealing with simple probability functions (e.g., Gaussian); however, its calculation
could be intractable when the $\tilde{p}(\mathbf{x}; \beta)$ is complex, high-dimensional, or its integral doesn't have a closed-form solution.
Instead of doing that, let's apply the $\log$ operation to the previous equation:

$$\log p(\mathbf{x}; \beta) = \log \tilde{p}(\mathbf{x}; \beta) - \log Z_{\beta}.$$

Since $Z_{\beta}$ was integrated over all possible values of $\mathbf{x}$, it's independent of $\mathbf{x}$.
Thus, if we take the gradient w.r.t $\mathbf{x}$, we obtain:

$$\nabla_{\mathbf{x}} \log p(\mathbf{x}; \beta) = \nabla_{\mathbf{x}} \log \tilde{p}(\mathbf{x}; \beta) - 0.$$

By doing so, we removed the influence of the normalizing constant. 
Hence, $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ is known as the **Stein score** (or simply "score") function. 

## Score Matching

Let $\mathbf{s}_{\mathbf{\theta}}$ represent a _score network_; i.e., a neural network parameterized by $\mathbf{\theta}$.
Instead of training a model to estimate $p_{\text{data}}(\mathbf{x})$ directly, the score network is trained to estimate the score of $p_{\text{data}}$; i.e., 
$\mathbf{s}_{\mathbf{\theta}} (\mathbf{x}) \approx \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})$.
This task is known as _score matching_.

In score matching, the objective is to minimize the difference:

$$\min_{\mathbf{\theta}} \frac{1}{2} \mathbb{E}_{p_{\text{data}}}[ ||\mathbf{s}_{\mathbf{\theta}}(\mathbf{x}) - \nabla_{\mathbf{x}}\log p_{\text{data}}(\mathbf{x}) ||_2^2],$$

which, according to Eq. 1 in Song & Ermon (2019) [1], is equivalent to:

$$ \mathbb{E}_{p_{\text{data}}}[ \tr(\nabla_{\mathbf{x}} \mathbf{s}_{\mathbf{\theta}} (\mathbf{x})) + \frac{1}{2} ||\mathbf{s}_{\mathbf{\theta}} (\mathbf{x})||_2^2  ].$$

This equivalency between both optimization problems was not immediately clear to me, so I started reading more about it.
It turns out that the proof can be found in the Appendix 1 of the
"[Estimation of Non-Normalized Statistical Models by Score Matching](https://jmlr.csail.mit.edu/papers/volume6/hyvarinen05a/old.pdf)" paper by Aapo Hyvärinen (2005) [2].
Nevertheless, this proof didn't seem intuitive enough so I decided to do it in a way I can understand:

We begin by expanding the objective function:

$$\frac{1}{2} \mathbb{E}_{p_{\text{data}}}[ (\mathbf{s}_{\mathbf{\theta}}(\mathbf{x}) - \nabla_{\mathbf{x}}\log p_{\text{data}}(\mathbf{x}))^{\top}(\mathbf{s}_{\mathbf{\theta}}(\mathbf{x}) - \nabla_{\mathbf{x}}\log p_{\text{data}}(\mathbf{x}))  ],$$

$$\frac{1}{2} \mathbb{E}_{p_{\text{data}}}[ ||\mathbf{s}_{\mathbf{\theta}}(\mathbf{x})||_2^2 - 2 \mathbf{s}_{\mathbf{\theta}}(\mathbf{x})^{\top} \nabla_{\mathbf{x}}\log p_{\text{data}}(\mathbf{x}) + ||\nabla_{\mathbf{x}}\log p_{\text{data}}(\mathbf{x})||_2^2 ]. $$

From this, notice that the last term is not dependent on $\mathbf{\theta}$; thus, it can be ignored during optimization.
The new optimization problem can be written as:

$$\min_{\mathbf{\theta}} J = \min_{\mathbf{\theta}} \frac{1}{2} \mathbb{E}_{p_{\text{data}}}[||\mathbf{s}_{\mathbf{\theta}}(\mathbf{x})||_2^2] - \mathbb{E}_{p_{\text{data}}}[\mathbf{s}_{\mathbf{\theta}}(\mathbf{x})^{\top} \nabla_{\mathbf{x}}\log p_{\text{data}}(\mathbf{x})] = \min_{\mathbf{\theta}} \frac{1}{2} \mathbb{E}_{p_{\text{data}}}[||\mathbf{s}_{\mathbf{\theta}}(\mathbf{x})||_2^2] - M.$$

(...post in construction)


## References

[1] Y. Song and S. Ermon, “Generative modeling by estimating gradients of the data distribution,” in Proceedings of the 33rd International Conference on Neural Information Processing Systems (2019), pp. 11918–11930.
[2] H. Aapo, “Estimation of non-normalized statistical models by score matching.” Journal of Machine Learning Research (2005), vol. 6, pp. 695-809.




