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
Thus, if we take the gradient w.r.t. $\mathbf{x}$, we obtain:

$$\nabla_{\mathbf{x}} \log p(\mathbf{x}; \beta) = \nabla_{\mathbf{x}} \log \tilde{p}(\mathbf{x}; \beta) - 0.$$

By doing so, we removed the influence of the normalizing constant. 
$\nabla_{\mathbf{x}} \log p(\mathbf{x})$ is known as the **Stein score** (or simply, "score") function. 

## Score Matching

Let $\mathbf{s}_{\mathbf{\theta}}$ represent a _score network_; i.e., a neural network parameterized by $\mathbf{\theta}$.
Instead of training a model to estimate $p_{\text{data}}(\mathbf{x})$ directly, the score network is trained to estimate the score of $p_{\text{data}}$; i.e., 
$\mathbf{s}_{\mathbf{\theta}} (\mathbf{x}) \approx \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})$.
This task is known as _score matching_.

In score matching, the objective is to minimize the difference:

$$\min_{\mathbf{\theta}} \frac{1}{2} \mathbb{E}_{p_{\text{data}}}[ ||\mathbf{s}_{\mathbf{\theta}}(\mathbf{x}) - \nabla_{\mathbf{x}}\log p_{\text{data}}(\mathbf{x}) ||_2^2],$$

which, according to Eq. 1 in Song & Ermon (2019) [1], is equivalent to:

$$ \mathbb{E}_{p_{\text{data}}}[ \text{tr}(\nabla_{\mathbf{x}} \mathbf{s}_{\mathbf{\theta}} (\mathbf{x})) + \frac{1}{2} ||\mathbf{s}_{\mathbf{\theta}} (\mathbf{x})||_2^2  ].$$

This equivalency between both optimization problems was not immediately clear to me, so I started reading more about it.
It turns out that the proof can be found in the Appendix 1 of the
"[Estimation of Non-Normalized Statistical Models by Score Matching](https://jmlr.csail.mit.edu/papers/volume6/hyvarinen05a/old.pdf)" paper by Aapo Hyvärinen (2005) [2].
Nevertheless, this proof didn't seem intuitive enough so I decided to do it in a way I can understand:

We begin by expanding the objective function:

$$\frac{1}{2} \mathbb{E}_{p_{\text{data}}}[ (\mathbf{s}_{\mathbf{\theta}}(\mathbf{x}) - \nabla_{\mathbf{x}}\log p_{\text{data}}(\mathbf{x}))^{\top}(\mathbf{s}_{\mathbf{\theta}}(\mathbf{x}) - \nabla_{\mathbf{x}}\log p_{\text{data}}(\mathbf{x}))  ],$$

$$\frac{1}{2} \mathbb{E}_{p_{\text{data}}}[ ||\mathbf{s}_{\mathbf{\theta}}(\mathbf{x})||_2^2 - 2\, \mathbf{s}_{\mathbf{\theta}}(\mathbf{x})^{\top} \nabla_{\mathbf{x}}\log p_{\text{data}}(\mathbf{x}) + ||\nabla_{\mathbf{x}}\log p_{\text{data}}(\mathbf{x})||_2^2 ]. $$

From this, notice that the last term is not dependent on $\mathbf{\theta}$; thus, it can be ignored during optimization.
The new optimization problem can be written as:

$$\min_{\mathbf{\theta}} J = \min_{\mathbf{\theta}} \frac{1}{2} \mathbb{E}_{p_{\text{data}}}[||\mathbf{s}_{\mathbf{\theta}}(\mathbf{x})||_2^2] - \mathbb{E}_{p_{\text{data}}}[\mathbf{s}_{\mathbf{\theta}}(\mathbf{x})^{\top} \nabla_{\mathbf{x}}\log p_{\text{data}}(\mathbf{x})],$$

or:

[//]: # ([Eq. 1]&#40;#EQ1&#41;)

<a name="EQ1"></a>
$$\begin{equation}
\min_{\mathbf{\theta}} J = \min_{\mathbf{\theta}} \frac{1}{2} \mathbb{E}_{p_{\text{data}}}[||\mathbf{s}_{\mathbf{\theta}}(\mathbf{x})||_2^2] - \mathbf{M}.
\end{equation}$$

{{% callout note %}}
 Before moving on, recall that we can transform the expectation of function $f(\mathbf{x})$ into an integral expression as follows:

 $$\mathbb{E}_{p(\mathbf{x})} [f(\mathbf{x})] = \int f(\mathbf{x})p(\mathbf{x}) d\mathbf{x}.$$
{{% /callout %}}

Applying the same idea to $\mathbf{M}$, we have:

$$\begin{align*}
\mathbf{M} &= \mathbb{E}_{p_{\text{data}}}[\mathbf{s}_{\mathbf{\theta}}(\mathbf{x})^{\top} \nabla_{\mathbf{x}}\log p_{\text{data}}(\mathbf{x})] \\
&= \int \mathbf{s}_{\mathbf{\theta}} (\mathbf{x}) \cdot p_{\text{data}}(\mathbf{x}) \nabla_{\mathbf{x}}\log p_{\text{data}}(\mathbf{x}) d\mathbf{x}
\end{align*}$$

{{% callout note %}}
Recall that applying the gradient to the log of a function can be expanded as:
$$\nabla_{\mathbf{x}} \log f(\mathbf{x}) = \frac{\nabla_{\mathbf{x}}f(\mathbf{x})}{f(\mathbf{x})}$$
$$f(\mathbf{x}) \nabla_{\mathbf{x}} \log f(\mathbf{x}) = \nabla_{\mathbf{x}}f(\mathbf{x})$$
{{% /callout %}}

Then, we can rewrite $\mathbf{M}$ as:

<a name="EQ2"></a>
$$\begin{equation}
\mathbf{M} = \int \mathbf{s}_{\mathbf{\theta}} (\mathbf{x}) \cdot \nabla_{\mathbf{x}} p_{\text{data}}(\mathbf{x}) d\mathbf{x}.
\end{equation}$$

{{% callout note %}}
**Integration by Parts:**
If you have $y = u\, v$ and apply the derivative w.r.t. $x$ ($u$ and $v$ are functions of $x$), you obtain:
$$ \frac{dy}{dx} = u \frac{dv}{dx} + v \frac{du}{dx}.$$
Then, isolating $u \frac{dv}{dx}$ and integrating on both sides of the equation, we have:
$$ \int u \frac{dv}{dx} dx = \int \frac{d(uv)}{dx} dx - \int v \frac{du}{dx}, $$
<a name="EQ3"></a>
$$ \begin{equation} 
\int u v'\, dx = uv - \int v u'\, dx 
\end{equation}$$
{{% /callout %}}

Now, $\mathbf{M}$ looks like the left side of [Eq. 3](#EQ3), isn't it?
To be more clear, consider that $u = \mathbf{s}_{\mathbf{\theta}} (\mathbf{x})$ and $v = p_{\text{data}}(\mathbf{x})$.
With that, let's repeat the "integration by parts" process I explained before but with the new expressions $u$ and $v$.
But first, note that, since $\mathbf{s}_{\mathbf{\theta}} (\mathbf{x}): \mathbb{R}^D \rightarrow \mathbb{R}^D$ and $p_{\text{data}}(\mathbf{x}): \mathbb{R}^D \rightarrow \mathbb{R}$, 
we can treat them as a vector field and a scalar field, respectively.
These definitions will help us to apply the gradient operator correctly.
In particular, recall that if the gradient is applied to a vector field $\mathbf{r}(\mathbf{x}) = \{ r_1(\mathbf{x}), \dots, r_D(\mathbf{x}) \}$, we express it as:
$$\nabla_\mathbf{x} \cdot \mathbf{r}(\mathbf{x}) = \sum_{i=1}^D \frac{\partial r_i(\mathbf{x})}{\partial x_i}.$$
For the sake of notation, let's call $\nabla_\mathbf{x} \cdot$ the divergence operator. 

Now, applying the gradient operator w.r.t. $\mathbf{x}$ to $\mathbf{s}_{\mathbf{\theta}} (\mathbf{x}) \, p_{\text{data}}(\mathbf{x})$ (the product is also a vector field) and applying the product rule, we have:

$$\nabla_{\mathbf{x}} \cdot (\mathbf{s}_{\mathbf{\theta}}(\mathbf{x}) \, p_{\text{data}}(\mathbf{x})) =
(\nabla_{\mathbf{x}} \cdot \mathbf{s}_{\mathbf{\theta}}(\mathbf{x}))\, p_{\text{data}}(\mathbf{x}) +
\mathbf{s}_{\mathbf{\theta}} (\mathbf{x}) \cdot \nabla_{\mathbf{x}} p_{\text{data}}(\mathbf{x}).$$

Then, isolating $\mathbf{s}_{\mathbf{\theta}} (\mathbf{x}) \cdot \nabla_{\mathbf{x}} p_{\text{data}}(\mathbf{x})$:

$$\mathbf{s}_{\mathbf{\theta}}(\mathbf{x}) \cdot \nabla_{\mathbf{x}} p_{\text{data}}(\mathbf{x}) =
  \nabla_{\mathbf{x}} \cdot (\mathbf{s}_{\mathbf{\theta}} (\mathbf{x}) \, p_{\text{data}}(\mathbf{x})) -
(\nabla_{\mathbf{x}} \cdot \mathbf{s}_{\mathbf{\theta}}(\mathbf{x}))\, p_{\text{data}}(\mathbf{x}).$$

Here, we can integrate over the entire input space (i.e., $\mathbb{R}^D$):

$$\begin{align*}
\int_{\mathbb{R}^D} \mathbf{s}_{\mathbf{\theta}} (\mathbf{x}) \cdot \nabla_{\mathbf{x}} p_{\text{data}}(\mathbf{x})\, d\mathbf{x} =
&\int_{\mathbb{R}^D} \left(\nabla_{\mathbf{x}} \cdot (\mathbf{s}_{\mathbf{\theta}} (\mathbf{x}) \, p_{\text{data}}(\mathbf{x})) \right)\, d\mathbf{x} - \\
&\int_{\mathbb{R}^D} (\nabla_{\mathbf{x}} \cdot \mathbf{s}_{\mathbf{\theta}}(\mathbf{x}))\, p_{\text{data}}(\mathbf{x})\, d\mathbf{x}.
\end{align*}
$$

The first term of the right side of the equation, $\int_{\mathbb{R}^D} (\nabla_{\mathbf{x}} \cdot (\mathbf{s}_{\mathbf{\theta}} (\mathbf{x}) \, p_{\text{data}}(\mathbf{x}))) d\mathbf{x}$,
has a particular structure in vector calculus.
So, before moving on, let's talk about the divergence theorem:

{{% callout note %}}
**Divergence Theorem:** The flux of a vector field through a closed surface is equal to the volume integral of the 
divergence of the field over the region enclosed by the surface:
$$\int_{V} (\nabla \cdot \mathbf{F}) \, dV = \int_{S} \mathbf{F} \cdot \mathbf{n} \, dS,$$
where:
* $V$ is the volume in $\mathbb{R}^D$,
* $S$ is the boundary (or surface) of $V$,
* $\mathbf{F}$ is a continuously differentiable vector field,
* $\mathbf{n}$ is the outward-pointing unit normal vector on $\partial V$.
{{% /callout %}}

It may seem unclear why to bring this theorem here considering we're not dealing with surfaces or volumes.
However, let's pretend we do for a minute. 
The product $(\mathbf{s}_{\mathbf{\theta}} (\mathbf{x}) \, p_{\text{data}}(\mathbf{x}))$ is a vector field whose flux 
passes through a surface enclosing a volume that is given by $\mathbb{R}^D$.
In this context, according to the divergence theorem, the sum of the divergences inside $\mathbb{R}^D$ (i.e., $\int_{\mathbb{R}^D} \left( \nabla_{\mathbf{x}} \cdot (\mathbf{s}_{\mathbf{\theta}} (\mathbf{x}) \, p_{\text{data}}(\mathbf{x})) \right) \, d\mathbf{x}$) 
is equal to the outward flux that crosses the surface of the volume $\mathbb{R}^D$.
Since the "volume" $\mathbb{R}^D$ is infinite, its surface is located at the boundary $||\mathbf{x} \rightarrow \infty||$.
Then:

$$\lim_{||\mathbf{x} \rightarrow \infty||} \mathbf{s}_{\mathbf{\theta}} (\mathbf{x}) \, p_{\text{data}}(\mathbf{x}) = 0.$$

In other words, the outward flux when $||\mathbf{x} \rightarrow \infty||$ is negligible, which implies that all divergences are cancelled inside $\mathbb{R}^D$ and 

$$\int_{\mathbb{R}^D} (\nabla_{\mathbf{x}} \cdot (\mathbf{s}_{\mathbf{\theta}} (\mathbf{x}) \, p_{\text{data}}(\mathbf{x}))) d\mathbf{x} = 0.$$

Therefore:

<a name="EQ4"></a>
$$\begin{equation}
\int_{\mathbb{R}^D} \mathbf{s}_{\mathbf{\theta}} (\mathbf{x}) \cdot \nabla_{\mathbf{x}} p_{\text{data}}(\mathbf{x})\, d\mathbf{x} = - \int_{\mathbb{R}^D} (\nabla_{\mathbf{x}} \cdot \mathbf{s}_{\mathbf{\theta}}(\mathbf{x}))\, p_{\text{data}}(\mathbf{x})\, d\mathbf{x}.
\end{equation}
$$

Then, replacing [Eq. 4](#EQ4) in [Eq. 2](#EQ2):

$$\begin{equation}
\mathbf{M} = - \int_{\mathbb{R}^D} (\nabla_{\mathbf{x}} \cdot \mathbf{s}_{\mathbf{\theta}}(\mathbf{x}))\, p_{\text{data}}(\mathbf{x})\, d\mathbf{x}.
\end{equation}$$

(...post in construction)


## References

[1] Y. Song and S. Ermon, “Generative modeling by estimating gradients of the data distribution,” in Proceedings of the 33rd International Conference on Neural Information Processing Systems (2019), pp. 11918–11930.

[2] H. Aapo, “Estimation of non-normalized statistical models by score matching.” Journal of Machine Learning Research (2005), vol. 6, pp. 695-809.





