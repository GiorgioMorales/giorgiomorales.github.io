---
published: true
---
## 1. Introduction

Given a set of $n$ **independet variables**: $X_1, X_2, ..., X_n$ and a real-valued **dependant variable** $Y$ (or response variable), the goal of the linear regression problem is to find a **regression function** $f$ such that:

 $$Y = f(X_1, X_2, ..., X_n) + \epsilon = f(X) + \epsilon$$

where $\epsilon$ is the error term, which is independent of $X$, and accounts for the uncertainty inherent in $Y$.

## 2. Linear Regression Model

The regression function $f$ can be expressed based on the multivariate random variable $X$ and its parameters as follows:

> $ f(X) = \beta + \omega_1X_1 + \omega_2X_2 + ... + \omega_nX_n = \beta + \sum_{i=1}^{n}\omega_iX_i = \beta + \omega^T X $

where:

* $\beta$: bias.
* $\omega_i$: regression coefficient or weight for $X_i$.
* $\omega = (\omega_1, \omega_2, ..., \omega_3)^T$ .

**Note:**
* If $n=1$, $f$ represents a line with slope $\omega_i$ and offset $\beta$
* In general: $f$ represents a hyperplane, $\omega$ is the vector notmal to the hyperplane and $\beta$ is the offset.

><div>
<img src="https://www.cs.montana.edu/~moralesluna/images/linear/plane.jpg" width="500"/>
</div>

In practice, the parameters $\beta$ and $\omega$ are **unkown**, so the idea is to estimate them from a training set $D$ consisting of $N$ points $x_i \in \mathbb{R}^n$. Let $b$ denote the estimated value of the bias $\beta$, and $\texttt{w}=(w_1, w_2, ..., w_n)^T$ denote the estimated value of the vector $\omega$; then, the estimated dependant variable given a test point $\texttt{x}=(x_1, x_2, ..., x_n)^T$ can be written as:

$$\hat{y} = b + w_1x_1 + w_2x_2 + ... + w_nx_n = \beta + \texttt{w}^T \texttt{x}$$

## 2 Bivariate Regression

Let's consider the linear regression problem with only one attribute; i.e., the dataset $D$ consists of points with only one dimension:

$$\hat{y}_i = f(x_i) = b + w \cdot x_i$$  

The *residual error* between the estimated value $\hat{y}_i$ and the actual observed response $y_i$ for of the $i$-th data point of $D$ is expressed as:

$$\epsilon_i = y_i - \hat{y}_i$$

Since the objective is to minimize the error of our estimation, we can use the *least squares* method to **minimize the sum of aqueared errors**:

$$\min_{b, w} SSE = \sum_{i=1}^n \epsilon_i^2 = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n (y_i - b - w\cdot x_i)^2$$

In order to solve this objective, we differentiate it w.r.t $b$ and $w$, and set the result to 0:

* w.r.t. $b$:

$$\frac{\partial SSE}{\partial b} = 2 \sum_{i=1}^n (y_i - b - w\cdot x_i)\cdot 1 = 0$$

$$b = \frac{1}{n} \sum_{i=1}^n y_i - w \frac{1}{n} \sum_{i=1}^n x_i$$

$$b = \mu_Y - w \cdot \mu_X$$

* w.r.t. $w$:

$$\frac{\partial SSE}{\partial b} = 2 \sum_{i=1}^n (y_i - b - w\cdot x_i)\cdot x_i = 0$$

$$\sum_{i=1}^n x_iy_i - (\mu_Y \sum_{i=1}^n x_i - w \cdot \mu_X \sum_{i=1}^n x_i) - w \sum_{i=1}^n x_i^2 = 0$$

$$w = \frac{\sum_{i=1}^n (x_i - \mu_X)(y_i - \mu_Y)}{\sum_{i=1}^n(x_i - \mu_X)^2} = \frac{\sigma_{XY}}{\sigma_{X}^2} $$

where $\sigma_{XY}$ is the covariance between $X$ and $Y$, and $\sigma_{X}^2$ is the variance of $X$.

## 2.1. Univariate Regression Example

Since we just found a way to estimate the coefficients $w$ and $b$ for the univariate regression problem, let's get our hands dirty with an example. 

For this, we will use the **Iris dataset**. Specifically, we will try to estimate the width of the petal ($Y$) given only the length of the petal ($X$).

First, let's load the dataset:

{% highlight python %}
import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()

>>
sepal_length	sepal_width	petal_length	petal_width	species
0	5.1	3.5	1.4	0.2	setosa
1	4.9	3.0	1.4	0.2	setosa
2	4.7	3.2	1.3	0.2	setosa
3	4.6	3.1	1.5	0.2	setosa
4	5.0	3.6	1.4	0.2	setosa
{% endhighlight %}

Now, we will separate the $X$ and $Y$ variables:

{% highlight python %}
X = iris['petal_length']
Y = iris['petal_width']

print('Shape of the independent variable: ' + str(X.shape))
print('Shape of the dependant variable: ' + str(Y.shape))

>> 
Shape of the independent variable: (150,)
Shape of the dependant variable: (150,)
{% endhighlight %}

For the sake of visualization, let's plot this set of points:

{% highlight python %}
import matplotlib.pyplot as plt

plt.scatter(X, Y)
plt.xlabel("Petal length")
plt.ylabel("Petal width")
{% endhighlight %}

><div>
<img src="https://www.cs.montana.edu/~moralesluna/images/linear/irispetal.png" width="500"/>
</div>

Now, let{s use the expressions for $w$ and $b$ that we found after least squares minimizatoin:

$$b = \mu_Y - w \cdot \mu_X$$

$$w = \frac{\sigma_{XY}}{\sigma_{X}^2} $$

{% highlight python %}
import numpy as np

# Get number of samples
N = len(X)

# Calculate means
u_x = np.mean(X)
u_y = np.mean(Y)

# Calculate variance an covariance
varx = np.var(X)
varxy = np.sum((X - u_x) * (Y - u_y)) / N

# Calculate parameters
w = varxy / varx
b = u_y - w * u_x

print("Parameter w = " + str(w))
print("Parameter b = " + str(b))  
{% endhighlight %}




