---
title: Blog post - Unraveling the Complexity of Multivariate Systems with Symbolic Regression
summary: Explanation of our ECML-PKDD paper "Univariate Skeleton Prediction in Multivariate Systems Using Transformers"
date: 2024-08-24
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

This post focuses on our paper "[Univariate Skeleton Prediction in Multivariate Systems Using Transformers](/publication/morales-univariate-2024/)".
By leveraging a regression neural network and a novel Multi-Set Transformer model, we process synthetic data to identify these univariate skeletons. Our experimental results demonstrate that this approach not only improves the accuracy of the learned skeletons but also outperforms existing SR methods, including those based on genetic programming and deep learning.

In this blog post, I'll walk you through the key ideas behind our method and its potential implications for SR.

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

In tackling the SR problem, we break it down into simpler, single-variable sub-problems. This approach is a twist on the Symbolic Skeleton Prediction (SSP) problem explored in other research. To illustrate why we deviate from traditional SSP, let's consider an example:

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
$\mathbf{S}^{(s)} \in \mathbb{R}^{n \times d_{in}}$ denotes the $s$-th input of our proposed Multi-Set Transformer such that its $i$-th row, $\mathbf{s}_i^{(s)}$, consists of the concatenation of the input value $x_{v, i}^{(s)}$ and its corresponding output $y_i^{(s)}$; i.e., $\mathbf{s}_i^{(s)} = [x_{v, i}^{(s)}, y_i^{(s)}]$.

Our Multi-Set Transformer has two main parts: an encoder and a decoder. The encoder converts all input sets into a single latent representation \(\mathbf{Z}\). It does this by using an encoder stack \(\phi\) to transform each input set \(\mathbf{S}^{(s)}\) into its own latent representation \(\mathbf{z}^{(s)} \in \mathbb{R}^{d}\) (where \(d\) is the embedding size).

The full encoder, \(\Phi\), generates \(N_S\) individual encodings \(\mathbf{z}^{(1)}, \dots, \mathbf{z}^{(N_S)}\), which are then combined into the final latent representation:

\[
\mathbf{Z} = \Phi ( \mathbf{S}^{(1)}, \dots, \mathbf{S}^{(N_S)}, \boldsymbol{\theta}_e ) = \rho ( \phi ( \mathbf{S}^{(1)}, \boldsymbol{\theta}_e  ) , \dots, \phi ( \mathbf{S}^{(N_S)}, \boldsymbol{\theta}_e  ) )
\]

Here, \(\rho(\cdot)\) is a pooling function, \(\boldsymbol{\theta}_e\) are the trainable weights, and \(\phi\) is a stack of \(\ell\) induced set attention blocks (ISABs) that encode the interactions within each input set in a permutation-invariant way.
Note that $\boldsymbol{\Theta} = [\boldsymbol{\theta}_e, \boldsymbol{\theta}_d]$ contains the weights of the encoder and the decoder.

Figure 4 illustrates the simplified architecture of the Multi-Set Transformer. The decoder \(\psi\) generates sequences based on the representation \(\mathbf{Z}\) produced by \(\Phi\). The output sequence \(\hat{\mathbf{e}} = \{ \hat{e}_1, \dots, \hat{e}_{N_{out}} \}\) represents the skeleton as a sequence of indexed tokens in prefix notation. Each token is mapped to a numerical index using a predefined vocabulary of unique symbols.

For instance, the expression \(\frac{c}{x} e^{\frac{c}{\sqrt{x}}}\) is represented as `{mul, div, c, x, exp, div, c, square, x}` in prefix notation, which is then converted to a sequence of indices like `{"0, 14, 11, 2, 3, 12, 11, 2, 18, 3, 1"}` using the vocabulary.

<div style="display: flex; justify-content: center;">
  <figure style="text-align: center;">
    <img src=Multi-settransformer.jpg alt="figure" width="100%">
    <figcaption>Figure 4: An example of a MSSP problem using the Multi-Set Transformer.</figcaption>
  </figure>
</div>

We train our model on a [large dataset of artificially generated MSSP problems](https://huggingface.co/datasets/AnonymousGM/MultiSetTransformerData), aiming to reduce 
discrepancies between predictions and target skeletons by minimizing cross-entropy loss. 
The training dataset, consisting of expressions in prefix notation, is generated as explained in 
Section B of the [Supplementary Materials](https://github.com/NISL-MSU/MultiSetSR/blob/master/Supplementary%20Material-ECML2024_Skeleton_prediction.pdf).



### Univariate Symbolic Skeleton Prediction

When analyzing the $v$-th variable of the system, $x_v$, we create $N_S$ artificial sets 
${\tilde{\mathbf{X}}^{(1)}, \dots, \tilde{\mathbf{X}}^{(N_S)}}$, where $x_v$ varies while the 
other variables are fixed at random values. The $s$-th set $\tilde{\mathbf{X}}^{(s)}$ consists of 
$n$ samples, with $x_v$ values sampled from a uniform distribution 
$\mathcal{U} (x_v^{\min}, x_v^{\max})$, and the remaining variables sharing constant values across all samples.

In real-world datasets, finding subsets where non-analyzed variables are fixed can be difficult or 
result in small sets. To overcome this, we generate sets with the desired behavior and estimate 
their responses using a trained regression model $\hat{f}$. Our method then derives univariate skeletons 
from these sets, serving as explanations of the function approximated by the model, as shown in Fig. 5.

<div style="display: flex; justify-content: center;">
  <figure style="text-align: center;">
    <img src=step2.jpg alt="figure" width="100%">
    <figcaption>Figure 5: An example of a univariate symbolic skeleton prediction.</figcaption>
  </figure>
</div>

The response for each set $\tilde{\mathbf{X}}^{(s)}$ is estimated as 
$\tilde{\mathbf{y}}^{(s)} = \hat{f}(\tilde{\mathbf{X}}^{(s)})$. The set $\tilde{\mathbf{D}}_v^{(s)} = (\tilde{\mathbf{X}}_v^{(s)}, \tilde{\mathbf{y}}^{(s)})$ is used to analyze the $v$-th variable. The collection of sets $\tilde{\mathbf{D}}_v = { \tilde{\mathbf{D}}_v^{(1)}, \dots, \tilde{\mathbf{D}}_v^{(N_S)} }$ is then fed into the pre-trained Multi-Set Transformer $g$ to estimate the skeleton for $x_v$ as $\tilde{\mathbf{e}}(x_v) = g(\tilde{\mathbf{D}}_v, \boldsymbol{\Theta})$. This process is repeated for all variables to obtain their symbolic skeletons.


## Example Using Pre-defined Datasets

In this example, we will predict the symbolic skeletons corresponding to each variable of a system whose underlying equation is one of the following:

| <div style="width:%;">Eq.</div> | <div style="width:70%;">Underlying equation</div>                                                     | <div style="width:30%;">Domain range</div>                             |
|---------------------------------|-------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| E1                              | $\frac{3.0375 x_1 x_2 + 5.5 \sin \left(\frac{9}{4} (x_1 - \frac{2}{3})(x_2 - \frac{2}{3})\right)}{5}$ | $[-5, 5]^2$                                                            |
| E2                              | $5.5 + \left(1- \frac{x_1}{4}\right)^2 + \sqrt{x_2 + 10} \sin\left(\frac{x_3}{5}\right)$              | $[-10, 10]^2$                                                          |
| E3                              | $\frac{1.5 e^{1.5 x_1} + 5 \cos(3 x_2)}{10}$                                                          | $[-5, 5]^2$                                                            |
| E4                              | $\frac{(1- x_1)^2 + (1- x_3)^2 + 100 (x_2 - x_1^2)^2 + 100 (x_4 - x_3^2)^2}{10000}$                   | $[-5, 5]^4$                                                            |
| E5                              | $\sin(x_1 + x_2 x_3) + \exp{(1.2  x_4)}$                                                              | $x_1 \in [-10, 10], x_2 \in [-5, 5], x_3 \in [-5, 5], x_4 \in [-3, 3]$ |
| E6                              | $\tanh(x_1 / 2) + \text{abs}(x_2) \cos(x_3^2/5)$                                                      | $[-10, 10]^3$                                                                     |  \cos\left(\frac{x_3^2}{5}\right)$ | $[-10, 10]^3$ |
| E7                              | $\frac{1 - x_2^2}{\sin(2 \pi x_1) + 1.5}$                                                             | $[-5, 5]^2$                                                            |
| E8                              | $\frac{x_1^4}{x_1^4 + 1} + \frac{x_2^4}{x_2^4 + 1}$                                                   | $[-5, 5]^2$                                                            |
| E9                              | $\log(2 x_2 + 1) - \log(4 x_1^2 + 1)$                                                                 | $[0, 5]^2$                                                             |
| E10                             | $\sin(x_1 e^{x_2})$                                                                                   | $x_1 \in [-2, 2], x_2 \in [-4, 4]$                                     |
| E11                             | $x_1 \log(x_2^4)$                                                                                     | $[-5, 5]^2$                                                            |
| E12                             | $1 + x_1 \sin\left(\frac{1}{x_2}\right)$                                                              | $[-10, 10]^2$                                                          |
| E13                             | $\sqrt{x_1}\log(x_2^2)$                                                                               | $x_1 \in [0, 20], x_2 \in [-5, 5]$                                     |


To install our Python package, you can execute the following command in the terminal: `pip install git+https://github.com/NISL-MSU/MultiSetSR`.
You can also try to execute this example yourself on Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NISL-MSU/MultiSetSR/blob/master/DemoMSSP.ipynb)

For instance, if we select problem E6, we load it running the following:

```python
from EquationLearning.SymbolicRegressor.MSSP import *

datasetName = 'E6'
data_loader = DataLoader(name=datasetName)
data = data_loader.dataset
```

### Define NN and Load Weights

For this example, we have already trained a feedforward neural network on the generated dataset so we only load their corresponding weights.

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = get_project_root()
folder = os.path.join(root, "EquationLearning//saved_models//saved_NNs//" + datasetName)
filepath = folder + "//weights-NN-" + datasetName
nn_model = NNModel(device=device, n_features=data.n_features, NNtype=data_loader.modelType)
nn_model.loadModel(filepath)
```

### Get Skeletons

The following method will generate some candidate symbolic skeletons and select the most appropriate for each variable:

```python
regressor = MSSP(dataset=data, bb_model=nn_model)
regressor.get_skeletons()
```

```shell
OUTPUT:

********************************
Analyzing variable x0
********************************
Predicted skeleton 1 for variable x0: c*tanh(c*x0) + c
Predicted skeleton 2 for variable x0: c*tanh(c*x0 + c) + c
Predicted skeleton 3 for variable x0: c + tanh(c*x0)
Predicted skeleton 4 for variable x0: c + tanh(c*x0 + c)
Predicted skeleton 5 for variable x0: c*sqrt(c*tanh(c*x0) + c) + c

Choosing the best skeleton... (skeletons ordered based on number of nodes)
	Skeleton: c + tanh(c*x0). Correlation: 0.9997416536770057. Expr: tanh(0.536603*x0)
-----------------------------------------------------------
Selected skeleton: c*tanh(c*x0) + c

********************************
Analyzing variable x1
********************************
Predicted skeleton 1 for variable x1: c*Abs(x1) + c
Predicted skeleton 2 for variable x1: c*x1*tanh(c*x1) + c
Predicted skeleton 3 for variable x1: c + x1*tanh(c*x1 + c)
Predicted skeleton 4 for variable x1: c + x1*tanh(c*x1)
Predicted skeleton 5 for variable x1: c*x1*tanh(c*x1 + c) + c

Choosing the best skeleton... (skeletons ordered based on number of nodes)
	Skeleton: c*Abs(x1) + c. Correlation: 0.9987919523613259. Expr: Abs(x1)
-----------------------------------------------------------
Selected skeleton: c*Abs(x1) + c

********************************
Analyzing variable x2
********************************
Predicted skeleton 1 for variable x2: c*cos(c*(c + x2)**2 + c) + c
Predicted skeleton 2 for variable x2: c*cos(c*x2 + c) + c
Predicted skeleton 3 for variable x2: c*(c*cos(c*x2 + c) + c)**3 + c
Predicted skeleton 4 for variable x2: c*(c*cos(x2) + c)**3 + c
Predicted skeleton 5 for variable x2: c*cos(c*(c + x2)**3 + c) + c

Choosing the best skeleton... (skeletons ordered based on number of nodes)
	Skeleton: c*cos(c*x2 + c) + c. Correlation: 0.45346070794917664. Expr: cos(2.419372*x2 - 6.283185)
	Skeleton: c*cos(x2)**3 + c*cos(x2)**2 + c*cos(x2) + c. Correlation: 0.38832039013792896. Expr: -15.209976*cos(x2)**3 + cos(x2)**2 + 20.0*cos(x2)
	Skeleton: c*cos(c*x2**2 + c*x2 + c) + c. Correlation: 0.9990666723733511. Expr: cos(0.199791*x2**2 - 0.001698*x2 + 0.016807)
-----------------------------------------------------------
Selected skeleton: c*cos(c*x2**2 + c*x2 + c) + c

[c*tanh(c*x0) + c, c*Abs(x1) + c, c*cos(c*x2**2 + c*x2 + c) + c]
```

These results coincide with the targe symbolic skeletons. 
That is, if we apply the skeleton function $\kappa$ to the underlying function $f$ for $x_1$, $x_2$, and $x_3$, we would obtain:
$\kappa(f, x_1) = \mathbf{e}(x_1) = c_1\, \tanh(c_2\,x_1) + c_3$, $\kappa(f, x_2) = \mathbf{e}(x_2) = c_1 |x_2| + c_2$, and 
$\kappa(f, x_3) = \mathbf{e}(x_3) = c_1\, \cos(c_2\, x_3^2 + c_3\,x_3) + c_4$, respectively.  



## Example Using Custom Equations

Here we will show how to use data generated from your own equations. Alternatively, you can bring your dataset (e.g., a CSV file) and load the matrix $X$ (explainable variables) and $Y$ (response variable).

In this example, consider the simple equation $y = \frac{\sin(x_1 + 1.2 \, x_2) \, x_3^2}{2}$. Suppose that $x_1$ and $x_2$ are continuous variables and $x_3$ is discrete and can take 100 possible values ($x_1 \in [-10, 10]$, $x_2 \in [-5, 5]$, and $x_3 \in [-8, ... , 8]$)

### Generate and Format Data

```python
np.random.seed(7)
n = 10000
# Generate data from the equation
x1 = np.random.uniform(-10, 10, size=n)
x2 = np.random.uniform(-5, 5, size=n)
x3 = np.array([np.random.choice(np.linspace(-8, 8, 100)) for _ in range(n)])  # Example of discrete variable
X = np.array([x1, x2, x3]).T
Y = np.sin(x1 + 1.2 * x2) * (x3**2 / 2)  # Or load matrices X and Y from a CSV file

# Format the dataset
names = ['x0', 'x1', 'x2']  # Specify the names of the variables
types = ['continuous', 'continuous', 'discrete']  # Specify if the variables are continuous or discrete
dataset = InputData(X=X, Y=Y, names=names, types=types)
```

### Train a NN

Unlike the previous example, we haven't trained a NN for this problem so let's train it now. If you're not satisfied with the validation MSE, you can try increasing the number of epochs or try a different architecture. By default, we use the `modelType='NN'`; if you need less complexity, try `modelType='NN2'`; or if you need more complexity, try `modelType='NN3'`.

```python
from EquationLearning.Trainer.TrainNNmodel import Trainer

predictor = Trainer(dataset=dataset, modelType='NN')
predictor.train(batch_size=128, epochs=3000, printProcess=False)
# Save the model
# predictor.model.saveModel(path)  # Specify your own path
```

```shell
OUTPUT:

*****************************************
Start MLP training
*****************************************
100%|██████████| 3000/3000 [09:20<00:00,  5.35it/s]Val MSE: 0.07133777567923286
```

### Get Skeletons

The following method will generate some candidate symbolic skeletons and select the most appropriate for each variable:

```python
regressor = MSSP(dataset=dataset, bb_model=predictor.model)
regressor.get_skeletons()
```

```shell
OUTPUT:

********************************
Analyzing variable x0
********************************
Predicted skeleton 1 for variable x0: c*cos(c + x0) + c
Predicted skeleton 2 for variable x0: c*sin(c + x0) + c
Predicted skeleton 3 for variable x0: c*cos(c*x0 + c) + c
Predicted skeleton 4 for variable x0: c*cos(c*x0) + c
Predicted skeleton 5 for variable x0: c*cos(x0) + c

Choosing the best skeleton... (skeletons ordered based on number of nodes)
	Skeleton: c*cos(x0) + c. Correlation: 0.9549434622348927. Expr: cos(x0)
	Skeleton: c*cos(c + x0) + c. Correlation: 0.9998902707120207. Expr: cos(x0 - 5.968739)
-----------------------------------------------------------
Selected skeleton: c*cos(c + x0) + c

********************************
Analyzing variable x1
********************************
Predicted skeleton 1 for variable x1: c*sin(c*x1 + c) + c
Predicted skeleton 2 for variable x1: c*cos(c*x1 + c) + c
Predicted skeleton 3 for variable x1: c*sin(c + x1) + c
Predicted skeleton 4 for variable x1: c*cos(c*x1) + c
Predicted skeleton 5 for variable x1: c*cos(c + x1) + c

Choosing the best skeleton... (skeletons ordered based on number of nodes)
	Skeleton: c*sin(c + x1) + c. Correlation: 0.8109687552029223. Expr: sin(x1 - 5.021683)
	Skeleton: c*cos(c*x1) + c. Correlation: 0.9437250798178028. Expr: cos(1.195566*x1)
	Skeleton: c*cos(c + x1) + c. Correlation: 0.8109687552029744. Expr: cos(x1 + 5.973891)
	Skeleton: c*sin(c*x1 + c) + c. Correlation: 0.9969592793660724. Expr: sin(1.19879*x1 + 1.25925)
	Skeleton: c*cos(c*x1 + c) + c. Correlation: 0.9969591651624546. Expr: cos(1.19903*x1 - 0.311585)
-----------------------------------------------------------
Selected skeleton: c*sin(c*x1 + c) + c

********************************
Analyzing variable x2
********************************
Predicted skeleton 1 for variable x2: c*x2**2 + c
Predicted skeleton 2 for variable x2: c*x2**2 + c*x2 + c
Predicted skeleton 3 for variable x2: c*(c + x2)**2 + c
Predicted skeleton 4 for variable x2: c*x2**2 + c*tanh(c*x2 + c) + c
Predicted skeleton 5 for variable x2: c*x2**3 + c

Choosing the best skeleton... (skeletons ordered based on number of nodes)
	Skeleton: c*x2**2 + c. Correlation: 0.9997445507920887. Expr: x2**2
-----------------------------------------------------------
Selected skeleton: c*x2**2 + c

[c*cos(c + x0) + c, c*sin(c*x1 + c) + c, c*x2**2 + c]
```

Again, these results coincide with the target symbolic skeletons. 
If we apply the skeleton function $\kappa$ to the underlying function $f$ for $x_1$, $x_2$, and $x_3$, we would obtain:
$\kappa(f, x_1) = \mathbf{e}(x_1) = c_1 \, \sin(x_1 + c_2)$, 
$\kappa(f, x_2) = \mathbf{e}(x_2) = c_1 |x_2| + c_2$, and 
$\kappa(f, x_3) = \mathbf{e}(x_3) = c_1\, \cos(c_2\, x_3^2 + c_3\,x_3) + c_4$, respectively.  
Note that although the predicted skeleton for $x_1$, $\hat{\mathbf{e}}(x_1)=c_1\,\cos(c_2 + x_1) + c_3$, does not coincide exactly with $\mathbf{e}(x_1)$.
However, a skeleton with a $\sin$ operator can be rewritten as $c_3 \, \cos(c_4 + x_1)$ (if $c_3 = -c_1$ and $c_4 = c_2 - 90$).
Thus, we consider they express the same functional form, just using different coefficients.

## Conclusions

Symbolic regression (SR) seeks to discover explicit mathematical equations that represent relationships in data, offering a transparent alternative to black-box models. This interpretability is crucial in fields like healthcare, finance, and scientific research, where understanding underlying mechanisms is essential for informed decision-making. SR also plays a vital role in scientific discovery by revealing fundamental laws and relationships governing natural phenomena.

**Objective:** We aim to generate symbolic skeletons that describe the relationship between each variable and the system's response in a multivariate regression problem. To achieve this, we introduced the Multi-Set Skeleton Prediction (MSSP) problem and proposed a novel Multi-Set Transformer model, pre-trained on synthetic symbolic expressions. The multivariate SR problem is then approached as a sequence of MSSP problems using a black-box regression model.

**Interpretability:** Our method serves as an interpretability tool, generating univariate symbolic skeletons that provide transparent insights into the relationships between variables and the system's response. More accurate skeletons lead to clearer interpretations, enhancing our understanding of how each variable influences the system.

**Limitation:** Unlike other methods, our approach does not produce multivariate expressions. 
The focus is on accurately discerning the relationship between individual variables and the system's response. 
Future work will aim to use these **univariate skeletons as building blocks** to create multivariate expressions that approximate the entire system's behavior, with performance evaluated against observed data.
