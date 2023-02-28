---
layout: distill
title: Neural Tangent Kernel 
description: notes on NTK and related material
giscus_comments: false
date: 2023-02-26

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Functional Gradient Descent
  - name: Functionals
    subsections:
    - name: Reproducing Kernel Hilbert Space (RKHS)
    - name: Inner product and norm
    - name: Reproducing property
    - name: Evaluation functional 
    - name: Functional derivative
    - name: Chain rule 
  - name: Gaussian Processes (GP)
  - name: Neural Tangent Kernel
    subsections:
    - name: Setting
    - name: Random functions approximation
    - name: NTK
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
---
## Functional Gradient Descent
For examples and plots see [here](https://simple-complexities.github.io/optimization/functional/gradient/descent/2020/03/04/functional-gradient-descent.html).

Suppose we want to learn a function $$ f(x) $$ using gradient descent. One example would be to parameterize $$ f $$ as a linear function using weights $$ w $$: $$ f(x) = w^\top x $$. 

In order to learn $$ w $$ we take a loss function, i.e. MSE: 

$$
\begin{align}
L(w) &= \sum_{i=1}^n (y_i - w^\top x_i)^2 + \| w \|^2 \\
\nabla L(w) &= \sum_{i=1}^n -2 x_i (y_i - w^\top x_i) + 2w \\
&= \sum_{i=1}^n (-2x_iy_i + 2w^\top \| x_i \|^2) + 2w
\end{align}
$$. 

Then we do gradient descent steps on $w$: 

$$
w = w - \alpha \nabla L(w)
$$

Using functional gradient descent, this can be generalized to any function $$ f $$: 

$$
\begin{align}
L(f) &= \sum_{i=1}^n (y_i - f(x_i)) + \| f \|^2 \\
f_{t+1} &= f_t - \alpha \nabla L(f)
\end{align}
$$

This has 2 advantages:
- Some loss functions are non-convex in parameter space but can be convex in functional space
- In NTK, when width $$ \rightarrow \infty $$, the weights become almost static between GD steps; however we can still study the function trajectory during GD in functional space

***

## Functionals
Some basic functional notions are needed to understand this post.

### Reproducing Kernel Hilbert Space (RKHS)
Denote RKHS of (fixed) kernel $$ k $$ by $$ \mathcal{H_k} $$, $$ k(\cdot, \cdot) $$ is a kernel function and $$ K_{ij} = k(x_i, x_j) $$. Then:

$$
f \in \mathcal{H_k} \implies f(\cdot) = \sum_{i=1}^n \beta_i k(x_i, \cdot), \beta_i \in \mathbb{R}
$$

In other words, $$ f $$ is in RKHS if it can be written as a weighted sum of kernel functions evaluated over $$ n $$ points. Note that $$ f $$ is completely determined by the $$ \beta_i $$ and $$ x_i $$.

### Inner product and norm
Let $$ f, g \in \mathcal{H_k} $$. Then:

$$
\begin{align}
f \cdot g &= \sum_{i=1}^{n_f} \sum_{j=1}^{n_g} \alpha_i \beta_j k(x_i, x_j) = \alpha K^\top \beta \\
\| f \|^2 &= f \cdot f = \alpha K^\top \beta
\end{align}
$$

### Reproducing property

$$
f \cdot k(x, \cdot) = \sum_{i=1}^n \beta_i [k(x_i, \cdot) k(x, \cdot)] = \sum_{i=1}^n \beta_i k(x_i, x) = f
$$

### Evaluation functional

$$
E_x[f] = f(x)
$$

### Functional derivative
From the definition of the derivative, the functional derivative is the coefficient of the linear term in the Taylor expansion of the functional.

Example 1 - for the functional $$ E_x[f] = f(x) $$:

$$
\begin{align}
E_x[f + df] &= f(x) + df(x) \\
&= E_x[f] + df(x) \\
&= E_x[f] + k(x, \cdot) \cdot df \\
&\implies \nabla E_x[f] = k(x, \cdot)
\end{align}
$$

Example 2 - for the functional $$ E[f] = \| f \|^2 $$

$$
\begin{align}
E[f + df] &= (f + df) (f + df) \\
&= \| f \| + 2 f \cdot df + \| df \|^2 \\
&= E[f] + 2 f \cdot df + \| df \|^2 \\ 
&\implies \nabla E[f] = 2f
\end{align}
$$

### Chain rule
Let $$ E[f] $$ be a functional and $$ g : \mathbb{R} \to \mathbb{R} $$. Then:

$$
\nabla g(E[f]) = \nabla E[f] g'(E[f])
$$

## Gaussian Processes (GP)
This is a (very) short and handwavy recap on GPs, but it suffices to show the relevant connections to NTK. 

Let $$ Y \sim \mathcal{N(0, \Sigma_y)} $$ be the training set and $$ X \sim \mathcal{N(0, \Sigma_x)} $$ be the test set (we can always substract the mean to center them in 0). We setup the prior distribution $$ P_X $$ such that $$ \Sigma_x^{ij} = K(X_i, X_j) $$. We also have the joint distribution 

$$
P_{X, Y} = \begin{bmatrix} X \\ Y \end{bmatrix} \sim \mathcal{N}(0, \begin{bmatrix} \Sigma_{xx} & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_{yy}\end{bmatrix})
$$

where each covariance is setup similarly via the kernel $$ K $$. 
Then, due to the property that the Normal distribution is closed under conditioning and marginalization, we can compute the posterior 

$$
P_{X|Y}
$$

From the posterior we can sample multiple functions that fit our training data. 

## Neural Tangent Kernel
- We know that at initialization, when taking width $$ \to \infty $$, ANNs behave like GPs 
- NTK proves that this is also true during training, where the GP kernel used in the training is the "Neural Tangent Kernel"
- The NTK stays constant during training when taking width $$ \to \infty $$

In the following derivations, I will **not use biases**.

### Setting
The following setting is used in the paper:

$$
\begin{align}
f_\theta(x) &= \tilde{\alpha}^{(L)}(x) \\
\alpha^{(0)}(x) &= x \\
\tilde{\alpha}^{(l+1)}(x) &= \frac{1}{\sqrt{n_l}} W^{(l)} \alpha^{(l)}(x) \\
\alpha^{(l)} &= \sigma(\tilde{\alpha}^{(l)}(x))
\end{align}
$$
where $$ W_{ij}^{(l)} \sim \mathcal{N}(0, \frac{1}{n_l}) $$. 


Define the bilinear form: 

$$ 
\langle f, g  \rangle_{p^{in}} = \mathbb{E}_{x \sim p^{in}}[ f(x)^\top g(x)]
$$

where $$ p^{in} $$ is the distribution of the input set (assume empirical distribution over $$ N $$ points). Similarly, 

$$
\langle f, g  \rangle_{K} = \mathbb{E}_{x, x' \sim p^{in}}[ f(x)^\top K(x, x') g(x')]
$$

. 

Also, let $$ F^{(L)}: \mathbb{R}^P \to \mathcal{F} $$ be the realization function, which maps parameters $$ \theta $$ to a function $$ f_\theta $$ (basically takes in parameters and returns a function parameterized by these parameters), and $$ \nabla_{W_{ij}^{(l)}}F^{(L)} $$ be the derivative of the realization function w.r.t. the weights. Define also $$ \mu : \mathcal{F} \to \mathbb{R}, \mu = \langle d, \cdot \rangle_{p^{in}}, d \in \mathcal{F} $$. Plugging in $$ d = K_{i, \cdot}(x, \cdot) $$ in the previous definition (since $$ K_{i, \cdot}(x, \cdot) \in \mathcal{F} $$), we get:

$$
\begin{align}
f_{\mu, i}(x) = \langle d, K_{i, \cdot}(x, \cdot) \rangle
\end{align}
$$

Instead of doing gradient descent on the parameters $$ \theta $$ (which we will see stay almost constant during training as the width $$ \to \infty $$), we do functional gradient descent on the function $$ f_\theta $$ itself, using a cost $$ C : \mathcal{F} \to \mathbb{R} $$.

Define the functional derivative of $$ C $$ at a point 

$$
f_0 \in \mathcal{F}$$ as $$\nabla_fC|_{f_0}
$$

and the dual 

$$
d|_{f_0} \in \mathcal{F}
$$

such that 

$$
\nabla_fC|_{f_0} = \langle d|_{f_0}, \cdot \rangle_{p^{in}}
$$

.

***

### Random functions approximation
Before moving onto ANNs, this is a simplification in which the realization function is linear.

Let $$ P $$ random functions $$ f^{(p)} $$ define a random linear parametrization: 

$$
\begin{align}
F^{lin} = f_\theta^{lin} = \frac{1}{\sqrt{P}} \sum_{p=1}^P \theta_p f^{(p)}
\end{align}
$$

The partial derivatives are then:

$$
\begin{align}
\nabla_{\theta_p} F^{lin}(\theta) &= \frac{1}{\sqrt{P}} f^{(p)} \\
\nabla_{t} F^{lin}(\theta(t)) &= \frac{1}{\sqrt{P}} \sum_{p=1}^P \nabla_t \theta_p(t) f^{(p)} \\
\end{align}
$$

Writing down the gradient descent step on the parameters:

$$
\begin{align}
\theta_p(t+dt) &= \theta_p(t) - dt\nabla_{\theta_p}(C \circ F^{lin})|_{\theta(t)} \\
\implies \nabla_t \theta_p(t) &= -\nabla_{\theta_p}(C \circ F^{lin})|_{\theta(t)} \\
&= -\nabla_{\theta_p}F^{lin} \nabla_{F^{lin}}C|_{f_{\theta(t)}^{lin}} \\
&= -(\frac{1}{\sqrt{P}} f^{(p)}) \nabla_{F^{lin}}C|_{f_{\theta(t)}^{lin}} \\
&= -\frac{1}{\sqrt{P}} \langle d|_{f_{\theta(t)}^{lin}}, f^{(p)} \rangle_{p^{in}} 
\end{align}
$$

Plugging in the above 2 equations we get the evolution of the function $$ f_{\theta(t)}^{lin} $$ in function space through GD:

$$
\begin{align}
\nabla_{t} f^{lin}_{\theta(t)} &= -\frac{1}{P} \sum_{p=1}^P \langle d|_{f_{\theta(t)}^{lin}}, f^{(p)} \rangle_{p^{in}} f^{(p)}   \\
&= - \nabla_{\tilde{K}}C|_{f_\theta^{lin}}
\end{align}
$$

where $$ \tilde{K} $$ is the tangent kernel (by the definition in the paper in section 3). 

In conclusion, parameter gradient descent on $$ C \circ F^{lin} $$ is equivalent to functional gradient descent in the function space with the tangent kernel $$ \tilde{K} $$.

### NTK
Similar to above, in ANNs, the network function evolution is:

$$
\begin{align}
\nabla_t f_{\theta(t)} &= - \nabla_{\Theta^{(L)}C|_{f_{\theta(t)}}} \\
\Theta^{L}(\theta) &= \sum_{p=1}^P \nabla_{\theta_p}F^{(L)}(\theta) \otimes \nabla_{\theta_p}F^{(L)}(\theta)
\end{align}
$$

where $$ \Theta^{L}(\theta) $$ is the NTK.

For the following statements, check proof in the paper. 

#### At initialization
Recall that $$ f_\theta(x) $$ is the preactivation of the final layer in an $$ L $$ layer deep ANN and $$ f_{\theta, k}(x), k=1,\dots,n_L $$ are the neurons in the final layer preactivation. At initialization, when taking the width of each layer to infinity, i.e. $$ n_1, \dots n_{L-1} \to \infty $$, the neurons $$ f_{\theta,k} $$ converge to GPs, with covariance $$ \Sigma_L $$, defined recursively by:

$$
\begin{align}
\Sigma^{(1)}(x, x') &= \frac{1}{n_0}x^\top x' \\
\Sigma^{(L+1)}(x, x') &= \mathbb{E}_{f \sim \mathcal{N}(0, \Sigma^{(L)})}[\sigma(f(x))\sigma(f(x'))]
\end{align}
$$

Note that, the above result showed that **each of the output neurons** converges to a GP, each with its own covariance matrix. However, the stronger result showed in the paper is that under the same conditions and the same limit, **all of the covariance matrices** converge to a deterministic NTK kernel $$ \Theta^{(L)} $$ (see definition in the paper under Theorem 1).

#### During training
It is also true that the NTK stays asymptotically constant during training, under the infinite width regime.
Note that, during the infinite width regime, while the individual variation of each weight entry is small, the total variation is large, allowing the lower layers to learn.