---
layout: distill
title: Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent
description: notes on Lee et. al
giscus_comments: false
date: 2023-02-27

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Introduction
  - name: Theoretical results
    subsections:
      - name: Setting
      - name: Linearized networks
      - name: GPs from GD training
      - name: Infinite width networks are linearized networks
# if a section has subsections, you can add them as follows:
# subsections:
#   - name: Example Child Subsection 1
#   - name: Example Child Subsection 2
---

## Introduction

In the previous posts, we saw that:
1) NTK - gradient descent is parameter space is equivalent to functional gradient descent in function space, following the Neural Tangent Kernel
2) Under the infinite width limit, both single layer and multilayer networks are equivalent to a GP with a kernel defined using a recursive formula

In this paper, the authors show that for wide neural networks (i.e. infinite width), the models can be linearized into their first-order Taylor expansion, around the parameters $w_0$ at initialization. In addition, they show that gradient-based training with MSE produces test set predictions drawn from a GP with a particular kernel.

Main insight: when width $\to \infty$, models can be replaced by their first-order Taylor expansion, around $\theta_0$ (the params at initialization).  This approximation holds well under finite width too.

More concretely, a deep neural network with infinite width can be simplified to its linearized model. Under MSE, the dynamics become an ODE that can be solved in close form.

***

## Theoretical results

### Setting

Note the abuse of notation: $\nabla_t f := \frac{df}{dt} = \dot{f}$. In the paper they use the differential instead of the derivative in writing the statements below (I divided by the learning rate $\eta$, where $\eta \to 0$).

Setting is as in the previous posts, weights and biases are Gaussian distributed. The notation is:

$$
\begin{align}
h^{l+1} &= x^l W^{l+1} + b^{l+1} \\
x^{l+1} &= \phi(h^{l+1})
\end{align}
$$

with layer width $n_l$. Also define $\theta^l = [W^l, b^l]$ the vector of all parameters for layer $l$ and $\theta$ the vector of all parameters in the network. Denote $f_t(x) = h^{L+1}(x) \in \mathbb{R}^k$, and $l(\hat{y}, y)$ as the loss function. Finally, $\mathcal{L} = \sum_{(x, y) \in \mathcal{D}} l(f_t(x), \hat{y})$.

We can derive the ODE describing the evolution of parameters $\theta$ (also called gradient flow) as:

$$
\begin{align}
\theta_{t+1} &= \theta_t - \eta \nabla_\theta \mathcal{L} \\
&= \theta_t - \eta \nabla_\theta f_t(\mathcal{X})^\top \nabla_{f_t(\mathcal{X})}\mathcal{L} & \text{chain rule}\\
\implies \frac{\theta_{t+1} - \theta_{t}}{\eta} &= -\nabla_\theta f_t(\mathcal{X})^\top \nabla_{f_t(\mathcal{X})}\mathcal{L} \\
\implies \textcolor{red}{\dot{\theta_t}} &= -\nabla_\theta f_t(\mathcal{X})^\top \nabla_{f_t(\mathcal{X})}\mathcal{L} & \eta \to 0 
\end{align}
$$

Similarly, we can derive the evolution of the logits $f_t$:

$$
\begin{align}
\nabla_t f_t(\mathcal{X}) &= \nabla_\theta f_t(\mathcal{X}) \textcolor{red}{\nabla_t \theta} & \text{chain rule} \\
&= -\underbrace{\nabla_\theta f_t(\mathcal{X})\nabla_\theta f_t(\mathcal{X})^\top}_{\Theta_t(\mathcal{X}, \mathcal{X})} \nabla_{f_t(\mathcal{X})}\mathcal{L} \\
\end{align}
$$

where we refer to $\hat{\Theta}_t$ as the empirical tangent kernel.

### Linearized networks

We replace the outputs of the neural network with their Taylor expansion:

$$ 
\begin{align} 
f_t^{lin}(x) = f_0(x) + \nabla_\theta f_0(x)|_{\theta = \theta_0} \underbrace{(\theta_t - \theta_0)}_{\omega_t}
\end{align} 
$$ 
where $f_0$ is the initial output of the network.

Plugging into the above derivation of gradient flow, we get:

$$ 
\begin{align} 
\dot{\omega}_t &= \nabla_\theta f_0(\mathcal{X})^\top \nabla_{f_t^{lin}(\mathcal{X})}\mathcal{L} \\
\dot{f_t}^{lin}(x) &= \hat{\Theta}_0({x, \mathcal{X}}) \nabla_{f_t^{lin}(\mathcal{X})}\mathcal{L} \\
\end{align} 
$$ 

Under the MSE loss, these ODEs have clsoed form solutions (see section 2.2 in the paper). Therefore, without doing GD steps, we can compute the time evolution of the weights and the logits in a linearized neural network.

Taking the width to infinity in each layer yields a Gaussian Process, with a certain mean and covariance described by a kernel (see form in paper, eq 13).

### GPs from GD training

TODO (I don't understand the details in this section too well)
The main insight here is that $\forall x \in \mathcal{X}_{test}$, $f_t^{lin}(x)$ converges to a Gaussian distribution when taking the width $\to \infty$. 

### Infinite width networks are linearized networks

The authors show that applying GD with learning rate $\eta < \eta_{critical}$ and taking the width of all layers $\to \infty$, then $f_t^{lin}(x) \to \mathcal{N}(\mu(\mathcal{X}_T), \Sigma(\mathcal{X}_T, \mathcal{X}_T))$ (see form of mean and covariance in paper, eq. 15).