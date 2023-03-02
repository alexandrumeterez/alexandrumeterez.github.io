---
layout: distill
title: Deep Transformers wihtout Shortcuts - Modifying Self-Attention for Faithful Signal Propagation
description: notes on He et al.
giscus_comments: false
date: 2023-03-02

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Setting
  - name: Constructing trainable deep transformers without shortcuts

---

## Setting

I will not replicate the full setting here, just the necessary notation.

Let $X_l \in \mathbb{R}^{T \times d}$ be the output sequence for block $l$, where $T$ is the number of tokens and $d$ is the embedding size.

Skipless transformer: transformer without skip connection

Vanilla transformer: skipless transformer without normalization

### Self-attention

One headed attention mechanism:

$$
\begin{align}
\text{Attn}(X) &= A(X) V(X) \\
A(X) &= \text{softmax}(\frac{1}{\sqrt{d_k}}Q(X)K(X)^\top) \\
Q(X) &= XW^Q \\
K(X) &= XW^K \\
V(X) &= XW^V \\
\end{align}
$$

Causal attention:

$$
\begin{align}
A(X) = \text{softmax}(M \circ \frac{1}{\sqrt{d_k}}Q(X)K(X)^\top - \Gamma(1 - M))
\end{align}
$$

where $M$ is a lower triangular matrix and $\Gamma(1-M)$ is $\infty$ on the upper triangular side.

## Constructing trainable deep transformers without shortcuts

For a simplification, consider that we have only self-attention networks:

$$
\begin{align}
X_l &= A(X_{l-1})V(X_{l-1}) W_l^O \\
&= A_{l-1} V_{l-1} W_l^O
\end{align}
$$


We have that $X_0$ is the input sequence. Then:

$$
\begin{align}
X_L &= A_{L}V_{L} W_L^O \\
&= A_{L} X_{L-1} W^V_{L} W_L^O \\
&= \dots \\
&= [A_{L} \dots A_1] X_0 [\prod_{l=1}^{L}W_l^V W_l^O]
\end{align}
$$

Let $\Pi_l = A_{l} \dots A_1$ and $W_l = \prod_{l=1}^{L}W_l^V W_l^O$. Under the assumption that $W_l^O, W_l^V$ are both orthogonal, we get that $W$ is orthogonal. Then we can compute the kernel matrices at each layer:

$$
\begin{align}
\Sigma_0 &= X_0 X_0^\top \in \mathbb{R}^{T \times T} \\
\Sigma_l &= X_l X_l^\top \\
&= \Pi_l \Sigma_0 \Pi_l^\top
\end{align}
$$

The goal is now to control the product $\Pi_l$, which is done in the paper. Quite a nice solution.