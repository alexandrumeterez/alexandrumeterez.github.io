---
layout: distill
title: Deep Information Propagation
description: notes on Schoenholz et al.
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

## General

The correlation between 2 inputs, $c$, converges to a fixed point $c^*$ in depth. Note that $c^* = 1$ is a fixed point.

A fixed point is stable if a perturbation around the fixed point decays. This is equivalent to seeing how the fixed point varies between layers ($\chi_1$) and check whether the value increases $\chi_1 > 1$ or decreases $\chi_1 < 1$.

## Asymptotic expansions and depth scales

The distance between the empirical q, c maps and the their fixed points shrinks exponentially in depth, with scales $\xi_c, \xi_q$.

Proof follows expansion around the fixed point i.e. $q_{aa}^l = q^* + \epsilon^l$, similar for $c$. The formulas derived for the depth scales are Eq. 7 and 9.