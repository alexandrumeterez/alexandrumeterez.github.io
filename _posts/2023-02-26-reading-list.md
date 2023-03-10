---
layout: distill
title: reading list
description: papers relevant for my msc thesis
giscus_comments: false
date: 2023-02-25

# bibliography: 2018-12-22-distill.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Side note
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Training without normalization
  - name: NTK
  - name: Mean Field, Dynamical Isometry and designing inits
  - name: Transformers
  - name: Misc
    
# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.

---

## Side note

❌ Reminder to myself: these notes **should not, under any circumstance, be copy-pasted into the thesis/paper, without rewriting them**. These are simply study notes and the wording **is not original**, hence copy-pasting is considered **plagiarism**. ❌

Papers marked with ✅, are the ones I've read in depth (and preferably took notes on).

There is no particular order to these sections.

---

## Training without normalization

- ✅ Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks
- ✅ Fixup Initialization: Residual Learning Without Normalization

---

## NTK

- ✅ Neural Tangent Kernel: Convergence and Generalization in Neural Networks
- ✅ Deep Neural Networks as Gaussian Processes
- ✅ Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent
- Tensor Programs II: Neural Tangent Kernel for Any Architecture

---

## Mean Field, Dynamical Isometry and designing inits

- ✅ Exponential expressivity in deep neural networks through transient chaos
- Deep Information Propagation
- Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice
- Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks
- Dynamical Isometry and a Mean Field Theory of RNNs: Gating Enables Signal Propagation in Recurrent Neural Networks
- Dynamical Isometry and a Mean Field Theory of LSTMs and GRUs
- The Emergence of Spectral Universality in Deep Networks
- Mean Field Residual Networks: On the Edge of Chaos
- Exact solutions to the nonlinear dynamics of learning in deep linear neural networks.

---

## Transformers

- Rapid training of deep neural networks without skip connections or normalization layers using Deep Kernel Shaping
- Deep Learning without Shortcuts: Shaping the Kernel with Tailored Rectifiers
- ✅ Deep Transformers without Shortcuts: Modifying Self-attention for Faithful Signal Propagation
- [https://transformer-circuits.pub/2021/framework/index.html]
- Infinite attention: NNGP and NTK for deep attention networks
- [https://hyunjik11.github.io/talks/Attention_the_Analogue_of_Kernels_in_Deep_Learning.pdf]

---

### Rank collapse

- ✅ Attention is not all you need: pure attention loses rank doubly exponentially with depth
- ✅ Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse
- ✅ ReZero is All You Need: Fast Convergence at Large Depth

---

## Misc
- $L_2$ L2 Regularization versus Batch and Weight Normalization