---
title: Decomposable Neural Symbolic Regression
authors:
- Giorgio Morales
- John W. Sheppard
date: '2026-14-08'
publishDate: 2026-14-08T17:13:20.278345Z'
publication_types:
- manuscript
publication: 'Transactions on Machine Learning Research (TMLR)'
abstract:  Symbolic regression (SR) models complex systems by discovering mathematical expressions that capture underlying relationships in observed data. However, most SR methods prioritize minimizing prediction error over identifying the governing equations, often producing overly complex or inaccurate expressions. To address this, we present a decomposable SR method that generates interpretable multivariate expressions leveraging transformer models, genetic algorithms (GAs), and genetic programming (GP). In particular, our explainable SR method distills a trained "opaque'' regression model into mathematical expressions that serve as explanations of its computed function. Our method employs a Multi-Set Transformer to generate multiple univariate symbolic skeletons that characterize how each variable influences the opaque model's response. We then evaluate the generated skeletons' performance using a GA-based approach to select a subset of high-quality candidates before incrementally merging them via a GP-based cascade procedure that preserves their original skeleton structure. The final multivariate skeletons undergo coefficient optimization via a GA. We evaluated our method on problems with controlled and varying degrees of noise, demonstrating lower or comparable interpolation and extrapolation errors compared to two GP-based methods, three neural SR methods, and a hybrid approach. Unlike these methods, our approach consistently learned expressions that matched the original mathematical structure. Similarly, our method achieved both a high symbolic solution recovery rate and competitive predictive performance relative to benchmark methods on the Feynman dataset.

featured: true

tags:
- Symbolic regression
- XAI
- Transformers
- Knowledge discovery

links:
- name: arXiv
  url: https://arxiv.org/abs/2511.04124
- name: OpenReview
  url: https://openreview.net/forum?id=54EL928uCf

url_pdf: 'https://openreview.net/pdf?id=54EL928uCf'
url_code: 'https://github.com/NISL-MSU/MultiSetSR'

image:
  caption: ''
  focal_point: ""
  preview_only: false

projects:
- Dissertation-project
---
