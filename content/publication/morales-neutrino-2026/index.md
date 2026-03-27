---
title: Neutrino Oscillation Parameter Estimation Using Structured Hierarchical Transformers
authors:
- Giorgio Morales
- Gregory Lehaut
- Antonin Vacheret
- Frederic Jurie
- Jalal Fadili
date: '2026-03-24'
publishDate: '2026-03-24T19:23:08.273721Z'
publication_types:
- paper-conference
publication: '*International Joint Conference on Neural Networks (IJCNN) (IJCNN 2026)*'
doi: 10.48550/arXiv.2603.22342
abstract: Neutrino oscillations encode fundamental information about neutrino masses and mixing parameters, offering a unique window into physics beyond the Standard Model. Estimating these parameters from oscillation probability maps is, however, computationally challenging due to the maps' high dimensionality and nonlinear dependence on the underlying physics. Traditional inference methods, such as likelihood-based or Monte Carlo sampling approaches, require extensive simulations to explore the parameter space, creating major bottlenecks for large-scale analyses. In this work, we introduce a data-driven framework that reformulates atmospheric neutrino oscillation parameter inference as a supervised regression task over structured oscillation maps. We propose a hierarchical transformer architecture that explicitly models the two-dimensional structure of these maps, capturing angular dependencies at fixed energies and global correlations across the energy spectrum. To improve physical consistency, the model is trained using a surrogate simulation constraint that enforces agreement between the predicted parameters and the reconstructed oscillation patterns. Furthermore, we introduce a neural network-based uncertainty quantification mechanism that produces distribution-free prediction intervals with formal coverage guarantees. Experiments on simulated oscillation maps under Earth-matter conditions demonstrate that the proposed method is comparable to a Markov Chain Monte Carlo baseline in estimation accuracy, with substantial improvements in computational cost (around 240x fewer FLOPs and 33x faster in average processing time). Moreover, the conformally calibrated prediction intervals remain narrow while achieving the target nominal coverage of 90%, confirming both the reliability and efficiency of our approach.
tags:
- Neutrino oscillation maps
- Transformer Networks
- Uncertainty Quantification
- Physics
links:
- name: ARXIV
  url: https://arxiv.org/abs/2603.22342
---
