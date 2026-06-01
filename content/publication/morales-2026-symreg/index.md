---
title: Learning Parametric Nitrogen Fertilizer Response Curves Using Neuro Symbolic Regression
authors:
- Giorgio Morales, John Sheppard
date: '2026-01-06'
publishDate: '2026-01-06T07:13:20.268831Z'
publication_types:
- paper-conference
publication: '*IEEE Congress on Evolutionary Computation (CEC): Workshop on Symbolic Regression and Equation Discovery (SymReg@WCCI2026)*'
abstract: Accurately modeling crop response to Nitrogen (N) fertilization is a fundamental challenge in precision agriculture, as it impacts both economic returns and environmental sustainability. Existing approaches either rely on predefined parametric forms or opaque machine learning models, limiting their ability to interpret or discover site-specific functional relationships from data. In this work, we propose a neuro symbolic regression (SR) approach to learn parametric N-response curves without assuming a predefined functional form. Our approach integrates a transformer-based Multi-Set Symbolic Skeleton Prediction strategy, enabling the discovery of shared functional structures across multiple subdomains or management zones (MZs). By constructing diverse input subsets and enforcing consistency across them, the method recovers robust symbolic skeletons that are subsequently fitted to observed data using a genetic algorithm. This framework was first evaluated on synthetic one-dimensional problems to assess its robustness under varying levels of epistemic uncertainty. The results demonstrate the ability of the proposed SR approach to recover correct expressions even in data-scarce regimes. In this work, we present the results of applying our method to real-world winter wheat data, learning distinct parametric N-response curves for different MZs within a field. The results show that the discovered expressions not only achieve lower fitting errors than traditional models such as quadratic-plateau and exponential functions, but also capture diverse functional behaviors across spatial regions. This demonstrates the potential that neuro SR has to enable the discovery of site-specific agronomic relationships and support informed decision-making in precision agriculture.

featured: false

tags:
- Symbolic regression
- XAI
- Transformers
- Knowledge discovery
- Precision Agriculture
links:
- name: ARXIV
  url: https://arxiv.org/abs/2605.31276

---
