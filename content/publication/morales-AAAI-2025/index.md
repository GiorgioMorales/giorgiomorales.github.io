---
title: Adaptive Sampling to Reduce Epistemic Uncertainty Using Prediction Interval-Generation Neural Networks
authors:
- Giorgio Morales
- John W. Sheppard
date: '2024-12-09'
publishDate: '2024-12-09T19:23:08.310824Z'
publication_types:
- paper-conference
publication: '*The 39th Annual AAAI Conference on Artificial Intelligence*'
# (doi: 10.1109/TNNLS.2023.3339470)
abstract: 'Obtaining high certainty in predictive models is crucial for making informed and trustworthy decisions in many scientific and engineering domains. However, extensive experimentation required for model accuracy can be both costly and time-consuming. This paper presents an adaptive sampling approach designed to reduce epistemic uncertainty in predictive models. Our primary contribution is the development of a metric that estimates potential epistemic uncertainty leveraging prediction interval-generation neural networks. This estimation relies on the distance between the predicted upper and lower bounds and the observed data at the tested positions and their neighboring points. Our second contribution is the proposal of a batch sampling strategy based on Gaussian processes (GPs). A GP is used as a surrogate model of the networks trained at each iteration of the adaptive sampling process. Using this GP, we design an acquisition function that selects a combination of sampling locations to maximize the reduction of epistemic uncertainty across the domain. We test our approach on three unidimensional synthetic problems and a multi-dimensional dataset based on an agricultural field for selecting experimental fertilizer rates. The results demonstrate that our method consistently converges faster to minimum epistemic uncertainty levels compared to Normalizing Flows Ensembles, MC-Dropout, and simple GPs.'

featured: false

tags:
- Adaptive Sampling
- Uncertainty quantification
- Prediction Intervals
- Regression neural networks


links:
- name: Arxiv
  url: https://arxiv.org/pdf/2412.10570

# (url_pdf: 'https://arxiv.org/pdf/2212.06370')
url_code: 'https://github.com/NISL-MSU/AdaptiveSampling'

# (url_source: 'https://colab.research.google.com/github/NISL-MSU/PredictionIntervals/blob/master/DualAQD_PredictionIntervals.ipynb')



# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: 'Image Credits: AAAI 2025'
  focal_point: ""
  preview_only: false

projects:
- Uncertainty-quantification
---
