---
title: Counterfactual Analysis of Neural Networks Used to Create Fertilizer Management
  Zones
authors:
- Giorgio Morales
- John Sheppard
date: '2024-03-01'
publishDate: '2024-06-28T19:23:08.273721Z'
publication_types:
- paper-conference
publication: '*2024 International Joint Conference on Neural Networks (IJCNN) (IJCNN
  2024)*'
doi: 10.1109/IJCNN60899.2024.10650046
abstract: In Precision Agriculture, the utilization of management zones (MZs) that
  take into account within-field variability facilitates effective fertilizer management.
  This approach enables the optimization of nitrogen (N) rates to maximize crop yield
  production and enhance agronomic use efficiency. However, existing works often neglect
  the consideration of responsivity to fertilizer as a factor influencing MZ determination.
  In response to this gap, we present a MZ clustering method based on fertilizer responsivity.
  We build upon the statement that the responsivity of a given site to the fertilizer
  rate is described by the shape of its corresponding N fertilizer-yield response
  (N-response) curve. Thus, we generate N-response curves for all sites within the
  field using a convolutional neural network (CNN). The shape of the approximated
  N-response curves is then characterized using functional principal component analysis.
  Subsequently, a counterfactual explanation (CFE) method is applied to discern the
  impact of various variables on MZ membership. The genetic algorithm-based CFE solves
  a multi-objective optimization problem and aims to identify the minimum combination
  of features needed to alter a site's cluster assignment. Results from two yield
  prediction datasets indicate that the features with the greatest influence on MZ
  membership are associated with terrain characteristics that either facilitate or
  impede fertilizer runoff, such as terrain slope or topographic aspect.
tags:
- Precision Agriculture
- Management Zones
- Response Curve
- Counterfactual explanations
- XAI
links:
- name: ARXIV
  url: http://arxiv.org/abs/2403.10730
---
