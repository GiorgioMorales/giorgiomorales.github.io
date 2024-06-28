---
title: Towards reduced-cost hyperspectral and multispectral image classification
authors:
- Giorgio Morales
date: '2021-01-01'
publishDate: '2024-06-28T19:23:08.429464Z'
publication_types:
- thesis
abstract: 'In recent years, Hyperspectral Imaging systems (HSI) have become a powerful
  source for reliable data in applications such as remote sensing, agriculture, and
  biomedicine. However, the abundant spectral and spatial information of hyperspectral
  images makes them highly complex, which leads to the need for specialized Machine
  Learning algorithms to process and classify them. In that sense, the contribution
  of this thesis is multi-folded. We present a low-cost convolutional neural network
  designed for hyperspectral image classification called Hyper3DNet. Its architecture
  consists of two parts: a series of densely connected 3-D convolutions used as a
  feature extractor, and a series of 2-D separable convolutions used as a spatial
  encoder. We show that this design involves fewer trainable parameters compared to
  other approaches, yet without detriment to its performance. Furthermore, having
  observed that hyperspectral images benefit from methods to reduce the number of
  spectral bands while retaining the most useful information for a specific application,
  we present two novel hyperspectral dimensionality reduction techniques. First, we
  propose a filter-based method called Inter-Band Redundancy Analysis (IBRA) based
  on a collinearity analysis between a band and its neighbors. This analysis helps
  to remove redundant bands and dramatically reduces the search space. Second, we
  apply a wrapper-based approach called Greedy Spectral Selection (GSS) to the results
  of IBRA to select bands based on their information entropy values and train a compact
  Convolutional Neural Network to evaluate the performance of the current selection.
  We also propose a feature extraction framework that consists of two main steps:
  first, it reduces the total number of bands using IBRA; then, it can use any feature
  extraction method to obtain the desired number of feature channels. Finally, we
  use the original hyperspectral data cube to simulate the process of using actual
  filters in a multispectral imager. Experimental results show that our proposed Hyper3DNet
  architecture in conjunction with our dimensionality reduction techniques yields
  better classification results than the compared methods, producing more suitable
  results for a multispectral sensor design.'
links:
- name: URL
  url: https://scholarworks.montana.edu/xmlui/handle/1/16643
---
