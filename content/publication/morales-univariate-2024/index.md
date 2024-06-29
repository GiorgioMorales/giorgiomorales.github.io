---
title: Univariate Skeleton Prediction in Multivariate Systems Using Transformers
authors:
- Giorgio Morales
- John W. Sheppard
date: '2024-06-01'
publishDate: '2024-06-28T19:23:08.317271Z'
publication_types:
- paper-conference
publication: '*ECML PKDD 2024. European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases*'
abstract: Symbolic regression (SR) methods attempt to learn mathematical expressions
  that approximate the behavior of an observed system. However, when dealing with
  multivariate systems, they often fail to identify the functional form that explains
  the relationship between each variable and the system's response. To begin to address
  this, we propose an explainable neural SR method that generates univariate symbolic
  skeletons that aim to explain how each variable influences the system's response.
  By analyzing multiple sets of data generated artificially, where one input variable
  varies while others are fixed, relationships are modeled separately for each input
  variable. The response of such artificial data sets is estimated using a regression
  neural network (NN). Finally, the multiple sets of input-response pairs are processed
  by a pre-trained Multi-Set Transformer that solves a problem we termed Multi-Set
  Skeleton Prediction and outputs a univariate symbolic skeleton. Thus, such skeletons
  represent explanations of the function approximated by the regression NN. Experimental
  results demonstrate that this method learns skeleton expressions matching the underlying
  functions and outperforms two GP-based and two neural SR methods.

featured: true

tags:
- Symbolic regression
- XAI
- Transformers
- Knowledge discovery
links:
- name: URL
  url: http://arxiv.org/abs/2406.17834


links:
- name: Custom Link
  url: http://arxiv.org/abs/2406.17834
url_pdf: https://arxiv.org/pdf/2406.17834
url_code: 'https://github.com/NISL-MSU/MultiSetSR'
url_dataset: 'https://huggingface.co/datasets/AnonymousGM/MultiSetTransformerData'
url_source: 'https://colab.research.google.com/github/NISL-MSU/MultiSetSR/blob/master/DemoMSSP.ipynb'

# (url_poster: '#')
# (url_project: '')
# (url_slides: '')
# (url_video: '#')



# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: 'An example of a Multi-set skeleton prediction (MSSP) problem using the Multi-set Transformer'
  focal_point: ""
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects:
- Dissertation-project

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: example
# This work is driven by the results in my [previous paper]&#40;/publication/conference-paper/&#41; on LLMs.
# ({{% callout note %}})
# (Create your slides in Markdown - click the *Slides* button to check out the example.
# ({{% /callout %}})
# Add the publication's **full text** or **supplementary notes** here. You can use rich formatting such as including [code, math, and images]&#40;https://docs.hugoblox.com/content/writing-markdown-latex/&#41;.
---

