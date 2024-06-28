---
title: Shadow Removal in High-Resolution Satellite Images Using Conditional Generative
  Adversarial Networks
authors:
- Giorgio Morales
- Samuel G. Huamán
- Joel Telles
date: '2019-01-01'
publishDate: '2024-06-28T19:23:08.392954Z'
publication_types:
- paper-conference
publication: '*Information Management and Big Data*'
doi: 10.1007/978-3-030-11680-4_31
abstract: In satellite image processing, obscure zones that were affected by shadows
  are normally discarded from further processing. Nevertheless, for specific applications,
  such as surveillance, it is desirable to remove shadows despite the fact that reconstructed
  zones do not necessarily have real reflectance values. In that sense, we propose
  a shadow removal method in high-resolution satellite images using conditional Generative
  Adversarial Networks (cGANs). The generator network is trained to produce shadow-free
  RGB images with condition on a satellite image patch altered with artificial shadows
  and concatenated with its respective binary shadow mask, while the discriminator
  is adversely trained to discern if a given shadow-free image comes from the generator
  or if it is an original RGB image without artificial alteration. The method is tested
  in the proposed dataset achieving an error ratio comparable with the state of the
  art. Finally, we confirm the feasibility of the proposed network using real shadowed
  images.
tags:
- Generative Adversarial Networks
- Satellite imagery
- Shadow removal
- Remote sensing
---
