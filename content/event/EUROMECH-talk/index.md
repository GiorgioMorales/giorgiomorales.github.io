---
title: Symbolic Regression Talk at EUROMECH Colloquium 662

event: Euromech Colloquium “Physics-enhanced machine learning and data-driven nonlinear dynamics” 
event_url: https://662.euromech.org/welcome-to-the-physics-enhanced-machine-learning-and-data-driven-nonlinear-dynamics/

location:
address:
  street: 
  city: 
  region: Como
  postcode: 
  country: Italy

summary: Presentation “Discovering Non-Linear Equations Under Epistemic Uncertainty Using Transformer-Based Multi-Set Skeleton Prediction.”
abstract: Discovering Non-Linear Equations Under Epistemic Uncertainty Using Transformer-Based Multi-Set Skeleton Prediction.


# Talk start and end times.
#   End time can optionally be hidden by prefixing the line with `#`.
date: '2026-04-28T17:00:00Z'
date_end: '2026-05-30T19:00:00Z'
all_day: false

# Schedule page publish date (NOT talk date).
# publishDate: '2017-01-01T00:00:00Z'

authors:
  - admin

tags: []

# Is this a featured talk? (true/false)
featured: true

image:
  caption: 'EUROMECH Colloquium 662. Photo: Alice Cicirello'
  focal_point: ""

#links:
#  - icon: twitter
#    icon_pack: fab
#    name: Follow
#    url: https://twitter.com/georgecushen
# url_code: 'https://github.com'
# url_pdf: ''
# url_slides: 'https://slideshare.net'
# url_video: 'https://youtube.com'

# Markdown Slides (optional).
#   Associate this talk with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects:
  - Symbolic regression
  - XAI
  - Uncertainty quantification


# {{% callout note %}}
# Click on the **Slides** button above to view the built-in slides feature.
# {{% /callout %}}

# Slides can be added in a few ways:

# - **Create** slides using Hugo Blox Builder's [_Slides_](https://docs.hugoblox.com/reference/content-types/) feature and link using `slides` parameter in the front matter of the talk file
# - **Upload** an existing slide deck to `static/` and link using `url_slides` parameter in the front matter of the talk file
# - **Embed** your slides (e.g. Google Slides) or presentation video on this page using [shortcodes](https://docs.hugoblox.com/reference/markdown/).

# Further event details, including [page elements](https://docs.hugoblox.com/reference/markdown/) such as image galleries, can be added to the body of this page.

---

Discovering interpretable mathematical descriptions of nonlinear systems from data is a central goal in scientific machine learning. However, existing data-driven and symbolic regression (SR) approaches often struggle in data-scarce settings, where epistemic uncertainty leads to unstable models and overfitting to local artifacts. We propose an uncertainty-aware framework that integrates adaptive sampling (AS) with a Multi-Set Symbolic Skeleton Prediction (MSSP) approach, enabling the progressive extraction of stable and accurate symbolic expressions from learned models as data coverage improves.
We present a pipeline that combines a function approximator (e.g., a neural network trained on the currently available observations) with an MSSP-based stage. Rather than performing SR on a single global input–response pairing, MSSP constructs multiple input–response subsets sampled from the model’s response surface. These distinct yet related subsets are used to recover a common symbolic skeleton that captures the shared structure of the underlying mapping while being robust to localized distortions caused by sparse sampling or noise. After skeletons are proposed using a [pre-trained Multi-Set Transformer](publication/morales-univariate-2024/), coefficients are fitted against the observed data to produce the final expressions.

We use an [AS loop](publication/morales-aaai-2025/) to iteratively reduce epistemic uncertainty across the input domain. At each iteration, the learned predictor is re-evaluated on a fixed test grid to characterize where uncertainty remains large. AS then prioritizes new observations in these epistemically uncertain regions using prediction interval-based metrics and a batch sampling strategy based on Gaussian processes. MSSP is re-applied throughout this process to monitor how recovered expressions evolve as coverage improves.
As a proof-of-concept, we demonstrate our pipeline on 1-D synthetic problems, where the estimated expressions begin to match the true functional form after sufficient AS iterations. Although correct or near-correct expressions can occasionally be identified at early stages, they are typically unstable. By coupling MSSP with AS, these effects are progressively mitigated as uncertainty is reduced and coverage improves, allowing convergence toward simpler and correct functional forms. While results are presented for 1-D problems, the framework naturally extends to higher-dimensional systems.

<div style="position: relative; width: 100%; height: 0; padding-bottom: 58.52%;">
  <iframe src="https://1drv.ms/p/c/c56982f783f2d4b4/IQTFEBoczc55T43sCt8L9k0TAbISh-CQzOjFqawC1YC2xRE?em=2&amp;wdAr=1.7777777777777777" frameborder="0" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
</div>

<figure style="display: flex; flex-direction: column; align-items: center;">
    <img src="1.png" alt="figure" width="90%">
    <figcaption style="text-align: center; margin-top: 5px; font-style: italic;">
        Lake Como.
    </figcaption>
</figure>

<figure style="display: flex; flex-direction: column; align-items: center;">
    <img src="2.png" alt="figure" width="90%">
    <figcaption style="text-align: center; margin-top: 5px; font-style: italic;">
        View from the conference room.
    </figcaption>
</figure>

<figure style="display: flex; flex-direction: column; align-items: center;">
    <img src="3.png" alt="figure" width="90%">
    <figcaption style="text-align: center; margin-top: 5px; font-style: italic;">
        View from the conference room.
    </figcaption>
</figure>

<figure style="display: flex; flex-direction: column; align-items: center;">
    <img src="2.png" alt="figure" width="90%">
    <figcaption style="text-align: center; margin-top: 5px; font-style: italic;">
        Presentation day.
    </figcaption>
</figure>