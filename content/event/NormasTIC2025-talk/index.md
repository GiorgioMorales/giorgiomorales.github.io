---
title: Managing Uncertainty in Regression Neural Networks - From Prediction Intervals to Adaptive Sampling

event: DAC Axis Day on Causality and Uncertainty Quantification
event_url: https://www.normastic.fr/event/journee-de-laxe-dac-sur-la-causalite-et-la-quantification-dincertiture/#:~:text=rapidly%20evolving%20field.-,2%3A45%20p.m,-.%3A%20Giorgio%20Morales

location: UNICAEN, Campus 2 (Room S3-351)
address:
  street: 
  city: 
  region: Caen
  postcode: 
  country: France

summary: 🎤 I will be speaking at the upcoming Causality and Quantification of Uncertainties Day. 
abstract: '📚 Understanding and managing uncertainty is a critical aspect of deploying regression neural 
network models in real-world scientific and engineering applications. This presentation introduces two 
novel contributions aimed at improving uncertainty quantification and guiding data acquisition under 
uncertainty. The first is DualAQD, a dual-network architecture for generating high-quality prediction 
intervals (PIs). DualAQD integrates a custom loss function that minimizes interval width while ensuring 
coverage constraints, striking a balance between tightness and reliability of uncertainty estimates. 
It consistently outperforms existing PI-generation techniques in both interval efficiency and prediction 
accuracy across diverse datasets. 
Building on DualAQD uncertainty modeling, we present ASPINN, an adaptive sampling strategy designed for 
data-scarce environments where measurement collection is costly or constrained. ASPINN addresses this 
by focusing on epistemic uncertainty reduction in regression problems, using NN-generated PIs to guide 
adaptive data acquisition to strategically select new data points that most reduce model uncertainty. 
By incorporating a Gaussian Process surrogate to support batch sampling, ASPINN balances informativeness 
and diversity in acquisition decisions. Empirical evaluations show that ASPINN achieves faster convergence 
and greater uncertainty reduction compared to leading alternatives. Together, these methods offer a robust 
framework for uncertainty-aware learning in regression tasks.'


# Talk start and end times.
#   End time can optionally be hidden by prefixing the line with `#`.
date: '2025-06-26T14:45:00Z'
date_end: '2025-06-26T15:30:00Z'
all_day: false

# Schedule page publish date (NOT talk date).
publishDate: '2025-06-16T00:00:00Z'

authors:
  - admin

tags: []

# Is this a featured talk? (true/false)
featured: true

image:
  caption: 'Image credit: NormasTIC site'
  focal_point: Center

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
  - Uncertainty Quantification
  - XAI


# {{% callout note %}}
# Click on the **Slides** button above to view the built-in slides feature.
# {{% /callout %}}

# Slides can be added in a few ways:

# - **Create** slides using Hugo Blox Builder's [_Slides_](https://docs.hugoblox.com/reference/content-types/) feature and link using `slides` parameter in the front matter of the talk file
# - **Upload** an existing slide deck to `static/` and link using `url_slides` parameter in the front matter of the talk file
# - **Embed** your slides (e.g. Google Slides) or presentation video on this page using [shortcodes](https://docs.hugoblox.com/reference/markdown/).

# Further event details, including [page elements](https://docs.hugoblox.com/reference/markdown/) such as image galleries, can be added to the body of this page.

---

I'm pleased to announce that I will be speaking at the upcoming **Causality and Quantification of Uncertainties Day**, organized by the [Data, Learning, Knowledge axis of the Normastic federation](https://www.normastic.fr/donnees-apprentissage-connaissances/). 
This full-day event will take place on Thursday, June 26, 2025, at the University of Caen Normandy, and is open to all members of GREYC and LITIS—including colleagues, doctoral and post-doctoral researchers, and master’s students.

📍 Location: Campus 2, UFR des Sciences, Sciences 3 Building, Room S3-351
🕤 Time: 2:45 p.m.