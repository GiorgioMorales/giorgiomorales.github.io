---
title: Presentation at the Dynamics Research Group @ The University of Sheffield
event: Dynamics Research Group Seminar Series
event_url: 

location:
address:
  street: 
  city: 
  region: Sheffield
  postcode: 
  country: UK

summary: Presentation “Beyond Surrogates - Distilling Opaque Machine Learning Models into Interpretable Equations with Symbolic Regression.”
abstract: While high-capacity opaque machine learning models excel at fitting complex non-linear relationships, their lack of interpretability restricts scientific insight and limits safe deployment in engineering contexts. To bridge this gap, symbolic regression can be used to distill trained opaque models into explicit mathematical equations. In this talk, I introduce SeTGAP, a neural symbolic regression framework that distills opaque models into concise and interpretable expressions without restricting equation discovery to predefined candidate libraries. SeTGAP employs a Multi-Set Transformer to uncover per-variable symbolic skeletons from the opaque model's predictions, followed by evolutionary techniques that systematically combine them into multivariate expressions. Finally, we will discuss how this distillation framework could naturally extend to non-linear dynamical system identification (i.e., distilling surrogates trained on dynamic state data into white-box equations), opening exciting avenues for collaboration..


# Talk start and end times.
#   End time can optionally be hidden by prefixing the line with `#`.
date: '2026-10-02T12:00:00Z'
date_end: '2026-10-02T10:00:00Z'
all_day: false

# Schedule page publish date (NOT talk date).
# publishDate: '2017-01-01T00:00:00Z'

authors:
  - admin

tags: []

# Is this a featured talk? (true/false)
featured: true

image:
  caption: ''
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


# {{% callout note %}}
# Click on the **Slides** button above to view the built-in slides feature.
# {{% /callout %}}

# Slides can be added in a few ways:

# - **Create** slides using Hugo Blox Builder's [_Slides_](https://docs.hugoblox.com/reference/content-types/) feature and link using `slides` parameter in the front matter of the talk file
# - **Upload** an existing slide deck to `static/` and link using `url_slides` parameter in the front matter of the talk file
# - **Embed** your slides (e.g. Google Slides) or presentation video on this page using [shortcodes](https://docs.hugoblox.com/reference/markdown/).

# Further event details, including [page elements](https://docs.hugoblox.com/reference/markdown/) such as image galleries, can be added to the body of this page.

---

I had the pleasure of giving a seminar at [The University of Sheffield](https://sheffield.ac.uk/), where I talked about my work on symbolic regression/equation discovery, and their potential for identifying dynamical systems.

It was wonderful to meet the members of the [Dynamics Research Group (DRG)](https://drg.ac.uk/) and learn more about their work. I also got to visit their lab; as a computer scientist, it’s been a while since I’ve been in an actual lab! And of course, Sheffield is a beautiful city. Definitely enjoyed the visit!

Many thanks to [Max Champneys](https://sheffield.ac.uk/mac/people/research-staff/max-champneys) for the invitation, and to [Collins Ogbodo](https://collins-ogbodo.github.io/) for being such a great host. Really enjoyed the discussions and the opportunity to connect with the group!


<figure style="display: flex; flex-direction: column; align-items: center;">
    <img src="DRG.jpg" alt="Giorgio Morales in the DRG" width="90%">
    <figcaption style="text-align: center; margin-top: 5px; font-style: italic;">
        Dynamics Research Group Lab.
    </figcaption>
</figure>

<figure style="display: flex; flex-direction: column; align-items: center;">
    <img src="giorgio-morales-Sheffield.jpg" alt="Giorgio Morales in Sheffield" width="90%">
    <figcaption style="text-align: center; margin-top: 5px; font-style: italic;">
        Peace Gardens.
    </figcaption>
</figure>


<figure style="display: flex; flex-direction: column; align-items: center;">
    <img src="Sheffield.jpg" alt="Giorgio Morales in Sheffield" width="50%">
    <figcaption style="text-align: center; margin-top: 5px; font-style: italic;">
        Maida Vale.
    </figcaption>
</figure>
