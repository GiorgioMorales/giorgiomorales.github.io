---
published: true
---
Being able to interpret a classifier's decision has become crucial lately. This ability allows us not only to ensure that a Convolutional Neural Network -for example- has learned the patterns that we expected but also to discover patterns that were not obvious at first glance. Most of the works related to Layer-wise Relevance Propagation (LRP) so far have been applied to image classification tasks; in this case, we are interested in finding the pixel positions that were more relevant for a given classification result. For example, the following image highlights the most relevant pixels to classify the image with the "cat" label:

![figure]({{ site.baseurl }}/images/pss.jpg)*Relevance is then backpropagated from the top layer down to the input, where $\{R_p\}$ denotes the pixel-wise relevance scores, that can be visualized as a heatmap. Source: [Montavon et al](https://doi.org/10.1016/j.patcog.2016.11.008)*


{% highlight python %}
x = ('a', 1, False)
{% endhighlight %}
