---
published: true
---
Being able to interpret a classifier's decision has become crucial lately. This ability allows us not only to ensure that a Convolutional Neural Network -for example- has learned the patterns that we expected, but also to discover patterns that were not obvious at first glance. Most of the works related to Layer-wise Relevance Propagation (LRP) so far have been applied to image classification tasks; in that case, we are interested in finding the pixel positions that were more relevant for a given classification result. For example, the following image highlights the most relevant pixels to obtain a prediction of the class "cat":

![figure]({{ site.baseurl }}/images/catLRP.jpg)*Relevance is backpropagated from the top layer down to the input, where $\{R_p\}$ denotes the pixel-wise relevance scores, that can be visualized as a heatmap. Source: [Montavon et. al](https://doi.org/10.1016/j.patcog.2016.11.008)*

In the previous case, we know what kind of features define a cat, so we validate the resulting heatmap comparing it to our preconceived notions. However, what if

In the previous case, we know what kind of features define a cat, so we validate the resulting heatmap comparing it to our preconceived notions. What if we have a more complicated classification task that is not that easy for a human to interpret? In that case, we would like to use LRP the other way around; that is, to understand the local patterns that were more relevant to achieve a correct classification result. For instance, suppose that we gathered a great number of images of leaves with and without a certain disease, and we managed to train a Convolutional Neural Network  (CNN) that detects if a leaf is infected. Now, the users may not trust the magic 'black box' that is our CNN so we would like to explain what features among all the possible set of possibilities (e.g. coloration, holes, shapes) were the most relevant for our network when taking a decision


{% highlight python %}
x = ('a', 1, False)
{% endhighlight %}
