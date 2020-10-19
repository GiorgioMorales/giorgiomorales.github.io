---
published: true
---
Being able to interpret a classifier's decision has become crucial lately. This ability allows us not only to ensure that a Convolutional Neural Network -for example- has learned the patterns that we expected, but also to discover patterns that were not obvious at first glance. Most of the works related to Layer-wise Relevance Propagation (LRP) so far have been applied to image classification tasks; in that case, we are interested in finding the pixel positions that were more relevant for a given classification result. For example, the following image highlights the most relevant pixels to obtain a prediction of the class "cat":

![figure]({{ site.baseurl }}/images/catLRP.jpg)*Figure 1: Relevance is backpropagated from the top layer down to the input, where $\{R_p\}$ denotes the pixel-wise relevance scores, that can be visualized as a heatmap. Source: [Montavon et. al (2016)](https://doi.org/10.1016/j.patcog.2016.11.008)*
&nbsp;
&nbsp;

In the previous case, we know what kind of features define a cat, so we validate the resulting heatmap comparing it to our preconceived notions. What if we have a more complicated classification task that is not that easy for a human to interpret? In that case, we would like to use LRP the other way around; that is, to understand the local patterns that were more relevant to achieve a correct classification result. For instance, suppose that we gathered a great number of images of leaves with and without a certain disease, and we managed to train a Convolutional Neural Network  (CNN) that detects if a leaf is infected. Now, the users may not trust the magic 'black box' that is our CNN so we might use LRP as an option to explain what features among all the possible set of possibilities (e.g. coloration, holes, shapes) were the most relevant for our network when taking a decision. 

In this post, I will provide a specific example of the use of LRP applied to mutispectral images (MSI). To do this, we will use an [in-greenhouse dataset of hyperspectral images (HSI) of Kochia leaves](https://montana.box.com/v/kochiadataset) to classify the resistance level to two components commonly found in commercial herbicides in three categories: herbicide-susceptible, glyphosate-resistant, and dicamba-resistant. These HSI images contain 300 spectral bands (input channels). Even though if we select a reduced subset of important spectral bands (e.g. $10\sim20$), it is difficult for a human to identify what kind of information differentiates from a class to the other. Therefore, I will use LRP not only to observe the most relevant pixel positions but also the most relevant spectral bands for a given classification; that is, instead of outputting one final heatmap (as shown in Fig. 1), I will get one heatmap per input channel.

## Reading the dataset

As stated before, we will use a hyperspectral image dataset, which means that each image (or "datacube") consists of several channels (or "spectral bands"). Specifically, the dimension of this dataset is of $25\times25\times300$. Luckily, I have nicely formatted the dataset as a ".h5" file so that we do not need to worry about reading complicated "tif" or "bip" formats. Furthermore, given that in a HSI datacube consecutive bands are very similar and that we do not need that level of detail, we will average pairs of consecutive pairs of bands in order to reduce the computational burden and use 150 bands instead of 300 (which is interpreted as reducing the spectral resolution from 2.12nm to 4.24nm).

{% highlight python %}
hdf5_file = h5py.File('weed_dataset_w25.hdf5', "r")
train_x = np.array(hdf5_file["train_img"][...]).astype(np.float32)
train_y = np.array(hdf5_file["train_labels"][...])
# Average consecutive pairs of bands
for n in range(0, train_x.shape[0]):
	img2[n, :, :, int(i / 2)] = (xt[:, :, i] + xt[:, :, i + 1]) / 2.
train_x = img2
# Reshape as a 4-D TENSOR (because we will use 3-D convolutions)
trainx = np.reshape(trainx, (trainx.shape[0], trainx.shape[1], trainx.shape[2], trainx.shape[3], 1))
# Permute according to Pytorch's order
trainx = trainx.transpose((0, 4, 3, 1, 2))
{% endhighlight %}

s

## Network architecture

e

## 



{% highlight python %}
x = ('a', 1, False)
{% endhighlight %}
