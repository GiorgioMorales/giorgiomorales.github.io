# How to Generate Prediction Intervals using Neural Networks on Python

__Paper: â€œDual Accuracy\-Quality\-Driven Neural Network for Prediction Interval Generationâ€__

__Giorgio Morales__

Gianforte School of Computing

Montana State University

---

10s

# Installation

[Git](https://git-scm.com/download/) and [Pytorch](https://pytorch.org/) have to be already installed

Run  __pip install \-q __  __git\+https__  __://github\.com/NISL\-MSU/__  __PredictionIntervals__  __ __ in the terminal

Training the Models

* DualAQD uses two NNs: a target\-estimation NN that generates accurate estimates and a prediction interval \(PI\)\-generation NN that produces the PI upper and lower bounds
* First\, create an instance of the class  __PredictionIntervals__ \.  __Parameters__ :
  * __X__ : Input data \(explainable variables\)\. 2\-D numpy array\, shape \(\#samples\, \#features\)
  * __Y__ : Target data \(response variable\)\. 1\-D numpy array\, shape \(\#samples\, \#features\)
  * __Xval__ : Validation input data\. 2\-D numpy array\, shape \(\#samples\, \#features\)
  * __Yval__ : Validation target data\. 1\-D numpy array\, shape \(\#samples\, \#features\)
  * __method__ : PI\-generation method\.  Options: 'DualAQDâ€™ or '[MCDropout](https://arxiv.org/pdf/1709.01907.pdf)'
  * __normData__ : If True\, apply z\-score normalization to the inputs and min\-max normalization to the outputs

![](img%5CPIs0.png)

---

94

Training the Models

* Normalization is applied to the training set; then\, the exact same scaling is applied to the validation set\.
* To train the model\, call the \` __train__ \` method\.  __Parameters__ :
  * __batch\_size__ : Mini batch size\. It is recommended a small number\.  _default: 16 _
  * __epochs__ : Number of training epochs\.  _default: 1000_
  * __eta\___ : Scale factor used to update the self\-adaptive coefficient lambda \(Eq\. 6 of the paper\)\.  _default: 0\.01_
  * __printProcess__ : If True\, print the training process \(loss and validation metrics after each epoch\)\.  _default: False _
  * __plotCurves__ : If True\, plot the training and validation curves at the end of the training process

![](img%5CPIs1.png)

---

94

Evaluate the model on a test set

* To do this\, call the method  __evaluate__ \.  __Parameters__ :
  * __Xeval__ : Evaluation data
  * __Yeval__ : Optional\. Evaluation targets\.  _default: None_
  * __normData__ : If True\, apply the same normalization that was applied to the training set
* Returns:
  * If  __Yeval__  is  _None_ : It returns predictions  __ypred__  __\, __  __y\_u__  __\, __  __y\_l__  \(i\.e\.\, target predictions\, PI upper bounds\, and PI lower bounds\)\.
  * If  __Yeval__  is not None: It returns performance metrics and predictions  __mse__  __\, PICP\, MPIW\, __  __ypred__  __\, __  __y\_u__  __\, __  __y\_l__  \(i\.e\.\, mean square error of target predictions\, PI coverage probability\, mean PI width\, target predictions\, PI upper bounds\, and PI lower bounds\)\.
* Note:  __Yeval__  is  _None_  in the case that the target values of the evaluation data are not known\.

![](img%5CPIs2.png)

---

94

![](img%5CPIs3.png)

![](img%5CPIs4.png)

---

94


