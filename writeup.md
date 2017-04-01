# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

Here is the link to my [project code](https://github.com/cfficaurzua/P2-SDCND/blob/master/Traffic_Sign_Classifier.ipynb)

---

[TOC]
 
## Goals
The goals of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./images_report/training_distribution.png "Training Distribution"
[image2]: ./images_report/raw_visualization.png "Raw_visualization"
[image3]: ./images_report/augmentations_examples.png "Augmentations_examples"
[image4]: ./images_report/augmented_data.png "Augmented_Data"
[image5]: ./images_report/balanced_training_distribution.png "Balanced Training Distribution"
[image6]: ./images_report/training_curve.PNG "training_curve"
[image7]: ./raw_german_street/Capture8.PNG "street view"
[image8]: ./raw_german_street/20170319_135216.jpg "shared dataset"
[image9]: ./raw_german_street/20170319_134041.jpg "shared dataset"
[image10]: ./raw_german_street/20170319_135255.jpg "shared dataset"
[image11]: ./images_report/predictions1.PNG "predictions"
[image12]: ./images_report/wrong_predictions.PNG "wrong_predicitons"
[image13]: ./images_report/no_entry_wrong.PNG "no_entry_wrong"
[image14]: ./images_report/narrows_on_right_wrong.PNG "no_entry_wrong"
[image15]: ./images_report/30_predictions.PNG "30_mislabel"
[image16]: ./images_report/network_activations.PNG "Network activations"

---


## Data Set Summary

The code for this step is contained in the 2nd code cell of the IPython notebook.  

I used the *shape* method embedded in the numpy library, in order to get the sizes corresponding to the training set, test set and validation set. To get the number of classes, I used *pandas*, reading  the *signnames.csv* file and from there, extract the quantity of classes using the aforementioned *shape* function.

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32,3)
* The number of unique classes/labels in the data set is 43
---

## Data Set Visualization

The code for this step is contained in the 4th code cell of the IPython notebook.  

To understand the distribution of the training dataset, I plotted a bar graph using the *matplotlib.pyplot* library.
At first glance, it can be noticed that some classes have a great amount of examples (~2000 examples) compared to others than have as few as  ~100 examples. this biased situation will induce a high probability of answering right in the training set if the neural network chooses the bigger classes, but this will not occur in any other set, leading to an overfit.

![alt text][image1]

Then, I decided to take a look into the training set, to get an insight of the quality of the pictures, besides any other interesting feature that may appear. To visualize the training set, I created a function that plots a grid of the giving set, with a configurable size, choosing random examples within each class.

![alt text][image2]
---
## Data Augmentation

The code for this step is contained in the 7th code cell of the IPython notebook.  

Since, the dataset was unbalanced, data augmentation is needed. In order to achive augmenting the data, I wrote a set of function. Each of them distort the input picture in someway and return a different image of the same traffic sign.

The functions are detailed bellow:

* **Transform augmentation:** Makes an affine transform using random shear and rotation factors within a normal distribution using *μ = 0* and *σ = 0.1*, This will return an rotated and/or skeewed image.
* **Perspective augmentation:** Return the same image, but tweaked in such way that it would appear as being looked either from above or below or from the right or the left side, as if the perspective would have been changed. To achive this trasformation, I took two adjacent vertices of the input image and moved them closer together by a factor of *d*, and moved the two vertices left far apart by the same distance. To perform the actual transformation I use the openCV library. 
* **Destroy augmentation:**  Return the same image but with *n* random pixeles substituted by the median of the image, this destroy a great percentage of the picture, making it harder to recognize, as a human being one tends to squint, in order to be able to recognize the picture, looking for the general features instead of details, I hope that the machine would be able to do the same.
* **Enhance augmentation:** Adjust randomly the brightness or the contrast of the image using the *Contrast*and *Brightness* fucntions from the *PIL.ImageEnhance* library.
* **Flip augmentation:** Return a horizontal, vertical and horizontal then vertical flipped version of the input image, but first checks if the flip action can be done by looking at the label of the input. This means that certain traffic sings can't be flipped, e.g: *Road Work Sign*; others can be flipped only vertically, e.g: *Speed limit (80km/h)*; finally there are Traffic Sings that can be flipped horizontally and then vertically, e.g: *End of all speed and passing limits*.

Here are some examples of what each function return.

![alt text][image3]

The augmentation process (augment_data function) consists in randomly applying one of this functions to a random image from the training set, until each class, has been extended in a fix length (4000).
Then the data is balanced choosing random images from the returned set from the previous function until each class has a fix length (4000) .

Visualizing the new dataset:

![alt text][image4]

Following these two processes I get a nicer and bigger dataset distribution as shown bellow:
![alt text][image5]

## Preprocessing

The code for this step is contained in the 13th code cell of the IPython notebook.  
In order to preprocess the data, the following sequence is applied:

 1. **Enhance**: This step consists in enhancing the details of a given image, I tried to copy what I do in Photoshop when I want to sharpen the edges without distorting the image.
In photoshop the algorithm is presented as follows:

	1. Duplicate the image in a different layer on top of the current one.
	2. Apply a Highpass filter to the top layer
	3. Apply a Linear light blend mode.

	I researched a little bit and find out that the highpass filter is a convolution of the image plus an offset of 0.5. Moreover, The linear light blending mode consists in a weighted sum between a linear dodge of the two stacked layers (A+B) , and a linear burn of the the two stacked layers (A+B)-1.
	
	Linear light = W*(A+B)+(1-W)*((A+B)-1)
	
	Therefore the algorithm in python is the following:

	1. Apply a Highpass filter and store it in a variable h_pass
	2. Apply a linear dodge with h_pass and the input image
	3. Apply a linear burn with h_pass and the input image
	4. Normalize the image to get the weights
	5. Multiply the linear dodge result with the weights
	6. Multiply the linear_burn results with the inverse of the weights
	7. Sum the two multiplication results.

 2. **Histogram equalization**:  Here a dynamic histogram equalization is perfomed in the intensity channel of the HSV colormap of the input image. I read that a lot of people were using the YUV map, but then read that the YUV colormap is optimized to the human eye perception field, due to the fact that machines don't have a biased perception field, I have chosen the HSV map. I opt for this technique, because some images were to dark and other were to bright, this method help to resolve that problem. The output returned is in HSV
 
 3. **Normalization**: I applied a normalization step to each channel of the input image. I chose to normalize the data because the training process develops better when the data has a low variance, the output range goes from 0.1 to 0.9.
 
 4. **Extract color information**: I researched that color doesn't contribute much to the final result in the trainning process, but for me, color is very important in traffic sign recognition, so I try to sum up the color information in a flat array. To do this, the input image is divided into *n* divisions and then a histogram is perfom in the hue channel of each division. the output is finally concatenated. This will be used as an input to the fully connected layer of the neural network.
 

## Model Architecutre

The code for my final model is located in the 18th cell of the ipython notebook. 

The model consisted in two inputs: an Image of intensity values and an array containing the normed histogram values from the Hue channel of the HSV image.

For the first set of layers, each layer consisted in a sequence of a convolution, followed by a, Relu activation, followed by a Max pooling and finally a dropout. The output of each layer is then flattened and concatenated in conjuction with the second input.

For the second set of layers, receives, as an input, the given flattened array. 

The structure is summed up in the next table.

|TAG  	| Layer   		|Input 			|     Description					| 
|:-----:|:---------------------:|:---------------------:|:-----------------------------------------------------:|
| I1	| Input         	|-			| 32x32x1 V channel of HSV image	 		|
| I2 	| Input         	|-			| 1x96 Histogram Color Information			|  
| L1.1	| Convolution 5x5 	|I1			| 1x1 stride, same padding, outputs 32x32x32		|
| L1.2  | RELU			|L1.1			|-							|
| L1.3  | Max pooling		|L1.2			| 2x2 stride,  outputs 16x16x32				|
| L1.4  | Dropout		|L1.3			|keep probability = 0.9					|
| L2.1  | Convolution 5x5  	|L1.4			| 1x1 stride, same padding, outputs 16x16x64 		|
| l2.2  | RELU			|L2.1			|			-				|
| l2.3  | Max pooling	    	|L2.2			| 2x2 stride,  outputs 16x16x32				|
| l2.4  | Dropout		|L2.3			|keep probability = 0.8					|
| l3.1  | Convolution 5x5 	|L2.4			| 1x1 stride, same padding, outputs 16x16x64 		|
| l3.2 	| RELU			|L3.1			|-							|
| l3.3  | Max pooling	    	|L3.2 			| 2x2 stride,  outputs 16x16x32				|
| l3.4	| Dropout		|L3.2			|keep probability = 0.5					|
| l4.1	| Concatenate		|L1.4, l2.4,l3.4, I2	|-							|
| l4.2	| Fully Connected Layer	|L4.1			|3680x1024-						|
| l4.3  | RELU			|L2.1			|							|
| l4.4	| Dropout		|L3.2			|keep probability = 0.1					|
| logits| Fully Connected Layer	|L4.1			|3680x1024-						|


## Model Training

To train the model, I used an adam optimizer with cross entropy and L2 regularization,
the parameters chosen are as follows:

|Parameter 	| Value 	| Function 		|
|:-------------:|:-------------:|:---------------------:|
| μ (mu)	| 0		|weights Initialization	|
| σ (sigma)	| 0.1		|weights Initialization	|
| Epochs	| 64		|Training		|
| Batch size	| 150		|Training		|
| learning rate	| 0.001		|Training		|
| β (beta)	| 0.01		|L2 regularization	|

The bias terms were all initialized as zeros.

## Results and discussion
The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

First I run the lenet model, and get about 86% of val accuracy, I chose this one as a starting point, because is was the only one that I knew and have already practice with it. 
Following up, I started to change some hyper parameters, In order to improve my result, but i couldn't achieve more than 90%, although, training the lenet model was way more faster than any other architecture I tested on.
Next I added more features to each layer, the result improve a little bit, but the training slowed down quite a lot. I added dropout to every layer with keep probabilities ranges between .5 and .95. Then, I try to adjust my model to the one posted by our classmate Alex Staravoitau [here](http://navoshta.com/traffic-signs-classification/). concatenating every relu activation output from the convolutional layers and insert it into the fully connected layers. Alex's publication concludes that the model might improve if some color information is added up. so I resume all color infor in Hue channel's histogram of the HSV color map and introduced it directly to the net in the fully connected layers.
Even though I included dropout, I was still getting some overfitting finally I achieve equal results in the training and validation set by lowering down the keep probability in the fully connected layer to .1 and put L2 regularization in both fully connected layers.
Sometines when I use keep probabilities too low, it failed completely to train. getting results as low as 0.2%.
At first I started to train with 10 EPOCHS, then as I moved from my local pc to an AWS gpu service I increase the EPOCHS to 100. Finally as I achieved that the training occur a little faster, threrefore I reduced the number of epochs to 35.
The final result took around ~60-90 min to train. + ~20 minutes to augment the dataset and ~10 to preprocess the data.

My final model results were:
* training set accuracy of 99.0%
* validation set accuracy of 99.0%
* test set accuracy of 96.7%

![alt text][image6]

## Test a Model on New Images

I download a set of images from google street view, particularly from the intersection between untermainbrücke and mainkai in frankfurt, germany. I also append to the set,some pictures shared by our classmate Sonja Krause-Harder. 
Here are some of the pictures taken:

![alt text][image7] ![alt text][image8] 

Then I cropped out 40 traffic signs. 

There are a couple of images that I found peculiar and hard to identify.
the first one comes from this picture:

![alt text][image9] 

As shown above, there are two signs of children crossing, but one is the flipped version of the official Sign. Neither the training set, test set nor validation set have any flipped version of the children crossing sign. therefore the model is prone to make a mistake here.

The second image is the following:

![alt text][image10]

In this image, I don't even understand what does it mean, for me, it looks like a double negation no entry Sign, the interesting thing is that if you look at the center there is a Yield sign within the original Traffic Sign, so definitely the program will have troubles there.

### Model's Predictions

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

The model correctly predicted 85% of the pictures taken.

Here are some results of the prediction, with the given certainty:

![alt text][image11]

The model seems to be very confident in the predictions it made, giving correct results for most of the Traffic signs

Yet, as expected, the model failed to recognize the flipped version of the children crossing sign.
but the correct label still appears within the top 5 probabilities, in the third position to be precise.

![alt text][image12]

Regarding to the *no entry* sign, as shown below, the model did not succeeded, instead of looking at the general picture, it focus in the unintended yield sign that appears at the centre.

![alt text][image13]

The model has a misconception of the *Road narrows on the right* Sign, mislabeling with the *children crossing* Sign

![alt text][image15]

Lastly the model finds it, sometimes, hard to differentiate between the *speed limit (30 km/h)*  and *speed limit (30 km/h)*

![alt text][image16]

## Neurons Activations

To get an insight of what the neurons are actually viewing, I use the function provided to plot the activations when a giving set of pictures is given, using the *speed limit (30 km/h)*, the output of the activation in each feature for the first layer is shown below:

![alt text][image17]

It can be seen that some featuremaps (8, 12, 21) focus in the numbers given as expected, while other features (18, 14) focus more on unrelated aspects of the picture like the background, this will produce some overfitting.

## Conclusion

In this project I could successfully load the data set, explore it and visualize it, apply functions in order to balance and augment the data to later apply preprocessing function to enhance and normalize the picture. 

I was able to Design, train a test a model architecture that can recognize traffic signals with a accuracy of 99% in the validation set, then use the model to actually perform predictions on new images with confident results. Finally the probabilities were represented using bar graphs to understand where the model fails, and which sign are more likely to make mistakes when predicting.
