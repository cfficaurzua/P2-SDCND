# **Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

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
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---
Here is the link to my [project code](https://github.com/cfficaurzua/P2-SDCND/blob/master/Traffic_Sign_Classifier.ipynb)

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
at first glance, it can be noticed that some classes have a great amount of examples (~2000 examples) compared to others than have as few as  ~100 examples. this biased situation will induce a high probability of answering right in the training set if the neural network chooses the bigger classes, but this will not occur in any other set, leading to an overfit.

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
![alt text][image3]

Following these two processes I get a nicer and bigger dataset distribution as shown bellow:

![alt text][image1]


##Preprocessing
In order to preprocess the data, the following sequence is applied:

 1. **Enhance**: This step consists in enhance the details of a given image, I tried to copy the algorithm I usually used in Photoshop when I want to sharpen the edges without distorting the image.
In photoshop the algorithm is presented as follows:

	1. Duplicate the image in a different layer on top of the current one.
	2. Apply a Highpass filter to the top layer
	3. Apply a Linear light blend mode.

I researched a little bit and find out that the  highpass filter is a is a convolution of the image plus an offset of 0.5. Also, The linear light blending mode consists in a weighted sum between a linear_dodge of the two stacked layers (A+B) , and a linear burn of the the two stacked layers (A+B)-1.
linear light = W*(A+B)+(1-W)*((A+B)-1)
Therefore the algorithm in python is the following:

	1. Apply a Highpass filter and store it in a variable h_pass
	2. Apply a linear dodge with h_pass and the input image
	3. Apply a linear burn with h_pass and the input image
	4. Normalize the image to get the weights
	5. Multiply the linear dodge result with the weights
	6. Multiply the linear_burn results with the inverse of the weights
	7. Sum the two multiplication results.

 2. **Histogram equalization**:  Here a dynamic histogram equalization is perfomed in the intensity channel of the HSV colormap of the input image. I read that lot of people were using the YUV map, but then read that the YUV colormap is optimized to the human eye perception field, due to the fact that machines don't have a biased perceptionfield I chose the HSV map. I chose this technique, because some images were to dark and other were to bright, this method help to resolve that problem. The output returned is in HSV
 3. **Normalization**: I applied a normalization step to each channel of the input image. I chose to normalize the data because the training process develops better when the data has a low variance and is centered in 0.
 4. **Extract color information**: I researched that color doesn't contribute much to the final result in the trainning process, but for me, color is very important in traffic sign recognition, so I try to sum up the color information in a flat array. To do this, the input image is divided into n divisions and then a histogram is perfom in the hue channel of each division. the output is then concatenated. This will be used as an input to the fully connected layer of the neural network.
 

## Model Architecutre
3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 
The model consisted in two inputs an Image of intensity values, and a array containing the normed histogram values from the Hue channel of the HSV image.
For the first set of layers, each layer consisted in a sequence of a convolution, followed by a, Relu activation. followed by a Max pooling and finally a dropout. The output of each layer is then flattened and concatenated in conjuction with the second input.
For the second set of layers, receive as an input the given 
y final model consisted of the following layers:

|TAG  						| Layer   |Input 				  	|     Description		| 
|:-----:|:-----------------:|:-------:|:-----------------------:|:---------------------:|
| I1	| Input         	|-		| 32x32x1 V channel of HSV image	 				|
| I2 	| Input         	|-		| 1x96 Histogram Color Information					|  
| L1.1	| Convolution 5x5 	|I1		| 1x1 stride, same padding, outputs 32x32x32		|
| L1.2  | RELU				|L1.1	|-													|
| L1.3  | Max pooling		|L1.2	| 2x2 stride,  outputs 16x16x32						|
| L1.4  | Dropout			|L1.3	|keep probability = 0.9								|
| L2.1  | Convolution 5x5  	|L1.4	| 1x1 stride, same padding, outputs 16x16x64 		|
| l2.2  | RELU				|L2.1	|													|
| l2.3  | Max pooling	    |L2.2	| 2x2 stride,  outputs 16x16x32						|
| l2.4  | Dropout			|L2.3	|keep probability = 0.8								|
| l3.1  | Convolution 5x5 	|L2.4	| 1x1 stride, same padding, outputs 16x16x64 		|
| l3.2 	| RELU				|L3.1	|-													|
| l3.3  | Max pooling	    |L3.2 	| 2x2 stride,  outputs 16x16x32						|
| l3.4	| Dropout			|L3.2	|keep probability = 0.5								|
| l4.1	| Concatenate		|L1.4, l2.4,l3.4, I2	|-									|
| l4.2	| Fully Connected Layer|L4.1|3680x1024-											|
| l4.3  | RELU				|L2.1	|													|
| l4.4	| Dropout			|L3.2	|keep probability = 0.1								|
| logits| Fully Connected Layer|L4.1|3680x1024-											|



## Model Training
 Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an adam optimizer with cross entropy,
the parameters chosen are as follows:
|Parameter					| Value   |
|:-----:|:-----------------:|
| Epochs| 35|
| batch size| 150|
| learning rate| 0.001|

The weights were initialized with a normal truncated distribution, with *μ = 0* and *σ = 0.1*.
The bias terms were all initialized as zeros.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 99.3
* validation set accuracy of 98.5
* test set accuracy of 96.6

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

##Test a Model on New Images

I download a set of images from google street view, particularly from the intersection between untermainbrücke and mainkai in frankfurt, germany. Also I append to the set some pictures shared by our classmate Sonja Krause-Harder. 
Here are some of the pictures taken:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Then I cropped out 40 traffic signs. 

There are a couple of image I found peculiar and hard to identify.
the first one comes from this picture:
![alt text][image6] 

As shown above, there are two signs of children crossing, but one is the flipped version of the official Sign. the training set, test set and validation set don't have any flipped version of the children crossing sign. therefore the model is prone to make a mistake here.

The second image is the following:

In this image, I don't even understand what does it mean, for me, it looks like a double negation no entry Sign, the interesting thing is that if you look at the center there is a Yield sign within the original Traffic Sign, so definitely the program will have troubles there.

###Model's Predictions

2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
