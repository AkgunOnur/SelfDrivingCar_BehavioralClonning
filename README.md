# **Behavioral Cloning** 
---
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/architecture.JPG "Model Architecture"
[image2]: ./examples/center.png "Image from center camera"
[image3]: ./examples/center_flip.png "Flipped image from center camera"
[image4]: ./examples/center_augment.png "Distortion applied on the image"
[image5]: ./examples/left.png "Image from left camera"
[image6]: ./examples/right.png "Image from right camera"
[image7]: ./examples/hist.png "Frequency of the steering angles"
[image8]: ./examples/left_augment.png "Distorted left image"
[image9]: ./examples/right_augment.png "Distorted right image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* Behavioral Cloning Projec.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

behavioral_cloning.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline which is used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model consists of a convolution neural network with 5 convolution layers and 4 flattened layers. Convolutional layers are 5x5x24, 5x5x36, 5x5x48, 3x3x64 and 3x3x64 sized filters. Flattened layers have 100, 50, 10, 1 nodes respectively. In first step, image data is normalized by Keras lambda layer. Afterwards, region of interest is extracted by Keras Cropping2D layer. In each layer, ReLu activation function is used to introduce nonlinearity in the model. 

#### 2. Attempts to reduce overfitting in the model

One of the indications of overfitting is to have very low loss value in training set and high loss value in validation set. 
The dropout layer is added after each hidden layer in order to reduce overfitting. According to the obtained results, dropout coefficient and where the regularization process will be executed is found. 

#### 3. Model parameter tuning

Adam method is used as an optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

In the data set there are three images which are center, left and right images for each moment. The three of them are used in the model. 80% of the data is used as training set and 20% of it is used as validation. 

For details about what kind processes are executed, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The very first important part of this study is to organize data used in the model. To get a model that correctly works, one needs to have a balanced dataset. Otherwise, the car will not complete the track regardless of the model. In order to have a balanced dataset, augmented images are added. Data augmentation process includes rotation, translation, shearing and addition of brightness. Correction factor is added to the steering angle of left image and it is subtracted for right image. 

In second step, Nvidia Self Driving Car architecture[1], which is described in section 1, is introduced. Creating a model from the scratch may not give always best results. Therefore, pre-trained models such as Alexnet, GoogLenet VGG etc. or model structures which gave better results before are preferred. 

According to a more recent research [2], using dropout layers after both convolutional and hidden layers may be useful. Therefore, after each convolution process, dropout layer, whose coefficient to drop is relatively small, is added. After each hidden unit dropout layer whose coefficient to drop is 0.5, is added 

At the end of the process, the car is driven autonomously by the model.h5 file and it is recorded to the output.mp4 file. 

#### 2. Final Model Architecture

The final model architecture consists of a layers and layer sizes


| Layer         		|     Description	        					| 
|:-----------------:|:-----------------------------------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Lambda    	| Each image is normalized between -0.5 and 0.5 	|
| Cropping2D    	| Region of interest is extracted for each image, output: 65x320x3|
| Convolution 5x5x24     	| 2x2 stride, valid padding, output: 31x158x24 	|
| RELU					|	Activation method: Rectified Linear Unit    |
| Dropout			| probability: 0.1 	|
| Convolution 5x5x36	    | 2x2 stride, valid padding, output: 14x77x36    |
| RELU          | Activation method: Rectified Linear Unit     |
| Dropout			| probability: 0.1 	|
| Convolution 5x5x48     	| 2x2 stride, valid padding, output: 5x37x48 	|
| RELU					|	Activation method: Rectified Linear Unit    |
| Dropout			| probability: 0.1 	|
| Convolution 3x3x64	    | 2x2 stride, valid padding, output: 3x35x64    |
| RELU          | Activation method: Rectified Linear Unit     |
| Dropout			| probability: 0.1 	|
| Convolution 3x3x64	    | 2x2 stride, valid padding, output: 1x33x64    |
| RELU          | Activation method: Rectified Linear Unit     |
| Dropout			| probability: 0.1 	|
| Fully connected		| (1x33x64,100) sized layer 	|
| RELU				| Activation method: Rectified Linear Unit 	|
| Dropout			| probability: 0.5 	|
|	Hidden layer | (100x50) sized layer   	|
| RELU				| Activation method: Rectified Linear Unit 	|
| Dropout			| probability: 0.5 	|
|	Hidden layer | (50x10) sized layer   	|
| RELU				| Activation method: Rectified Linear Unit 	|
| Dropout			| probability: 0.5 	|
|	Output layer |	(10x1) sized layer	|

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, firstly images from all cameras, whose steering angles are not zero, are obtained. 

![alt text][image2]
![alt text][image5]
![alt text][image6]

To have a larger and more balanced dataset, distorted images from left and right cameras are added to the dataset

![alt text][image4]
![alt text][image8]
![alt text][image9]

To double up the dataset, each image is flipped horizontally.

![alt text][image3]

At the end of this process, the histogram of the steering angle is like below:

![alt text][image7]

After the collection process, there are 28360 number of data points. Then, the data is normalized between -0.5 and 0.5 by Keras Lambda layer. Region of interest is obtained by Keras, Cropping2D layer

During the training process, data is shuffled and split as 80% of which is training set and the rest of which is validation set. 

Because adam optimizer is used, there is no need to tune the learning rate. Training process is executed 5 epochs

### References
[1] NVIDIA, "End to End Learning for Self-Driving Cars", 25-04-2016

[2] Sungheon Park and Nojun Kwak, "Analysis on the Dropout Effect in Convolutional Neural Networks", Asian Conference on Computer Vision, 2016
