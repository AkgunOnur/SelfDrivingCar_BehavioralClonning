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

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* behavioral_cloning.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

model.h5 file can be downloaded via [this](https://drive.google.com/file/d/1tDUiSfOtHz5QGUoLxUGV5N_LuZDJiNZa/view) due to large file size

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

behavioral_cloning.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline which is used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model consists of a convolution neural network with 5 convolution layers and 5 flattened layers. Convolutional layers are 5x5x24, 5x5x36, 5x5x48, 3x3x64 and 3x3x64 sized filters. Flattened layers have 1164, 100, 50, 10, 1 nodes respectively. In first step, image data is normalized by Keras lambda layer. Afterwards, region of interest is extracted by Keras Cropping2D layer. In each layer, ReLu activation function is used to introduce nonlinearity in the model. 

#### 2. Attempts to reduce overfitting in the model

One of the indications of overfitting is to have very low loss value in training set and high loss value in validation set. 
The dropout layer is added after convolutional layers in order to reduce overfitting. According to the obtained results, dropout coefficient and where the regularization process will be executed is found. 

#### 3. Model parameter tuning

Adam method is used as an optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

In the data set there are three images which are center, left and right images for each moment. The three of them are used in the model. 80% of the data is used as training set and 20% of it is used as validation. 

For details about what kind processes are executed, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The very first important part of this study is to organize data used in the model. In order to create a model that actually works left, right and center images are added to the dataset. Correction factor is added to the steering angle of left image and it is subtracted for right image. Data is augmented by rotating, translating and adding brightness the actual images so that the whole set is more balanced. 

In second step, Nvidia Self Driving Car architecture, which is described in section 1, is introduced. Because it actually gives better results than other models. 

Despite the fact that everything looks working and low loss values are obtained, the car could not complete the track. In order to fix this problem, the instances, whose steering angle is zero, are eliminated. By this means, the car completed the track successfully although the loss value obtained is bigger than before. 

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
| Convolution 5x5x36	    | 2x2 stride, valid padding, output: 14x77x36    |
| RELU          | Activation method: Rectified Linear Unit     |
| Convolution 5x5x48     	| 2x2 stride, valid padding, output: 5x37x48 	|
| RELU					|	Activation method: Rectified Linear Unit    |
| Convolution 3x3x64	    | 2x2 stride, valid padding, output: 3x35x64    |
| RELU          | Activation method: Rectified Linear Unit     |
| Convolution 3x3x64	    | 2x2 stride, valid padding, output: 1x33x64    |
| RELU          | Activation method: Rectified Linear Unit     |
| Dropout			| keep_prob: 0.7 	|
| Fully connected		| (1x33x64,1164) sized layer 	|
| RELU				| Activation method: Rectified Linear Unit 	|
|	Hidden layer | (1164x100) sized layer   	|
| RELU				| Activation method: Rectified Linear Unit 	|
|	Hidden layer | (100x50) sized layer   	|
| RELU				| Activation method: Rectified Linear Unit 	|
|	Hidden layer | (50x10) sized layer   	|
| RELU				| Activation method: Rectified Linear Unit 	|
|	Output layer |	(10x1) sized layer	|

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, firstly images from center camera are obtained. 

![alt text][image2]

The instances,whose steering angle is not zero, are only considered. To have a large amount of data, images from left and right cameras are added to the dataset and the new images are obtained by distorting the actual ones. 

![alt text][image5]
![alt text][image6]
![alt text][image4]

To double up the dataset, each image is flipped horizontally.

![alt text][image3]

After the collection process, there are 22050 number of data points. Then, the data is normalized between -0.5 and 0.5. Region of interest is obtained by Keras, Cropping2D

During the training process, data is shuffled and split as 80% of which is training set and the rest of which is validation set. 

Because adam optimizer is used, there is no need to tune the learning rate. Training process is executed 5 epochs
