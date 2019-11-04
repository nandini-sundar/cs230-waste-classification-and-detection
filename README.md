# CS230 - Waste Classification and Localization with Deep Learning

## Classification (Waste_Classification_(Resnet34).ipynb)

For this task, we used [TrashNet dataset](https://github.com/garythung/trashnet/blob/master/data/dataset-resized.zip)  collected manually by Gary Thung and Mindy Yang.

We referred to [this](https://towardsdatascience.com/how-to-build-an-image-classifier-for-waste-sorting-6d11d3c9c478) blog post by Collin Ching for doing this part.

**We did transfer learning on our dataset using resnet34 model for 40 epochs and got 93.2% accuracy**

We also plotted the losses (training/validation) and metrics (accuracy, error rate, precision and recall)

## Object Detection and Localization (Waste_Object_Detection_Using_Tensorflow.ipynb)

For this task, we did data augmentation on [TrashNet dataset](https://github.com/garythung/trashnet/blob/master/data/dataset-resized.zip). We took 100 images from each class (glass, cardboard, plastic, paper, metal, trash), cropped it and created a collage of it on a 300x300 white canvas and also calculated bounding boxes of where the object is on the canvas and used this as the dataset to train our model.

The data and scripts related to the above data augmentation process are in [this](https://github.com/nandini9cs230/cs230_waste_object_detection_data) repository

We referred to [this](https://www.dlology.com/blog/how-to-train-an-object-detection-model-easy-for-free/) blog post by Chengwei for doing this part.

**We did transfer learning on the augmented dataset using faster_rcnn_inception_v2 model for around 2000 epochs and achieved a loss of 1.0686216**

### Evaluation Metrics

These are our metrics at global step 1871 and learning rate 0.0002:

#### DetectionBoxes Precision :

* mAP = 0.32529658</br>
* mAP (large) = 0.3253003</br> 
* mAP@.50IOU = 0.5948102</br>
* mAP@.75IOU = 0.32665467</br>

#### DetectionBoxes Recall :

* AR@1 = 0.465474</br> 
* AR@10 = 0.642222</br>
* AR@100 = 0.65055126</br>
* AR@100 (large) = 0.65055126</br>

#### Losses :

BoxClassifierLoss :
* classification_loss = 0.6357156
* localization_loss = 0.5054222

RPNLoss :
* localization_loss = 1.2740649
* objectness_loss = 0.2569142

Total Loss : 2.6721172
