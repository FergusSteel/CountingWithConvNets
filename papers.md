# Papers

* Investigating the Ability of CNNs to Count Visual Concepts
* Fergus Steel
* 2542391s
* Dr. Paul Siebert

Summary of papers read with notes and links.

## Counting and locating high-density objects using convolutional neural networks

<https://www.sciencedirect.com/science/article/pii/S0957417422000549>

* Uses a three stage model to estimate density map that can be used to count objects
* Trained on a dataset of trees
* Feature Map Extraction using standard ConvNet
* Pyramid Pooling Module that concatenates many pooling layer outputs to encode local and global information.
* Multi-Sigma Stages Module that estimates the density map using gaussian kernels around objects, i.e. an object in the training set will have several gaussian kernels applied to it with decreasing s.d. such that the model can predict the likelihood an objects exists at a given location.
* If the peak in the confidence map at the final stage of the MSS module exceeds a threshold then it is counted as an object.
* Experiments found that T=4 was best number of stages in the MSS module as performance decreased past this point as network got deeper.
* The proposed approach was used on the CARPK and PUCPR+ datasets to evaluate its generality, and performed strongly compared to existing methods.
* The method was able to count and locate even partially occluded objects.
* Proposed future work - using difference distribution kernels.

## Focal Loss for Dense Object Detection (RetinaNet)

<https://arxiv.org/pdf/1708.02002v2.pdf>

* Two-Stage (R-CNN) vs One-Stage (YOLO) Object Detectors and their differences are evaluated.
* Two-Stage methods use proposals where a set of candidate locations are suggested and then they are classified in a second stage
* RetinaNet is a one-stage detector that used Focal loss to significantly improve performance.
* Class Imbalance (more background classes) is the main obstacle preventing one stage detectors from outperforming two stage detectors.
* Focal Loss means that the training will focus on hard examples therefore coutneracting class imbalance

## A survey of crowd counting and density estimation based on convolutional neural network

* Researchers have proposed many methods for this task but the main ones are detection-based, regression-based and density estimation.
* Using CNNs in crowd density esimtation works as they have the ability to learn non-linear relations.
* Embedded devices can be unsuitable for CNN architectures as the have many parameters still and require a lot of resources.
* Detection-Based CNN
  * Trained using image dataset annotated by bounding boxes. (locates/detects the people in the input)
* Regression-Based CNN
  * Trained by dataset annotated by point or using unsupervised methods. (Directly estimates the amount of people or the dnesity map).
* Mentions paper that uses LSTMs in crowd counting <https://ieeexplore.ieee.org/document/7780624>
* Domain Adaptation Model - these methods can count in any object domain

## End-to-end people detection in crowded scenes

* Combines RNN and CNN and uses LSTMs as a sort of "controller" that "propogates information between decoding steps and control location of the next output"
* Like Faster R-CNN directly predicts instances of objects (no classification afterwards)
* Related to OverFeat (maybe look into that idk)
* Uses a regression module to generate boxes.
* CNN -> RNN with LSTMs. Each step, LSTM outputs a new bounding box (woah). This stops when LSTM cannot create a bounding box that is above a certain confidence threshold.

## Lightweight convolutionla neural network for counting densely piled steel bars

* Research aimed at developing low cost network that can accurately count DPSB on a handheld device at construction site.
* Were able to reduce numer of parameters and computation cost by 76.86% and 44.82% respectively.
* 66 fps, 99.25% F1 Score.
* Used YoloV5 as backbone of network for high speed, high accuracy.
* "Modern object detector is usually composed of... a backbone for feature extraction, a neck for fusing feature maps of different scales, several heads for outputting detection information."
  * This is in-line with the density estimator paper that used a CNN for feature extraction, a PPM for the fusing of feature maps and the MSS for estimating the confidence maps.
