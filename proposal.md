# Research Proposal

* Fergus Steel
* 2542391s
* Dr. Paul Siebert
------------

## Research Outline

* Brief: Investigating the Ability of CNNs to Count Visual Concepts
* Research Direction: Investigating the encapsulation of visual conecepts in Capsule Networks and the resulting effect on class-agnostic object counting?

## Conceptual Overview - What is involved in this research? / Research Justification - What is the rationale for this research?

### Related Work: Object Counting, Convolutional Neural Networks 

Object Counting is one domain of vision research where human's are remarkably capable. The human brain is able to impulsively incorporate visual cues such as object density and group size to count objects at a glance. Additionally the human brain is able to subitize and then rapidly, and accurately, produce an estimated count of smaller groups of objects (Revkin et al., 2008).

In Computer Vision, however, these impulsive systems don't exist and therefore they must be emulated or alternative techniques must be used. Fan et al. (2022), discusses the two methods that are typically used in crowd counting, which is a sub-domain of object counting research: 
* Detection-based CNNs, where CNNs are trained on an image dataset which are annotated by bounding boxes, thus training a model to detect, localize and therefore count each instance of the trained object in the scene.
* Regression-based CNNs, where the CNNs are trained on a dataset of point-annotated images to directly estimate the count or more typically, the density map, which is a normalized heatmap that when integrated, provides a count of the map.
Density-Map Estimation is a computationally simpler task, and is able to perform better in more complex scenes i.e. High Density, High Occlusion, Sparse Scenes and Complex Backgrounds. (Gao et el., 2020).

Convolutional Neural Networks are able to achieve state-of-the-art results in object counting thanks to their excellent performance as non-linear function approximaters. One such implementation is outlined in de Arruda et al., (2022), which uses a three stage model to implement density-map esimtaiton. Which uses a typical Convolutional Neural Network to extract the features of an input image, which are then passed into a Pyramid Pooling Module that constructs a hierarchical feature map of different spatial resolutions, which is then passed to a Multi Stage Sigma module that uses multiple gaussian distributions (that select from multiple variance parameters) to create a high quality density map that can be used to extract the count of object in the image. This architecture was originally trained and tested on a new UAV imagery dataset, and was then also trained on popular car counting datasets (CARPK and PUCPR+) to show its generalisability. In both cases, the model was able to achieve state-of-the-art performance in object counting, showing the viability of density map estimation as a method for object counting. 

### Class-Agnostic Counting (Exemplar-based Few Shot Counting and GAN-Based Zero-Shot Counting) & Capsule Networks

The issue with the proposed architecture above, and other CNN-Based density-map estimation object counting, is that they are trained upon a dataset, that when presented at test time, with an input image that contains a visual concept that does not exist in the training sets distribution, it will be unable to "generalise" and count it successfully. Additionally, to retrain these networks on a new dataset, this typically requires millions of annotations on thousands of images, which requires a lot of labour to create these datasets and further computational power in the training process (Ranjan et al., 2022).


### Why are Convolution Neural Networks used in the first place?

### What is wrong with Convolutional Neural Networks?

### What are Capsule Networks and How do they Networks address these problems?

### Why are Capsule Networks potentially suitable architecture's for Class-Agnostic counting?


## Bibliography

* de Arruda, Mauro dos Santos, et al. "Counting and locating high-density objects using convolutional neural network." Expert Systems with Applications 195 (2022): 116555.
* Fan, Z., Zhang, H., Zhang, Z., Lu, G., Zhang, Y., & Wang, Y. (2022). A survey of crowd counting and density estimation based on convolutional neural network. Neurocomputing, 472, 224-251.
* Gao, G., Gao, J., Liu, Q., Wang, Q., & Wang, Y. (2020). Cnn-based density estimation and crowd counting: A survey. arXiv preprint arXiv:2003.12783.
* Viresh Ranjan, Udbhav Sharma, Thu Nguyen, Minh Hoai; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 3394-3403
* Revkin, S. K., Piazza, M., Izard, V., Cohen, L., & Dehaene, S. (2008). Does Subitizing Reflect Numerical Estimation? Psychological Science, 19(6), 607-614. https://doi.org/10.1111/j.1467-9280.2008.02130.x
