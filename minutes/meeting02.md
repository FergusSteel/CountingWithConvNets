# Meeting Minutes - 2542391s

## Meeting Description

**Date/Time:** 27/09/2023 14:00:00
**Meeting Number:** \#02
**Meeting Location** Online

## Meeting Agenda

| Talking Point 1 | Description |
| ----- | ----- |
| Demonstration of Density Map Estimation | -  |
| Directions for Proj | - Prompt-guided object counting: Combine Language models and Deep CNNS such that a model can analyse an image and the end user can then prompt the CNN to count a certain object. Allows for investigation into how the CNN "sees" the image and into how this can be manipulated by language. This idea would be more impressive if could run on video frames. i.e. "highlight 3 people in this video" and it woudl remove the masks for other people and track those 3. <br> - Visualisation attention maps in a counting task, to see how the object attends to a given object. <br> - Combining CNNs and NARX/LSTM networks to combine spatial and temporal information in crowd counting tasks, i.e. Using CCTV dataset, by incorporating temporal information can we improve the accuracy of count? Generally investigating the effect of treating videos as time series rather than individual frames in the context of Computer Vision|

## Research Direction

Investigating the effect of incorporating temporal information into Convolutional Neural Networks tasked with counting visual concepts.

In the wide berth of research in regards to convolutional neural networks and counting concepts in videos, most architectures use state of the art models that can count accurately and apply them to each frame of data individually. While some work has been done in crowd counting with promising results  (with ConvLSTM), I believe it would be fascinating to reserach if by considering videos as time series with LSTMs/NARX architectures if they are able to improve the accuracy of CNNs.

## Meeting Minutes

14:00 - Hellos
14:05 - Alfie's turn first, alleviating partial eye site loss in VR.
14:20 - Laurie's turn
15:00 - My turn, going over previous action points
15:05 - Going over reserach I looked into
15:10 - Discussing feature pyramids, how do CNNs model visual scesnes + human vision (focal point) disucssion
15:15 - modelling spacial relationships, forced/unforced clasification
15:30 - Neural Cellular Automata discussion and project plans
15:45 - where to go from here, build up in incremental steps. **Jeffrey Hinton**
15:50 - Closing questions and admin questions
16:00 - Paulius turn, more admin questions

## Meeting Notes

* Access to GPUs, ideally accessing machines in Paul's office, "for us to use". Have to contact Richard Menzies and ask for access
* Nice idea, but, get some demonstrations working!!!
* This is a perception task, focusing on FOV.
* Got to bear in mind that, what you can see in front of me, when you are focusing in on something you cant see much.
* Mentioned feature pyramid networks.
* Wants to do it in a more scientific way, looking at elucidating how many simple entities there are? 
* Strucutre it like this: things I can do, can you design a model that can detect and count multi-class simple concepts, then when they are moving can we extend that capabilities.
* Set out plan of how to go about this.
* Early investigation into investigating, the spatial relations of groups of objects. "where are the green objects"
* "forced classification" & "unforced classiication"
* Jeffrey Hinton - Capsule Networks
* Meld conventional front-end techniques (conv nets, getting the visual concepts) and then feeding those into grouping mechanisms
* Building a game-of-life sort of thing, not Neural Cellular Automata but concepts with behaviour.
* The juicy detail - can we encapsulate learned concepts.
* Blending visual space and automata, the visual embedding and the physical change.

## Action Points

### Action Point 1

* Contact Richard Menzies and ask for access to SoCS GPU cluster.

### Action Point 2

* Read about Hentons Capsule Networks

### Action Point 3

* Look into Cellular Automata

### Action Point 4

* Look for systems that I can lift from.

### Action Point 5

* Get something get concrete done or narrow down.
