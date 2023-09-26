# Meeting Minutes - 2542391s

## Meeting Description

**Date/Time:** DD/MM/YYYY 00:00:00
**Meeting Number:** \#00
**Meeting Location** Online/In-Person

## Meeting Agenda

| Talking Point 1 | Description |
| ----- | ----- |
| Demonstration of Density Map Estimation | -  |
| Directions for Proj | - Prompt-guided object counting: Combine Language models and Deep CNNS such that a model can analyse an image and the end user can then prompt the CNN to count a certain object. Allows for investigation into how the CNN "sees" the image and into how this can be manipulated by language. This idea would be more impressive if could run on video frames. i.e. "highlight 3 people in this video" and it woudl remove the masks for other people and track those 3. <br> - Visualisation attention maps in a counting task, to see how the object attends to a given object. <br> - Combining CNNs and NARX/LSTM networks to combine spatial and temporal information in crowd counting tasks, i.e. Using CCTV dataset, by incorporating temporal information can we improve the accuracy of count? Generally investigating the effect of treating videos as time series rather than individual frames in the context of Computer Vision|

## Research Direction

Investigating the effect of incorporating temporal information into Convolutional Neural Networks tasked with counting visual concepts.

In the wide berth of research in regards to convolutional neural networks and counting concepts in videos, most architectures use state of the art models that can count accurately and apply them to each frame of data individually. While some work has been done in crowd counting with promising results  (with ConvLSTM), I believe it would be fascinating to reserach if by considering videos as time series with LSTMs/NARX architectures if they are able to improve the accuracy of CNNs.

## Meeting Minutes

## Meeting Notes

| Item | Notes |
| ---- | ---- |
| Item 1 | notes |
| Item 2 | notes |

## Issues Discussed / Potential Issues

## Further Notes

Insert further notes here....

## Action Points

### Action Point 1

### Action Point 2
