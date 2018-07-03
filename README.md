# Using Deep learning and Open StreetMaps to find features in aerial images

## Marc Beltrán, Albert Companys

### Masters in Foundations of Data Science - Master's Thesis

The thesis document associated to this project can be found as a pdf file in the root of this repository.

A great amount of the interesting information captured by aerial imagery is still not being used given how labour intensive the processing and annotation of these images is. Despite this, improvements in technology and advancements in the computer vision field have made available tools and techniques that can help make this process semi-automatized. In this project we focus on the use case of extracting roads from aerial imagery. For this purpose, we will study and compare models based on image segmentation using deep learning and RoadTracer, a revolutionary model proposed recently.

### Objetive

Given the big scope of the proposed project, we reduced it to finding a specific type of features from aerial images. This project is dedicated to the study and implementation of techniques for locating roads in orthophotos. By this we mean to, given an input satelite image, create an output which shows the location of the road in the image.

## Structure

This project is devided into two main blocks, the part of sementic segmentation consists of the implementation and application of several deep learning models to road detection, while the second part is an application of the RoadTracer model to Barcelona imagery. 

### Semantic Segmentation





### RoadTracer

In the roadtracer folder there are the python scripts we have tweaked in order to reproduce the results shown in the paper [**RoadTracer: Automatic Extraction of Road Networks from Aerial Images**](https://roadmaps.csail.mit.edu/roadtracer.pdf)  by F. Bastani et al. Instead of Boston, we downloaded and infered the road network for Barcelona and its surroundings. The full implementation of the algorithm and how to use it properly can be found in https://github.com/mitroadmaps/roadtracer. There is also the information on how to obtain the data for your desired region and the Go scripts to obtain it as well.

- [infer.py](roadtracer/infer.py) is the script used to run the iterative graph building algorithm in the region defined
- [model.py](roadtracer/model.py) contains the CNN decision function implementation
- [model_utils.py](roadtracer/model_utils.py) contains the utility functions needed for the model. It is separated for the sake of clarity and modularity.
- [overlap_images.py](roadtracer/overlap_images.py) is the script we have used to create the final visualizations seen in the thesis report with the road network painted over.

![Results of roadtracer for the city of Barcelona](roadtracer/barcelona_overlapped_model.png | 250x250)