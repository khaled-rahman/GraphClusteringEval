# Evaluating The Community Structures From The Images Of Networks Using Deep Learning #
The source code for evaluating the community structures from the images of networks using deep learning. Please follow the instructions to generate training images, run CNN model, and get plots. Our generated datasets can be found [here](https://drive.google.com/drive/folders/1B_pLKlsbSqzkyDgnbwMms9cGOZD4JDIk?usp=sharing).

## Required Software packages/tools ##
User needs to  make sure that your machine has following version/packages installed:
```
Python version >= 3.6
PyTorch version >= 1.0.1
```
Other python packages are pandas, matplotlib, torchvision, sklearn, numpy, skimage, community, networkx.

## Generate Images ##
Go to datasetgen director. Before getting started, create a directory where generated images will be stored. In this version, images will be saved in `dataset` directory. So create a folder named `dataset` as following:

```
$ mkdir dataset
```
To generate an image, type the following command:
```
$ python clusterdatagen.py ./inputnetworks/network_2.mtx 1
```
Here, arguments to python file are inputnetwork and an id for this run. To generate bulk image dataset, user can use a script as following commands:
```
$ bash runimagegen.sh
```
This will generate images in dataset folder. Input graph will be collected from inputnetworks folder. This will also generate a groundtruth.txt file which will contain information of labels for each image. A sample groundtruth file and few images are given in this repository. As real-dataset is small in size, we have provided images and corresponding groundtruth file for this. 

## Train and Test deep learning Models ##

To train and test the models, move back to GraphClusteringEval directory and type the following command:
```
$ python classification/trainCNNmodel.py
```

This will read data from datasetgen folder, train and test the models, and generate values of loss, accuracy and f-beta measure. This will also generate plots for different number of epochs. Note that datasets will be divided into 70% for training, 20% for validation and remaining 10% for testing.

## Results ##
Our results, log files are available in results directory. User can check out this also.

## Citation ##
If you find this repository helpful, please cite the following paper:

Md. Khaledur Rahman and Ariful Azad, "Evaluating the Community Structures from Network Images using Neural Networks", In the 8th Proceedings of International Conference on Complex Networks and their Applications (Complex Networks 2019), December, 2019, Lisbon, Portugal.
```
@inproceedings{rahman2019evaluating,
  title={Evaluating the Community Structures from Network Images Using Neural Networks},
  author={Rahman, Md Khaledur and Azad, Ariful},
  booktitle={International Conference on Complex Networks and Their Applications},
  pages={866--878},
  year={2019},
  organization={Springer}
}

```


### Contact ##
If you have any question or comments, please do not hesitate to send me (Md. Khaledur Rahman) an email at `morahma@iu.edu`.
