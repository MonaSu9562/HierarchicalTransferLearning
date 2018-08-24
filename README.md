# HierarchicalTransferLearning
Hierarchical Transfer Component Analysis and Hierarchical Joint Domain Adaption are implemented in this repository. The code is written in Python with Jupyter Notebook. The other functions used contains the *tca.py* and *MyJDA.m*.

# Datasets
The datasets used in this dissertation is available in the data folder. The data from House A, B, and C are provided in the form of '.csv'. And the similarity matrix used is also provided in the form of '.csv', and named as 'Sensor_sim_XX'.

# HTCA and HJDA
These two files are written by Mengyuan Su individually, in each file a function called *runhtca()* or *runhjda()* is provided respectively.

The input contains: 
- x_src: the data from the source domain 
- y_src: the labels of the source domain
- x_tar: the data from the target domain
- y_tar: the true labels of the target domain (This is only used for calculating the accuracy and not being used during the HTCA or HJDA procedure.)
- options: the options for each method. The demo of how to define the options is shown in the code.

The output contains:
- acc_can: the accuracy of candidates
- acc_res: the accuracy of residuals
- acc_stl: the overall accuracy

To run HTCA, packages numpy, pandas, sklearn, and random need to be installed. And tca.py and tools.py should be imported.

To tun HJDA, scipy is further required to save and read '.mat' data. And MATLAB engine for Python is required as well. 

# Tools
*tools.py* is written by Mengyuan Su for calling the functions that are used both in HJDA and HTCA. 

It contains:
- Model: this is a class for saving the information of basic classifiers. The model in sklearn packages, model name, and both source domain and target domain is required to initialize a new Model. It will fit the classifier with the source domain and predict on both source and target domain.
- Models: this is also a class for comparing the performance of classifiers according to accuracy. It has several functions: *add()*, to add a new Model in the Models. *show_src()*, shows the classification accuracy of the multiple classifiers on source data while *show_tar()*  shows the classification accuracy of the multiple classifiers on target data. All the classes that the prediction contains and corresponding counts are also shown together with the result. Moreover, *sort_src()* and *sort_tar()* will sort the classifiers according to their classification accuracy on the source and target data and show the result.
- remove_files(): is a function for removing the '.mat' documents generated during the JDA procedure. As the data form from *numpy* in Python and the data form from MATLAB are different, the data should be changed to the form of '.mat' and then passed to the MATLAB engine.
- getdata():is a function require the *x_src*, *y_src*, *x_tar*, *y_tar* as input. User need to define the argument of *balance=True* or *z_score=True* to pre-process the data. The *x_src*, *y_src*, *x_tar*, *y_tar* that have been balanced or standardized is returned.
- voting(): is a function used during HTCA and HTJA to do the majority voting part.
