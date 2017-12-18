*******************************************************************************************
*******************************************************************************************
*******************************************************************************************

This project(CS F469) is done by Siddharth Mundada and Ashutosh Upreti.

*******************************************************************************************
*******************************************************************************************
*******************************************************************************************

===========================================================================================
===========================================================================================



Feature selection is done on three datasets, thus we have three folders.

Every folder has two python files
1. classCombine.py
2. feature_util_class

Packages used are:
1. scikit-learn
2. numpy 
3. pandas
4. seaborn
5. matplotlib
6. json
7. os
8. sys

===========================================================================================
===========================================================================================


Description of classCombine.py:

This is the driver script. Three classifiers are used for every dataset and 5 feature selection techniques.
We need to manually set them accordingly for desired combination of technique and classifier.


===========================================================================================
===========================================================================================

Description of feature_util_class.py

This file has all of the feature selection techniques used in the project.
One class python class implements all of these techniques. Other methods in this file are for classification which are imported in classCombine.py
Almost all functions have a proper documentation.


===========================================================================================
===========================================================================================
Every dataset folder has .json files which are used to load the pre-trained parameters. Do not 
modify these file content or name.



===========================================================================================
===========================================================================================
webkb folder has file named f1.py which is faster version of all the feature_util_class.py 
files.
