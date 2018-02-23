# US-power-plants
Analysis of EIA provided data of power plants in US

About:
The project compiles a data set around power plants within the US including plant attributes, geographic characteristics, and population information. All sources are included in the Sources folder.The projects is split between the Dataset Compilation, which cleans and consolidates the disparate sources, and Analysis, which performs feature engineering and machine learning.

Modules

DataSet_compiler.py
 numpy
 pandas
 os
 matplotlib.pyplot
 time
 math as m

Analysis
 pandas
 matplotlib.pyplot
 os
 numpy
 sklearn import preprocessing, base, model_selection, utils, linear_model
 sklearn.neighbors import KNeighborsClassifier
 sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer
 sklearn.naive_bayes import MultinomialNB, GaussianNB
 sklearn.cluster import KMeans

