# UMBC Data 606 Project
# Skin Cancer Analysis and Classification by Machine Learning and Deep Learning

<img src="figures/logo.jpg" alt="Cancer Cell" style="float: left; margin-right: 10px; width:100%; height: 300px; "  />

## Team Members: 

- Demarcus Wirsing
- Drishti Arora
- Nigist Woldeeyesus
- Sai Kumar Kanuru
- Yunpeng Li

## Project Goals:

- Train various machine learning models on clinic tabular data to predict and evaluate the results of skin lesion classifications.
- Use image processing techniques and construct various deep learning models on skin image data to evaluate the results of skin lesion classifications.
- Use skin cancer image data to inference missing clinic tabular column values through deep learning on image data. 
- Construct a deep leaning hybrid model that combines both image and tabular data to do skin lesion classifications.
- Prove or disapprove combining clinic tabular data and skin lesion image data in deep learning can achieve better accuracy score.
- Evaluate the accuracy and performance of various models in the end and find the best model for skin lesion classifications.

## Project Highlights
### Machine Learning
TODO

### Deep Learning

<ins>Column Inferencing aaa</ins>
<p>The main purpose of column inference is to use deep learning models trained by image data to predict missing values in columns.  The main strategy is to treat a particular tabular column as a class label, and then train images in deep learning models to predict the value of that column.
<br><img src="figures/column_inferencing.jpg" style="float: left; margin-right: 10px;"  />

<ins>Hybrid Model architecture</ins>
<p>It connects one of the prebuilt CNN models (ResNet18, GoogleNet, MobileNet V2, DenseNet121) for processing image data and fully connected layers with Relu activation for processing tabular data.
<br><img src="figures/hybrid_model_architecture.jpg" style="float: left; margin-right: 10px;"  />
