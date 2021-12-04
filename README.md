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

<ins>Column Inferencing</ins>
<p>The main purpose of column inference is to use deep learning models trained by image data to predict missing values in columns.  The main strategy is to treat a particular tabular column as a class label, and then train images in deep learning models to predict the value of that column.
<p>Both GoogleNet and DenseNet121 have relatively better overall performance on inferencing column values of our tabular data.  All results show the column biopsy has the highest inferencing scores, so it indicates column biopsy has strong correlation with skin lesion types.  A possible explanation is that a doctor may tend to order a skin biopsy when the patient is suspected to have skin cancer after viewing the patient’s skin image.
<br><img src="figures/column_inferencing.jpg" style="float: left; margin-right: 10px;"  />

<ins>Hybrid Model architecture</ins>
<p>In order to train a customized deep learning model with the combined tabular and image data, we have constructed a hybrid model named ImageTabularHybridModel, which extends Pytorch’s nn.Module class.  The hybrid model can be configured and initialized by a user parameter to train image only data, tabular only data, or combined image and tabular data.  Another user parameter specifies what the underlying prebuilt CNN model that the hybrid model should utilize to train image data.  Currently, it supports the four best candidates: ResNet18, GoogleNet, DenseNet121, and MobileNet V2.  When the hybrid model is initialized to train combined image and tabular data, it pushes image data to go through the prebuilt CNN model specified by the user parameter, and the last layer of the original prebuilt CNN model is hijacked and extended to connect 3 fully connected layers with Relu activation.  The tabular data is pushed through another 3 fully connected layers with Relu activation.  Then the output data of the image layer is concatenated with the output data of the tabular layer, and such concatenated data is further pushed through additional 3 fully connected layers with Relu activation.  The last layer outputs 6 nodes that represent 6 types of skin lesions.
<br><img src="figures/hybrid_model_architecture.jpg" style="float: left; margin-right: 10px;"  />
