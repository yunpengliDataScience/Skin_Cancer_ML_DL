
#Streamlit libs
from random import choices
from altair.vegalite.v4.schema.core import Align
from pandas.io.parsers import PythonParser
import streamlit as st
import warnings
from PIL import Image
#Machine learning libs
import sys
import os
#from imageio import imread
from PIL import Image
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, title
from sklearn import preprocessing
import altair as alt
#machine leaning libraries
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier 
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from numpy import mean
from numpy import std
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_squared_error

import statsmodels.api as sm

        
# Stepwise regression Foraward and backward
def forward_regression(X, Y,threshold_in):
    initial_list = []
    included = list(initial_list)
    while True:
        changed=False
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded,dtype='float64')
        for new_column in excluded:
            model = sm.OLS(Y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
    
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            print('Included variables  {:30} with p-value {:.6}'.format(best_feature,format( best_pval,'.4f')))

        if not changed:
            break
    return included
    
def backward_regression(X, Y,threshold_out):
            included=list(X.columns)
            while True:
                changed=False
                model = sm.OLS(Y, sm.add_constant(pd.DataFrame(X[included]))).fit()
                # use all coefs except intercept
                pvalues = model.pvalues.iloc[1:]
                worst_pval = pvalues.max() # null if pvalues is empty
        
                if worst_pval > threshold_out:
                #print(worst_pval)
                    changed=True
                    worst_feature = pvalues.idxmax()
                    included.remove(worst_feature)
                    print('Removed variable {:30} with p-value {:.6}'.format(worst_feature, format(worst_pval, '.4f')))
                if not changed:
                    break
            return included


st.set_page_config(
    page_title="Skin Cancer Application",
    page_icon='https://th.bing.com/th?q=Medicine+Cartoon+Image&w=120&h=120&c=1&rs=1&qlt=90&cb=1&pid=InlineBlock&mkt=en-US&cc=US&setlang=en&adlt=moderate&t=1&mw=247',
    layout='wide',
    menu_items={'Get help': "https://docs.streamlit.io/",
                ' Report a bug': "https://github.com/yunpengliDataScience/Skin_Cancer_ML_DL/issues",})

warnings.filterwarnings('ignore')
menu = ['Home','EDA and ML','Deep Learning']

with st.sidebar:
         choice = st.selectbox("Please make your selection below",menu)

col1, col2, col3 = st.columns(3)

def main():
    
    if choice == 'Home':
        st.title("Skin Cancer Machine Learning and Data Analysis App")
        st.caption('This App was built by Demarcus, Yunpeng, Drishti, Nigisti, and Sai')
        
        with st.container():
            with st.form("my_form1"):
                st.header("Project Road Map")
                with st.expander('Phase 1',expanded=True):
                    st.subheader("Why skin cancer?")

                    st.write(""" - Skin cancer is one of the most frequent types of cancer and manifests mainly in areas of the skin most exposed to the sun. 
                    Since skin cancer occurs on the surface of the skin, its lesions can be evaluated by visual inspection. 
                    Dermoscopy is a non invasive method which permits visualizing more profound levels of the skin as its surface reflection is removed. 
                    Prior research has found that this technique permits improved visualization of the lesion structures, enhancing the accuracy of dermatologists.""")

                    st.write(""" - The aim this project is to study the problem of classification of dermoscopic images of skin cancer, 
                    including lesions by using various of machine learning and deep learning techniques.""")

                    st.write(""" - The hypothesis of this project is that combining skin lesion image data with more tabular descriptions or patient data may improve skin lesion classifications and skin cancer diagnosis.""")

                    st.write(""" - PAD-UFES-20 dataset: a skin lesion dataset composed of patient data and clinical images collected from smartphones.  
                    The metadata.csv contains more attributes in tabular format.
                    Source: https://data.mendeley.com/datasets/zr7vgbcyr2/1
                    2298 skin lesion images
                    Nearly 3 gigabytes in total size
                    Tabular data is stored in the metadata.csv file that contains 26 columns""")
                    
                    st.subheader("""Tabular Data""")
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\TabularData.png')
                    st.image(image, caption='Figure 1: Depiction of what our csv file appears to be')
                    
                    st.subheader("""Image Data""")
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\ImageData.png')
                    st.image(image,caption='Figure 2: View of skin lession classification marking matching lession to classification')

                    st.subheader("""Skin Lession classification""")
                    st.write("""The skin lesions are:""")
                    st.json({"Basal Cell Carcinoma (BCC)":"845","Squamous Cell Carcinoma (SCC)":"730","Actinic Keratosis (ACK)":"244","Seborrheic Keratosis (SEK)":"235","Melanoma (MEL)":"52"})
                    
                    
                    st.subheader("""Data Description""")
                    st.write(""" - The dataset consists of 2,298 samples of six different types of skin lesions. 
                    Each sample consists of a clinical image and up to 22 clinical features including the patient's age, skin lesion location, 
                    fitspatrick skin type, and skin lesion diameter. """)
                    st.write(""" - In total, approximately 58% of the samples in this dataset are biopsy-proven. This information is described in the metadata.""")
                    
                    st.subheader("""Data Exploration""")
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\Data Exploration bar graph.png')
                    st.image(image,caption='Figure 3: The total number of skin lession classes shown in bar format')

                    st.subheader("""Feature Engineering""")
                    st.write("""- Metadata and image data downloaded separately from mendeley.com and merged to one dataframe in google colab.""")
                    st.write("""- Transformed the text data to numerical values using label encoder""")
                    st.write("""- 34% of missing values from in 13 columns has been handled by imputing random category “Unknown”.""")
                    st.write("""- Using Dython library identified the association between the categorical variables for the metaData  """)
                    st.write("""- Strongly correlated fields are elevation with bleed, Biopsy with Diagnostic, fitspatrick skin type with father,mother background and gender, image diameter 1 
                    with diameter 2.""")
                    
                    st.subheader("""Correlation Matrix Heatmap""")
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\Heatmap.png')
                    st.image(image,caption='Figure 4: Showing the correlation score between columns')

                    st.subheader("""Similar Projects or Approaches""")
                    st.write("""- Some similar skin cancer projects have used HAM10000 dataset from Harvard Dataverse, which is available at Kaggle.com.""")
                    st.write("""- However, those projects mainly or purely focus on image classifications, and there are not sufficient information for describing attributes of skin lesions or other 
                    patient conditions in the dataset.  
                    So lacking the descriptions of other characteristics of skin lesions besides images may make diagnostic less accurate in real practice. """)
                    st.write("""- However, those projects mainly or purely focus on image classifications, and there are not sufficient information for describing attributes of skin lesions or other 
                    patient conditions in the dataset. So lacking the descriptions of other characteristics of skin lesions besides images may make diagnostic less accurate in real practice. """)

                    st.subheader("""Our Project Purpose and Scope""")
                    st.write("""- Since our dataset contains not only image data but also other information about skin samples and patients in a tabular format, 
                    we plan to compare and utilize both tabular data and image data in our models or analysis to see if we can improve skin lesion classifications so as to improve 
                    skin cancer diagnosis.""")
                    st.write("""- We are going to practice various machines learning, deep learning, and image processing techniques in this project.""")

                    st.subheader("""Our Approaches""")
                    st.write("""- We plan to construct various machine learning models or predictive analysis on the tabular data to evaluate the results of skin lesion classifications.""")
                    st.write("""- We plan to use image processing techniques and construct various deep learning models on skin image data to evaluate the results of skin lesion classifications.""")
                    st.write("""- We plan to construct a model that combines both image and tabular data and uses deep learning approach to do skin lesion classifications.""")
                    st.write("""- We are going to evaluate the accuracy and performance of various models in the end and find the best model for skin lesion classifications.""")
                    st.write("""- We will also construct a data web application using streamlit to over take our google collab notebooks, for a more uniform/seamless experience when presenting out project 
                    and findings. """)
                    
                with st.expander('Phase 2',expanded=True):
                    st.subheader("""Project Goals""")
                    st.write("""- Construct various machine learning models or predictive analysis on the tabular data to evaluate the results of skin lesion classifications.""")
                    st.write("""- Use image processing techniques and construct and utilize various deep learning models on skin image data to evaluate the results of skin lesion classifications.""")
                    st.write("""- Construct a model that combines both image and tabular data and uses deep learning approach to do skin lesion classifications.""")
                    st.write("""- Evaluate the accuracy and performance of various models in the end and find the best model for skin lesion classifications.""")
                    st.write("""- Prove or disprove the hypothesis that combining image data and tabular data can improve model accuracy.""")
                    st.write("""- Construct Inference prediction from the image""")

                    st.subheader("""Experimented Feature Engineering on Tabular Data""")
                    st.write("""- Chi square test and correlation: From Sklearn we used SelectKbest module to select features according to the k highest 
                    (ANOVA F-value between label/feature for classification tasks.)scores.""")
                    code = """smoke: 129.364969
 drink: 124.056784
 background_father: 259.449196
 background_mother: 387.070310
 pesticide: 217.131971
 gender: 254.365572
 skin_cancer_history: 215.465226
 cancer_history: 88.942515
 has_piped_water: 58.949159
 has_sewage_system: 82.074732
 region: 105.452255
 itch: 232.420979
 grew: 537.236444
 hurt: 276.618891
 changed: 629.976932
 bleed: 427.057099
 elevation: 233.440734
 biopsed: 623.995996
 age: 3448.582874
 fitspatrick: 71.561837
 diameter_1: 67.381549
 diameter_2: 73.623881"""
                    st.code(code,language='python')

                    st.subheader("""Stepwise Regression Test""")
                    st.write("""- Applied Stepwise regression:To identify the best predictor variables for the dataset that feeds to the ML models. 
                    Out of 20 variables 6 variables with p-value <0.05 identified as high predictors for skin cancer diagnostics variable.""")
                    code2 = """[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
Included variables  itch                           with p-value 0.0000
Included variables  elevation                      with p-value 0.0000
Included variables  grew                           with p-value 0.0000
Included variables  biopsed                        with p-value 0.0000
Included variables  region                         with p-value 0.0000
Included variables  age                            with p-value 0.0006
Included variables  bleed                          with p-value 0.0033
Included variables  changed                        with p-value 0.0048
Included variables  drink                          with p-value 0.0159
['itch','elevation','grew','biopsed','region','age','bleed','changed','drink']"""
                    st.code(code2,language='python')
                    
                    st.subheader("""Feature Importance""")
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\FeatureImportance.png')
                    st.image(image,caption='Figure 1: Showing the scale of importance for each column in the dataset')
                    
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\FeatureImportance.png')
                    st.image(image,caption='Figure 2: Showing the scale of importance for each column in the dataset continued')
                    
                    st.subheader("""Best Predictor variables""")
                    st.write("""After running multiple statistical test to identify the best predictors variable the following variables were used to train the machine learning model:""")
                    st.json({'Age':'','Biopsed':'','grew':'','Region':'','pesticide':'', 'skin_cancer_history':'', 'cancer_history':'', 'background_father':'', 'drink':'', 'smoke':'','itch':'','bleed':'','Changed':'','Has_sewage_system':'','elevation':''})
                    st.subheader("""Experimenting Machine Learning Models""")
                    st.write("""Trained scaled tabular data using the following ML models.""")
                    st.write("""- SGDClassifier: 64%""")
                    st.write("""- OVO: test score: 55%""")
                    st.write("""- Randomforest: 83%""")
                    st.write("""- Decision tree: 77.9%""")
                    st.write("""- MultinomialNB: 72%""")
                    st.write("""- Multinomial Logistic regression: 73%""")
                    
                    st.subheader("""Experimented Various Pre-built Transfer Learning Model Architectures for Training Skin Lesion Image Data""")
                    st.write("""Purpose: To explore which model has better classification accuracy on skin image data.""")
                    st.write("""- DenseNet (DenseNet121, DenseNet161, DenseNet201)""")
                    st.write("""- GoogleNet""")
                    st.write("""- ResNet (ResNet18, ResNet50)""")
                    st.write("""- MobileNet V2""")
                    
                    st.subheader("""Why use DenseNet?""")
                    st.write("""- Connects each layer to every other layer in a feed-forward fashion.""")
                    st.write("""- Substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output.""")
                    st.write("""- Alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters""")
                    st.write("""- 4 Model structures – densenet121, densenet161, densenet169, densenet201.""")
                    
                    st.subheader("""DenseNet Results""")
                    st.write("""- DenseNet121: 70%""")
                    st.write("""- DenseNet161: 57%""")
                    st.write("""- DenseNet201: 63%""")
                    
                    st.subheader("""Why use GoogleNet?""")
                    st.write("""- Based on a deep convolutional neural network architecture codenamed 'Inception'""")
                    st.write("""- Convolutions use rectified linear activation.""")
                    st.write("""- Designed with computational efficiency and practicality in mind, so that inference can be run on individual devices including even those with limited computational resources, especially with low-memory footprint""")
                    
                    st.subheader("""GoogleNet Results""")
                    st.write("""- GoogleNet: 63%""")
                    
                    st.subheader("""Why use ResNet""")
                    st.write("""- Deep residual networks pre-trained on ImageNet""")
                    st.write("""- Easier to optimize and can gain accuracy from considerably increased depth.""")
                    st.write("""- 5 Model structures – restnet18, restnet34, restnet50, restnet101, restnet152.""")
                    
                    st.subheader("""ResNet Results""")
                    st.write("""- ResNet18: 66%""")
                    st.write("""- ResNet50: 55%""")
                    
                    st.subheader("""Why use MobileNet V2""")
                    st.write("""- Efficient networks optimized for speed and memory, with residual blocks""")
                    st.write("""- Based on an inverted residual structure where the shortcut connections are between the thin bottleneck layers  opposite to traditional residual models.""")
                    st.write("""- Uses lightweight depthwise convolutions to filter features in the intermediate expansion layer.""")
                    
                    st.subheader("""MobileNet V2 Results""")
                    st.write("""- MobileNet V2: 68%""")

                    st.subheader("""Hybrid Model for Deep Learning""")
                    st.write("""- Encoded patient clinic and cancer attribute categorical data in the metadata.csv file by using one-hot-encoding approach. """)
                    st.write("""- Generate a new csv file containing the encoded tabular data.""")
                    st.write("""- Keep the img_id column and image_path column in the csv file so that the model knows where to load the image.""")
                    st.write("""- Defined a class named SkinImageTabularDataset, which knows how to access both tabular and image data.""")
                    st.write("""- Drafted a class named ImageTabularHybridModel, which extends Pytorch nn.Module and can combine both image data and tabular data for deep learning. User can specify “image-only”, “tabular-only”, or “combined” parameters.""")
                    
                    st.subheader("""Training Results from all Hybrid Models.""")
                    st.write("""- Hybrid ResNet18: 81%""")
                    st.caption("""The hybrid model reaches accuracy near 0.8152 > 0.70 of purely Resnet18 model.""")
                    st.write("""- Hybrid MobileNet_V2: 82%""")
                    st.caption("""The hybrid model reaches accuracy near 0.8239 > 0.7087 of purely MobileNet V2 model.""")
                    st.write("""- Hybrid GoogleNet: 80%""")
                    st.caption("""The hybrid model reaches accuracy near 0.8087 > 0.7021 of purely GoogleNet model.""")
                    st.write("""- Hybrid DenseNet121: 81%""")
                    st.caption("""The hybrid model reaches accuracy near 0.8196 > 0.7087 of purely DenseNet121 model.""")
                    
                    st.subheader("""Hybrid Models: Tabular Data with Original Columns vs Tabular Data with Reduced Columns""")
                    st.write("""- We have also chosen the most important tabular columns (features) that we have studied in the machine learning phase to reduce the number of tabular data columns.""")
                    st.write("""- We have experimented and compared the training results of Image-Tabular hybrid model using full tabular columns and reduced tabular columns.""")
                    
                    st.subheader("""Results of Training Hybrid Models (Full Columns vs Reduced Columns)""")
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\HybridTrainingModel.png')
                    st.image(image,caption='Figure 3: Showing the score difference between full and reduce columns')
                    st.write("""- Resnet18 seems to have slightly improvement when tabular data is made up of reduced columns.  Others seem to have no or less improvement.  It may make sense because deep learning can figure out what features are important and ignore what are less important after sufficient big number of epochs of training.  
                             Or all the original columns may all have some contributions to the classification.""")
                    
                    st.subheader("""Results of Training Hybrid Models (Image only vs Tabular only vs Combined)""")
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\HybridTrainingModel2.png')
                    st.image(image,caption='Figure 4: Showing the score difference between Image only vs Tabular only vs Combined')
                    st.write("""- Image-Tabular combined model does perform better than pure image model or pure tabular model in deep learning after 100 epochs of trainings. It is **1+1>2**""")
                    
                with st.expander('Phase 3',expanded=True):
                    
                    st.subheader("""Project Goal""")
                    st.write("""- The aim this project is to study the problem of classification of dermoscopic images of skin cancer, including lesions by using various of machine learning and deep learning techniques.""")
                    st.write("""- The hypothesis of this project is that combining skin lesion image data with more tabular descriptions or patient data may improve skin lesion classifications and skin cancer diagnosis. """)
                    
                    st.subheader("""Project Main Process""")
                    st.write("""overall techniques we used to predict skin cancer classification for this project""")
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\Projectroadmap.png')
                    st.image(image,caption='Figure 1: Shows the overall process for the project',width=750)
                    
                    st.subheader("""Project Methodology""")
                    st.write("""To achieve the goal of this project we have applied both machine learning and deep learning methodologies consisting of several phases or steps as follows:""")
                    st.write("""- Preprocess and analyze the metadata and construct various machine learning models to predict and evaluate the results of skin lesion classifications.""")
                    st.write("""- Use image processing techniques and experiment with various prebuilt deep learning models on skin image data to evaluate the results of skin lesion classifications.""")
                    st.write("""- Use prebuilt Convolutional Neural Networks (CNN) to inference missing values in columns of clinic tabular data. """)
                    st.write("""- Construct a customized hybrid model that can take both image and tabular data and uses a deep learning approach to do skin lesion classifications.""")
                    st.write("""- Evaluate the accuracy and performance of various models and find the best model for skin lesion classifications.""")
                    st.write("""- Construct a data web application using streamlit to overtake our Google Colab notebooks, for a more uniform/seamless experience when presenting our project and findings.""")
                    
                    st.subheader("""Machine Learning""")
                    st.write("""- Multiple statistical test techniques Correlation coefficient, Chi square test and Stepwise regression used to identify best predictor. """)
                    st.write("""- Final list of variables trained in ML: ‘region’,’itch’,’grew’,’changed’,’bleed’,’elevation’,’biopsed’,’age’,’smoke’,’drink’,’background_father’,’pesticide’,’skin_cancer_history’,’cancer_history’,’has_swage_system’ and ’fitspatrick’.""")
                    
                    st.subheader("""ML Accuracy""")
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\MLAcc.png')
                    st.image(image,caption='Figure 2: Shows Highest accuracy score for the ML models',width=750)
                    
                    st.subheader("""Deep Learning""")
                    st.write("""- Data Preprocessing""")
                    st.write("""- Prebuilt CNN Model Selection""")
                    st.write("""- Tabular Column Inferencing by Prebuilt CNN""")
                    st.write("""- Hybrid Models""")
                    
                    st.subheader("""Data Preprocessing""")
                    st.write("""- Six types of skin lesion images are programmatically copied into six image sub-folders to comply with the directory structure that Pytorch’s torchvision.datasets.ImageFolder favors.""")
                    st.write("""- To deal with imbalanced data: Apply weight sampling on training data, Apply augmentation techniques: resizing, random flipping, random rotation, cropping, and color jittering transformations on the image data when data is being loaded for training.""")
                    
                    st.subheader("""Prebuilt CNN Model Selection""")
                    st.write("""- Experiment with training the prebuilt CNN models, such as ResNet18, ResNet50, GoogleNet, DenseNet121, DenseNet161, DenseNet201, and MobileNet V2.""")
                    st.write("""- Input: Skin lesion image data Output: Skin lesion types""")
                    st.write("""- The best 4 prebuilt models, Resnet18, GoogleNet, DenseNet121, and MobileNet V2, are selected for further usages.""")
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\DLAcc.png')
                    st.image(image,caption='Figure 3: Shows Highest accuracy score for the DL models',width=750)
                    
                    st.subheader("""Tabular Column Inferencing by Prebuilt CNN""")
                    st.write("""- Consider score 0.7 and higher""")
                    st.write("""- ResNet18: 5 cols, MobileNet V2: 7 cols, GoogleNet: 8 cols, DenseNet121: 8 cols""")
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\DLcolumngraph.png')
                    st.image(image,caption='Figure 4: A depiction of CNN models on column inference',width=750)
                    
                    st.subheader("""Hybrid Models""")
                    st.write("""- Encoded patient clinic and cancer attribute categorical data in the metadata.csv file""")
                    st.write("""- Keep the img_id column in the csv file so that the model knows where to load the image.""")
                    st.write("""- Defined a class named SkinImageTabularDataset, which knows how to access both tabular and image data.""")
                    st.write("""- Constructed a class named ImageTabularHybridModel, which extends Pytorch nn.Module and can combine both image data and tabular data for deep learning. User can specify “image-only”, “tabular-only”, or “combined” parameters.""")
                    
                    st.subheader("""Image Tabular Hybrid Model""")
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\Hybridflowchart.png')
                    st.image(image,caption='Figure 5: A depiction of process to create a hybrid prediction model',width=750)
                    
                    st.subheader("""One-Hot Encoding vs. Ordinal Encoding""")
                    st.write("""- Apply One-hot Encoding and Ordinal Encoding on tabular columns.""")
                    st.write("""- One-hot Encoding has significantly better accuracy scores than the accuracy scores of Ordinal Encoding yielded by all the 4 hybrid models.""")
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\OnehotOrdinal.png')
                    st.image(image,caption='Figure 6: Socres between One-Hot and Ordinal encoding',width=750)
                    
                    st.subheader("""Full Tabular Columns vs. Reduced Tabular Columns""")
                    st.write("""- We assume some tabular columns may contribute more than others to the classifications, so we compare results of reduced columns with full columns, but there is no significant improvement.""")
                    st.write("""- Deep learning can figure out what features are important and ignore what are less important after a sufficiently big number of epochs of training.  So, the original clinic columns may all have some contributions to the classification.""")
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\ReducedVs.png')
                    st.image(image,caption='Figure 6: Socres between One-Hot and Ordinal encoding',width=750)
                    
                    st.subheader("""With “Biopsed” Column vs. Without “Biopsed” Column""")
                    st.write("""- Although we suspect the “biopsed” column may introduce bias to skin lesion classifications, the results do not show the missing “biopsed” column has significant influence on our hybrid validation accuracy scores.""")
                    st.write("""- A possible explanation is that other clinic tabular columns could make up for the missing influence of “biopsed” columns and still contribute significantly to the classifications during deep learning.""")
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\BioWith_out.png')
                    st.image(image,caption='Figure 7: Socres between with and without Biopsed column',width=750)
                    
                    st.subheader("""Image Data Only vs. Tabular Data Only vs. Image Tabular Data Combined""")
                    st.write("""Regardless of whether including or excluding certain clinic tabular columns, after training the 4 hybrid models, all results consistently show that models trained by image tabular combined data have higher validation accuracy scores than the scores of the models only trained by tabular data or only trained by image data.""")
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\TabularVs.png')
                    st.image(image,caption='Figure 7: Socres Image Data Only vs. Tabular Data Only vs. Image Tabular Data Combined',width=750)
                    
                    st.subheader("""Best Hybrid Models""")
                    st.write("""- We have applied up-sampling to balance underrepresented skin lesion types in the training dataset and kept recording the best validation accuracy score of each hybrid model in each training epoch.""")
                    st.write("""- The highest accuracy score is 0.846 of the hybrid models that utilize DenseNet121""")
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\BestHybrid.png')
                    st.image(image,caption='Figure 8: Scores for the best Hybrid model',width=750)
                    
                    st.subheader("""Confusion Matrices of the Best Hybrid Models""")
                    st.caption("""ACK: 0, BCC: 1, MEL: 2, NEV: 3, SCC 4, SEK: 5""")
                    image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\ConfusMatrfinal.png')
                    st.image(image,caption='Figure 8: Scores for the best Hybrid model',width=750)
                    
                    st.subheader("""Conclusion and Future Study""")
                    st.write("""- Our hybrid models that are capable of taking combined clinic tabular data and image data outperforms all other models we have experienced.""")
                    st.write("""- In a practical clinic situation, doctors not only examine skin images but also consider other formats of clinic data before a diagnosis is made.  Our hybrid models can simulate such clinic situations because they take both image data and clinic data in tabular format to make relatively good predictions for cancer diagnosis.""")
                    st.write("""- Besides skin cancer classifications, our concept of combining tabular data with image data in deep learning can also be extended to other fields where both tabular data and image data play important roles.""")
                    st.write("""- We can plan to build a more advanced model that not only can combine tabular data and image data but also be able to integrate with unstructured text data, where Natural Language Processing can be applied to extract other essential information, to further assist skin cancer diagnosis.""")
                    
                submitted = st.form_submit_button("Submit")
                
            with st.form("my_form2"):
                st.header("References and Related work")
                
                st.subheader("""Related Skin lession papers""")
                st.write("""“Melanoma Skin Lesions Classification using Deep Convolutional Neural Network with Transfer Learning”""")
                st.write("""- Utilizes HAM10000 Dataset""")
                st.write("""- Focuses on image processing and CNN""")
                st.write("""- Only classifies skin lesions into 2 categories (benign vs malignant)""")
                
                st.write("""“Deep Learning Based Integrated Classification and Image Retrieval System for Early Skin Cancer Detection”""")
                st.write("""- Utilizes a skin cancer image database""")
                st.write("""- Uses pre-trained CNN model to extract features of cancer images""")
                st.write("""- Applies image segmentation and feature fusion to form image features""")
                st.write("""- Uses machine learning models such as LR and SVC on the fused features to do classification.""")
                
                st.write("""“Comparison of the accuracy of human readers versus machine-learning algorithms for pigmented skin lesion classification: an open, web-based, international, diagnostic study”""")
                st.write("""- Findings: “State-of-the-art machine-learning classifiers outperformed human experts in the diagnosis of pigmented skin lesions”""")
                
                st.write("""“Skin Lesion Classification using Deep Learning and Image Processing”""")
                st.write("""- Utilizes HAM10000 Dataset""")
                st.write("""- Focuses on image processing and deep learning models""")
                st.write("""- Integrates a CNN model with another pre-trained CNN model to achieve better accuracy for classification""")
                
                st.subheader("""References""")
                st.write("""- Islam, M. K., Ali, M. S., Ali, M. M., Haque, M. F., Das, A. A., Hossain, M. M., Duranta, D. S., & Rahman, M. A. (2021). 
                         Skin Lesions Classification using Deep Convolutional Neural Network with Transfer Learning. 2021 1st International Conference on Artificial Intelligence and Data Analytics (CAIDA), 
                         Artificial Intelligence and Data Analytics (CAIDA), 
                         2021 1st International Conference On, 48–53. https://doi-org.proxy-bc.researchport.umd.edu/10.1109/CAIDA51941.2021.9425117""")
                
                st.write("""- Jibhakate, A., Parnerkar, P., Mondal, S., Bharambe, V., & Mantri, S. (2020). Skin Lesion Classification using Deep Learning and Image Processing. 
                         2020 3rd International Conference on Intelligent Sustainable Systems (ICISS), Intelligent Sustainable Systems (ICISS), 
                         2020 3rd International Conference On, 333–340. https://doi-org.proxy-bc.researchport.umd.edu/10.1109/ICISS49785.2020.9316092""")
                
                st.write("""- Layode, O., Alam, T., & Rahman, M. M. (2019). Deep Learning Based Integrated Classification and Image Retrieval System for Early Skin Cancer Detection. 
                         2019 IEEE Applied Imagery Pattern Recognition Workshop (AIPR), 
                         Applied Imagery Pattern Recognition Workshop (AIPR), 2019 IEEE, 1–7. https://doi-org.proxy-bc.researchport.umd.edu/10.1109/AIPR47015.2019.9174586""")
                
                st.write("""- Tschandl, P., Codella, N., Akay, B. N., Argenziano, G., Braun, R. P., Cabo, H., Gutman, D., Halpern, A., 
                         Helba, B., Hofmann-Wellenhof, R., Lallas, A., Lapins, J., Longo, C., Malvehy, J., Marchetti, M. A., Marghoob, A., Menzies, S., Oakley, A., Paoli, J., … Kittler, H. (2019). 
                         Comparison of the accuracy of human readers versus machine-learning algorithms for pigmented skin lesion classification: an open,
                         web-based, international, diagnostic study. Lancet Oncology, 7, 938.""")
                
                st.write("""- Brownlee, J. (2020, August 19). 4 types of classification tasks in machine learning. Machine Learning Mastery. 
                         Retrieved November 3, 2021, from https://machinelearningmastery.com/types-of-classification-in-machine-learning/.""")
                
                st.write("""- Makwana, K. (2021, June 2). Frequent category imputation (missing data imputation technique). Medium. 
                         Retrieved November 3, 2021, from https://medium.com/geekculture/frequent-category-imputation-missing-data-imputation-technique-4d7e2b33daf7.""")
                
                st.write("""- Ivan Vasilev, Daniel Slater, Gianmario Spacagna, Peter Roelants, & Valentino Zocca. (2019). Python Deep Learning : 
                         Exploring Deep Learning Techniques and Neural Network Architectures with PyTorch, Keras, and TensorFlow, 2nd Edition: Vol. Second edition. Packt Publishing""")
                
                st.write("""- Amer Abdulkader, 10509213 Canada Inc., & Sarmad Tanveer. (2019). PyTorch for Deep Learning and Computer Vision. Packt Publishing""")
                
                st.write("""- Torchvision.models. torchvision.models - Torchvision 0.11.0 documentation. (n.d.). Retrieved November 7, 2021, from https://pytorch.org/vision/stable/models.html""")
                
                st.write("""- Huang, G., Liu, Z., Maaten, L.V.D., Densely Connected Convolutional Networks, (2016), https://arxiv.org/abs/1608.06993""")
                
                st.write("""- Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., Rabinovich, A., Going deeper with convolutions, (2014), https://arxiv.org/abs/1409.4842#""")
                
                st.write("""- He, K., Zhang, X., Ren, S., Sun, J., Deep Residual Learning for Image Recognition, (2015), https://arxiv.org/abs/1512.03385""")
                
                st.write("""- Sandler,  M.,  Howard A., Zhu, M.,  Zhmoginov, A., Chen, L.C., MobileNetV2: Inverted Residuals and 
                         Linear Bottlenecks, The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 4510-4520 https://arxiv.org/abs/1801.04381""")
                
                submitted = st.form_submit_button("Submit")
                
                

    if choice == 'EDA and ML':
        st.title("Exploritory Data Analysis and Machine Learning")
        metadata = pd.read_csv(r"C:\Users\marcu\Downloads\metadata.csv")
        
        code = """number = st.slider('Choose a number',max_value=len(metadata))
st.write(metadata.head(number))"""
        st.code(code,language='python')
        st.subheader("Please use the slider to determine how many rows of data you would like to see.")
        number = st.slider('Choose a number',max_value=len(metadata))
        st.write(metadata.head(number))
        
    
        count_nan_in_df = metadata.isnull().sum()
        code = count_nan_in_df
        st.subheader('Here is the total amount of null values in the dataframe that we are working with')
        st.code(code,language='python')
        code = """bars = alt.Chart(metadata).mark_bar().encode(
            y=alt.X('count(diagnostic)',title='Total'),
            x=alt.Y('diagnostic',sort='-x',title='Skin Lession'),
            color='diagnostic'
        ).properties(title='Types of Skin Lession Classification',height=700,width=900)
        text = bars.mark_text(
            baseline='middle',
            dy = -6
        ).encode(
            text = 'count(diagnostic)'
        )
        st.altair_chart(bars+text,use_container_width=True)"""
        st.code(code,language='python')
        bars = alt.Chart(metadata).mark_bar().encode(
            y=alt.X('count(diagnostic)',title='Total'),
            x=alt.Y('diagnostic',sort='-x',title='Skin Lession'),
            color='diagnostic'
        ).properties(title='Types of Skin Lession Classification',height=700,width=900)
        text = bars.mark_text(
            baseline='middle',
            dy = -6
        ).encode(
            text = 'count(diagnostic)'
        )
        st.altair_chart(bars+text,use_container_width=True)
        st.subheader('As we can see above in the bar graph the skin classification BCC is the highest reported classification and MEL is the lowest reported classification.')
        
        code = """bars = alt.Chart(metadata).mark_bar().encode(
            y=alt.X('count(biopsed)',title='Total'),
            x=alt.Y('biopsed',sort='-x'),
            color='biopsed'
        ).properties(title='Biopised breakdown',height=700,width=900)
        
        text = bars.mark_text(
            baseline='middle',
            dy = -6
        ).encode(
            text = 'count(biopsed)'
        )
        st.altair_chart(bars+text,use_container_width=True)"""
        st.code(code,language='python')
        bars = alt.Chart(metadata).mark_bar().encode(
            y=alt.X('count(biopsed)',title='Total'),
            x=alt.Y('biopsed',sort='-x'),
            color='biopsed'
        ).properties(title='Biopised breakdown',height=700,width=900)
        text = bars.mark_text(
            baseline='middle',
            dy = -6
        ).encode(
            text = 'count(biopsed)'
        )
        st.altair_chart(bars+text,use_container_width=True)
        st.subheader('As we can see above in the bar graph the total number of Biosies that were performed was **1342** and the one that were not was **956**.')
        
        
        metadata = metadata.drop(columns=['patient_id','img_id','lesion_id'])
        
        code = """bars = alt.Chart(metadata).mark_bar().encode(
            y=alt.X('count(elevation)',title='Total'),
            x=alt.Y('elevation',sort='-x'),
            color='elevation'
        ).properties(title='elevation breakdown',height=700,width=900)
        
        text = bars.mark_text(
            baseline='middle',
            dy = -6
        ).encode(
            text = 'count(elevation)'
        )"""
        st.code(code,language='python')
        bars = alt.Chart(metadata).mark_bar().encode(
            y=alt.X('count(elevation)',title='Total'),
            x=alt.Y('elevation',sort='-x'),
            color='elevation'
        ).properties(title='elevation breakdown',height=700,width=900)
        
        text = bars.mark_text(
            baseline='middle',
            dy = -6
        ).encode(
            text = 'count(elevation)'
        )
        
        metadata=metadata.replace('UNK',np.nan,regex=True)
        metadata['background_father'] = metadata['background_father'].str.replace('BRASIL','BRAZIL')
        
        bars1 = alt.Chart(metadata).mark_bar().encode(
            y=alt.X('count(background_father)',title='Total',sort='x'),
            x=alt.Y('background_father'),
            color='background_father'
        ).properties(title='background_father breakdown',height=700,width=900)
        st.altair_chart(bars+text,use_container_width=True)
        
        
        code = """bars1 = alt.Chart(metadata).mark_bar().encode(
            y=alt.X('count(background_father)',title='Total',sort='x'),
            x=alt.Y('background_father'),
            color='background_father'
        ).properties(title='background_father breakdown',height=700,width=900)
        
        st.altair_chart(bars+text,use_container_width=True)
        st.altair_chart(bars1,use_container_width=True)"""
        st.code(code,language='python')
        st.altair_chart(bars1,use_container_width=True)
        st.subheader('We identified there is value UNK for unknown values for variables:Background_Father,background_mother,Elevealtion,hurt,itch,grew,changed,bleed. Then we replace those values with nan and imputed with mode function.')
        
        st.title('Preparing data for Machine Learning')
        st.subheader("""Preprocessing and Data cleaning""")
        st.subheader("""Identifing the categorical variables from the Tabular data""")
        
        cat = metadata.select_dtypes(include='O').keys()
        code = """cat = metadata.select_dtypes(include='O').keys()
st.write(cat)"""
        st.code(code,language='python')
        st.write(cat)
        
        st.subheader('Most frequent / mode Imputation')
        st.subheader('Replace the missing values for data categorical variable using the most frequent values.Using mode function. https://medium.com/geekculture/frequent-category-imputation-missing-data-imputation-technique-4d7e2b33daf7')
        
        cat_columns = metadata[['smoke', 'drink', 'background_father', 'background_mother', 'pesticide',
       'gender', 'skin_cancer_history', 'cancer_history', 'has_piped_water',
       'has_sewage_system', 'region', 'itch', 'grew', 'hurt',
       'changed', 'bleed', 'elevation','biopsed']]
        
        for column in cat_columns.columns:
            cat_columns[column].fillna(cat_columns[column].mode()[0], inplace=True)

        code = """cat_columns = metadata[['smoke', 'drink', 'background_father', 'background_mother', 'pesticide',
       'gender', 'skin_cancer_history', 'cancer_history', 'has_piped_water',
       'has_sewage_system', 'region', 'itch', 'grew', 'hurt',
       'changed', 'bleed', 'elevation','biopsed']]
        
    for column in cat_columns.columns:
        cat_columns[column].fillna(cat_columns[column].mode()[0], inplace=True)"""
        st.code(code,language='python')
        st.write(cat_columns['background_father'].value_counts())
        
        st.subheader('Identify numeric features and impute missing values with mean')

        metaData_le= cat_columns
        col=metadata[['age','fitspatrick','diameter_1','diameter_2']]
        from sklearn.impute import SimpleImputer  
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer = imputer.fit(col)
        metaData_le[['age','fitspatrick','diameter_1','diameter_2']]= imputer.transform(col)
        metaData_le=metaData_le.join(metadata['diagnostic'])
        code = """        metaData_le= cat_columns
        
        col=metadata[['age','fitspatrick','diameter_1','diameter_2']]
        from sklearn.impute import SimpleImputer  
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer = imputer.fit(col)
        metaData_le[['age','fitspatrick','diameter_1','diameter_2']]= imputer.transform(col)
        
        metaData_le=metaData_le.join(metadata['diagnostic'])
        st.write(metaData_le.head())"""
        st.code(code,language='python')
        st.write(metaData_le.head())
        
        st.subheader('Appending the numerical variables. combine Categorical and numeric features into one dataframe')
        
        metaData_le.info()
        code = '''metaData_le.info()'''
        st.code(code,language='python')
        image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\info().png')
        st.image(image)
        
        st.subheader('Ordinal encoding')
        code = '''      from sklearn.preprocessing import OrdinalEncoder
        encoder = OrdinalEncoder()
        data_enc = encoder.fit_transform(metaData_le)
        st.write(data_enc)'''
        st.code(code,language='python')
        from sklearn.preprocessing import OrdinalEncoder
        encoder = OrdinalEncoder()
        data_enc = encoder.fit_transform(metaData_le)
        st.write(data_enc)
        
        y2=metaData_le[['diagnostic']]
        x2=metaData_le.drop(columns=['diagnostic'])
        code = '''y2=metaData_le[['diagnostic']]
x2=metaData_le.drop(columns=['diagnostic'])'''
        st.code(code,language='python')
        st.subheader("""Data prep for chi square test and correlation split data to "X" and "Y" for chi square test and step wise regression to identify the best predictor variables. Creating a dataframe from the encoded numpy array "x" an "y""")
        
        x_enc = pd.DataFrame(data=data_enc[:, :-1], columns=x2.columns) # for all but last column
        y_enc = pd.DataFrame(data=data_enc[:, -1], columns=y2.columns)# for last column
        code = '''x_enc = pd.DataFrame(data=data_enc[:, :-1], columns=x2.columns) # for all but last column
y_enc = pd.DataFrame(data=data_enc[:, -1], columns=y2.columns)# for last column'''
        st.code(code,language='python')
        st.write(x_enc.shape)
        
        from dython.nominal import associations
        num_cols = len(metaData_le.columns)
        associations(metaData_le, nom_nom_assoc='theil', figsize=(num_cols, num_cols))
        code = '''from dython.nominal import associations
num_cols = len(metaData_le.columns)
associations(metaData_le, nom_nom_assoc='theil', figsize=(num_cols, num_cols))'''
        st.code(code,language='python')
        image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\corrmatrix.png')
        st.image(image)
        
        from sklearn.feature_selection import SelectKBest, chi2 
        sf = SelectKBest(chi2, k='all')
        sf_fit = sf.fit(x_enc, y_enc)
        for i in range(len(sf_fit.scores_)):
            print(' %s: %f' % (x_enc.columns[i], sf_fit.scores_[i]))
        code = '''from sklearn.feature_selection import SelectKBest, chi2 
sf = SelectKBest(chi2, k='all')
sf_fit = sf.fit(x_enc, y_enc)
for i in range(len(sf_fit.scores_)):
    print(' %s: %f' % (x_enc.columns[i], sf_fit.scores_[i]))'''
        st.code(code,language='python')
        st.json({
            'smoke': '129.364969',
            'drink': '124.056784',
            'background_father': '329.449437',
            'background_mother': '428.725601',
            'pesticide': '217.131971',
            'gender': '254.365572',
            'skin_cancer_history': '215.465226',
            'cancer_history': '88.942515',
            'has_piped_water': '58.949159',
            'has_sewage_system': '82.074732',
            'region': '105.452255',
            'itch': '232.712008',
            'grew': '140.666595',
            'hurt': '295.562660',
            'changed': '224.839432',
            'bleed': '418.616763',
            'elevation': '234.720572',
            'biopsed': '623.995996',
            'age': '3448.582874',
            'fitspatrick': '71.561837',
            'diameter_1': '67.381549',
            'diameter_2': '73.623881',
        })
        
        st.subheader("""Stepwise regression is a way to build a model by adding or removing predictor variables, usually via a series of F-tests or T-tests. The variables to be added or removed are chosen based on the test statistics of the estimated coefficients. Start the test with no predictor variables (the “Forward” method), adding one at a time as the regression model progresses. If you have a large set of predictor variables, use this method""")
        def get_stats():
            results = sm.OLS(y_enc, x_enc).fit()
            print(results.summary())
        get_stats()
        forward_regression(X=x_enc,Y=y_enc,threshold_in=0.05)
        code = '''def get_stats():
    results = sm.OLS(y_enc, x_enc).fit()
    print(results.summary())
get_stats()
forward_regression(X=x_enc,Y=y_enc,threshold_in=0.05)'''
        st.code(code,language='python')
        code3 = ("""                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:             diagnostic   R-squared (uncentered):                   0.585
Model:                            OLS   Adj. R-squared (uncentered):              0.581
Method:                 Least Squares   F-statistic:                              146.0
Date:                Sat, 04 Dec 2021   Prob (F-statistic):                        0.00
Time:                        01:15:38   Log-Likelihood:                         -4167.0
No. Observations:                2298   AIC:                                      8378.
Df Residuals:                    2276   BIC:                                      8504.
Df Model:                          22                                                  
Covariance Type:            nonrobust                                                  
=======================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------
smoke                   0.1419      0.117      1.208      0.227      -0.088       0.372
drink                  -0.1165      0.099     -1.182      0.237      -0.310       0.077
background_father       0.0452      0.026      1.721      0.085      -0.006       0.097
background_mother       0.0193      0.027      0.711      0.477      -0.034       0.073
pesticide              -0.1455      0.085     -1.709      0.088      -0.313       0.021
gender                  0.0775      0.087      0.886      0.376      -0.094       0.249
skin_cancer_history     0.1472      0.076      1.944      0.052      -0.001       0.296
cancer_history          0.2322      0.072      3.203      0.001       0.090       0.374
has_piped_water         0.2319      0.121      1.922      0.055      -0.005       0.468
has_sewage_system       0.0763      0.120      0.634      0.526      -0.159       0.312
region                 -0.0392      0.010     -4.059      0.000      -0.058      -0.020
itch                   -1.0375      0.073    -14.230      0.000      -1.181      -0.895
grew                    0.4850      0.071      6.836      0.000       0.346       0.624
hurt                   -0.0734      0.101     -0.730      0.466      -0.271       0.124
changed                -0.0764      0.119     -0.642      0.521      -0.310       0.157
bleed                  -0.2902      0.090     -3.208      0.001      -0.468      -0.113
elevation               0.9685      0.071     13.651      0.000       0.829       1.108
biopsed                -0.0030      0.092     -0.032      0.974      -0.182       0.177
age                     0.0100      0.002      5.170      0.000       0.006       0.014
fitspatrick             0.0786      0.033      2.361      0.018       0.013       0.144
diameter_1              0.0087      0.012      0.715      0.474      -0.015       0.033
diameter_2              0.0171      0.015      1.128      0.259      -0.013       0.047
==============================================================================
Omnibus:                      172.268   Durbin-Watson:                   1.987
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              211.334
Skew:                           0.739   Prob(JB):                     1.29e-46
Kurtosis:                       2.853   Cond. No.                         293.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
Included variables  itch                           with p-value 0.0000
Included variables  elevation                      with p-value 0.0000
Included variables  biopsed                        with p-value 0.0000
Included variables  grew                           with p-value 0.0000
Included variables  region                         with p-value 0.0000
Included variables  bleed                          with p-value 0.0016
Included variables  pesticide                      with p-value 0.0029
Included variables  age                            with p-value 0.0118
Included variables  has_piped_water                with p-value 0.0302
['itch',
 'elevation',
 'biopsed',
 'grew',
 'region',
 'bleed',
 'pesticide',
 'age',
 'has_piped_water']""")
    
        st.code(code3,language='python')
    
        st.subheader("""Start the test with all available predictor variables (the “Backward: method), deleting one variable at a time as the regression model progresses. Use this method if you have a modest number of predictor variables and you want to eliminate a few.""")
    
        st.write(metaData_le.columns)
        code = '''def backward_regression(X, Y,threshold_out):
    included=list(X.columns)
        while True:
            changed=False
            model = sm.OLS(Y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            # use all coefs except intercept
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max() # null if pvalues is empty
        
            if worst_pval > threshold_out:
            #print(worst_pval)
                changed=True
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                print('Removed variable {:30} with p-value {:.6}'.format(worst_feature, format(worst_pval, '.4f')))
            if not changed:
                break
        return included
backward_regression(X=x_enc,Y=y_enc,threshold_out=.05)'''
        st.code(code,language='python')
        backward_regression(X=x_enc,Y=y_enc,threshold_out=.05)
    
        code4 = """Removed variable background_father              with p-value 0.9771
Removed variable diameter_1                     with p-value 0.9771
Removed variable changed                        with p-value 0.7984
Removed variable hurt                           with p-value 0.7197
Removed variable has_sewage_system              with p-value 0.6824
Removed variable gender                         with p-value 0.6869
Removed variable fitspatrick                    with p-value 0.4542
Removed variable background_mother              with p-value 0.3825
Removed variable skin_cancer_history            with p-value 0.3157
Removed variable cancer_history                 with p-value 0.2883
Removed variable smoke                          with p-value 0.1795
Removed variable drink                          with p-value 0.1236
Removed variable diameter_2                     with p-value 0.0823
['pesticide',
 'has_piped_water',
 'region',
 'itch',
 'grew',
 'bleed',
 'elevation',
 'biopsed',
 'age']"""
 
        st.code(code4,language='python')
        st.header('Data preprocessing for Machine Learning')
        st.subheader("The best predictors we will take for model development are: age, biopsed, grew, region, pesticide, skin_cancer_history, cancer_history, background_father, drink, smoke, itch, bleed, changed, has_sewage_system, elevation")
    
        #dropping all the unwanted columns from the encoded dataframe
        #dropping background_mother, gender, has_piped_water, hurt, fitspatrick, diameter1, diameter_2
        data_enc=np.delete(data_enc,[3,5,8,13,20,21],axis=1)
        feature_names=['smoke', 'drink', 'background_father', 'pesticide',
        'skin_cancer_history', 'cancer_history',
       'has_sewage_system', 'region', 'itch', 'grew', 'changed',
       'bleed', 'elevation', 'biopsed', 'age','fitspatrick','diagnostic']
        code = '''#dropping all the unwanted columns from the encoded dataframe
#dropping background_mother, gender, has_piped_water, hurt, fitspatrick, diameter1, diameter_2
data_enc=np.delete(data_enc,[3,5,8,13,20,21],axis=1)
feature_names=['smoke', 'drink', 'background_father', 'pesticide',
        'skin_cancer_history', 'cancer_history',
       'has_sewage_system', 'region', 'itch', 'grew', 'changed',
       'bleed', 'elevation', 'biopsed', 'age','fitspatrick','diagnostic']'''
        st.code(code,language='python')
        st.write(data_enc)
    
        X = data_enc[:, :-1]
        Y = data_enc[:, -1]
        Target_Names=['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
        #  Normalize data
        from sklearn.preprocessing import StandardScaler
        # define standard scaler
        scaler = StandardScaler()
        # transform data
        scaled = scaler.fit_transform(X)
        code = '''X = data_enc[:, :-1]
Y = data_enc[:, -1]
Target_Names=['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
#  Normalize data
from sklearn.preprocessing import StandardScaler
# define standard scaler
scaler = StandardScaler()
# transform data
scaled = scaler.fit_transform(X)'''
        st.code(code,language='python')
        st.write(scaled)
    
        st.subheader("Train and test differnet classification algorithms on the Tabular data")
    
        random_seed=5
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=random_seed)
        sgd_clf = SGDClassifier(random_state=random_seed)
        sgd_clf.fit(X_train, Y_train)
        sgd_pred=sgd_clf.predict(X_test)
        sgd_score = accuracy_score(Y_test, sgd_pred)
        print("Test score: ", sgd_score)
    
        Y_pred_sgd = sgd_clf.predict(X_test)
        y_unique = np.unique(Y_test)
        #cm = confusion_matrix(Y_test, Y_pred_sgd)
        #print(cm)
    
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        # Create the confusion matrix
        conf_mat = confusion_matrix(Y_test, Y_pred_sgd, normalize="true")
        # Plot the confusion matrix
        plt.subplots(figsize=(10,5))
        sns.heatmap(conf_mat.T, annot=True, fmt=".0%", cmap="cividis", xticklabels=Target_Names, yticklabels=Target_Names,linewidths=1)
        plt.xlabel("True label")
        plt.ylabel("Predicted label")
        print("SGDClassifier Score: ", sgd_score*100)

        code = '''random_seed=5
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=random_seed)
sgd_clf = SGDClassifier(random_state=random_seed)
sgd_clf.fit(X_train, Y_train)
sgd_pred=sgd_clf.predict(X_test)
sgd_score = accuracy_score(Y_test, sgd_pred)
print("Test score: ", sgd_score)
Y_pred_sgd = sgd_clf.predict(X_test)
y_unique = np.unique(Y_test)
cm = confusion_matrix(Y_test, Y_pred_sgd)
print(cm)

[[117  11   0   0   0   0]
 [ 35 147   0   0   2   0]
 [  6   4   0   0   0   0]
 [ 16   4   0  25   0   0]
 [ 20  28   0   0   2   0]
 [ 38   2   0   3   0   0]]'''
        st.code(code,language='python')
        code = '''from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Create the confusion matrix
conf_mat = confusion_matrix(Y_test, Y_pred_sgd, normalize="true")
# Plot the confusion matrix
plt.subplots(figsize=(10,5))
sns.heatmap(conf_mat.T, annot=True, fmt=".0%", cmap="cividis", xticklabels=Target_Names, yticklabels=Target_Names,linewidths=1)
plt.xlabel("True label")
plt.ylabel("Predicted label")
print("SGDClassifier Score: ", sgd_score*100)'''
        st.code(code,language='python')
        image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\SGDC.png')
        st.image(image)
    
        cr = classification_report(Y_test, Y_pred_sgd)
        print(cr)
        code = '''cr = classification_report(Y_test, Y_pred_sgd)
print(cr)'''
        st.code(code,language='python')
        code5 = """              precision    recall  f1-score   support

         0.0       0.50      0.91      0.65       128
         1.0       0.75      0.80      0.77       184
         2.0       0.00      0.00      0.00        10
         3.0       0.89      0.56      0.68        45
         4.0       0.50      0.04      0.07        50
         5.0       0.00      0.00      0.00        43

    accuracy                           0.63       460
   macro avg       0.44      0.38      0.36       460
weighted avg       0.58      0.63      0.57       460"""
        st.code(code5,language='python')
    
        ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=random_seed))
        ovo_clf.fit(X_train, Y_train)
        ovo_pred=ovo_clf.predict(X_test)
        OVO_score = accuracy_score(Y_test, ovo_pred)
        print("Test score: ", OVO_score)

        code = '''ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=random_seed))
ovo_clf.fit(X_train, Y_train)
ovo_pred=ovo_clf.predict(X_test)
OVO_score = accuracy_score(Y_test, ovo_pred)
print("Test score: ", OVO_score)'''
        st.code(code,language='python')
        st.write('Test score:  0.6543478260869565')
        
        code = '''cr = classification_report(Y_test, ovo_pred)
print(cr)
precision    recall  f1-score   support

         0.0       0.54      0.90      0.67       128
         1.0       0.74      0.85      0.79       184
         2.0       0.00      0.00      0.00        10
         3.0       0.85      0.64      0.73        45
         4.0       0.00      0.00      0.00        50
         5.0       0.00      0.00      0.00        43

    accuracy                           0.65       460
   macro avg       0.36      0.40      0.37       460
weighted avg       0.53      0.65      0.58       460'''
        st.code(code,language='python')
        cr = classification_report(Y_test, ovo_pred)
        print(cr)
    
        #   Create the confusion matrix
        conf_mat = confusion_matrix(Y_test, ovo_pred, normalize="true")
        # Plot the confusion matrix
        plt.subplots(figsize=(10,5))
        sns.heatmap(conf_mat.T, annot=True, fmt=".0%", cmap="cividis", xticklabels=Target_Names, yticklabels=Target_Names,linewidths=1)
        plt.xlabel("True label")
        plt.ylabel("Predicted label")
        plt.title('OVO_confusion_Matrix')
        print('OneVsOneClassifier Score:',OVO_score*100)

        code = '''#   Create the confusion matrix
conf_mat = confusion_matrix(Y_test, ovo_pred, normalize="true")
# Plot the confusion matrix
plt.subplots(figsize=(10,5))
sns.heatmap(conf_mat.T, annot=True, fmt=".0%", cmap="cividis", xticklabels=Target_Names, yticklabels=Target_Names,linewidths=1)
plt.xlabel("True label")
plt.ylabel("Predicted label")
plt.title('OVO_confusion_Matrix')
print('OneVsOneClassifier Score:',OVO_score*100)'''
        st.code(code,language='python')
        image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\onevsone.png')
        st.image(image)
    
        #split dataset into train and test
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=random_seed)
        rf_clf = RandomForestClassifier( criterion='entropy', max_depth=50)
        rf_clf.fit(X_train,Y_train)
        Y_pred_rf=rf_clf.predict_proba(X_test)
        rf_testScore=rf_clf.score(X_test,Y_test)
        print('rf_testscore is :',rf_testScore)
        code = '''#split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=random_seed)
rf_clf = RandomForestClassifier( criterion='entropy', max_depth=50)
rf_clf.fit(X_train,Y_train)
Y_pred_rf=rf_clf.predict_proba(X_test)
rf_testScore=rf_clf.score(X_test,Y_test)
print('rf_testscore is :',rf_testScore)'''
        st.code(code,language='python')
        st.write('rf_testscore is : 0.8052173913043478')
        
        Y_pred_rf = np.argmax(Y_pred_rf,axis=1)
        # Y_test= np.argmax(Y_test.reshape(1,-1),axis=0)
        # Y_pred_rf = np.argmax(Y_pred_rf,axis=1)
        # Y_test= np.argmax(Y_test,axis=1)
        # Create the confusion matrix
        conf_mat = confusion_matrix(Y_test, Y_pred_rf, normalize="true")
        # Plot the confusion matrix
        plt.subplots(figsize=(10,5))
        sns.heatmap(conf_mat.T, annot=True, fmt=".0%", cmap="cividis", xticklabels=Target_Names, yticklabels=Target_Names,linewidths=1)
        plt.xlabel("True label")
        plt.ylabel("Predicted label")
        plt.title('Random Forest Confusion Matrix')
        print('Random Forest*_testscore is :',(rf_testScore*100))
        code = '''Y_pred_rf = np.argmax(Y_pred_rf,axis=1)
# Y_test= np.argmax(Y_test.reshape(1,-1),axis=0)
# Y_pred_rf = np.argmax(Y_pred_rf,axis=1)
# Y_test= np.argmax(Y_test,axis=1)
# Create the confusion matrix
conf_mat = confusion_matrix(Y_test, Y_pred_rf, normalize="true")
# Plot the confusion matrix
plt.subplots(figsize=(10,5))
sns.heatmap(conf_mat.T, annot=True, fmt=".0%", cmap="cividis", xticklabels=Target_Names, yticklabels=Target_Names,linewidths=1)
plt.xlabel("True label")
plt.ylabel("Predicted label")
plt.title('Random Forest Confusion Matrix')
print('Random Forest*_testscore is :',(rf_testScore*100))'''
        st.code(code,language='python')    
        image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\Random.png')
        st.image(image)
    
        st.subheader('Identify importand features using Random Forest Feature importance')
    
        feature_names= pd.DataFrame(data=X, columns=feature_names[:16])
        feat_importances = pd.Series(rf_clf.feature_importances_, index=feature_names.columns)
        feat_importances.nlargest(24).plot(kind='barh',title='Random Forest Feature Importance',color='purple',figsize=(10, 8))
        
        code = '''feature_names= pd.DataFrame(data=X, columns=feature_names[:16])
feat_importances = pd.Series(rf_clf.feature_importances_, index=feature_names.columns)
feat_importances.nlargest(24).plot(kind='barh',title='Random Forest Feature Importance',color='purple',figsize=(10, 8))'''
        st.code(code,language='python')
        image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\RandomImportant.png')
        st.image(image)
    
        #split dataset into train and test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
        # decision tree classification problem
        from sklearn.tree import DecisionTreeClassifier
        from matplotlib import pyplot
        # define the model
        dt_clf = DecisionTreeClassifier(random_state=random_seed,max_depth=50,criterion='entropy')
        # fit the model
        dt_clf.fit(X_train, Y_train)
        pred_dt=dt_clf.predict(X_test)
        dt_Testscore=accuracy_score(Y_test,pred_dt)
        print('DT test score',dt_Testscore)
        
        code = '''#split dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
# decision tree classification problem
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
# define the model 
dt_clf = DecisionTreeClassifier(random_state=random_seed,max_depth=50,criterion='entropy')
# fit the model
dt_clf.fit(X_train, Y_train)
pred_dt=dt_clf.predict(X_test)
dt_Testscore=accuracy_score(Y_test,pred_dt)
print('DT test score',dt_Testscore)'''

        st.code(code,language='python')
        
        st.write('DT test score 0.7356521739130435')
        
        # Create the confusion matrix
        conf_mat = confusion_matrix(Y_test, pred_dt, normalize="true")
        # Plot the confusion matrix
        plt.subplots(figsize=(10,5))
        sns.heatmap(conf_mat.T, annot=True, fmt=".0%", cmap="cividis", xticklabels=Target_Names, yticklabels=Target_Names,linewidths=1)
        plt.xlabel("True label")
        plt.ylabel("Predicted label")
        plt.title('Random Forest Confusion Matrix')
        print('Decision Tree test score',dt_Testscore*100)
        
        code = '''# Create the confusion matrix
conf_mat = confusion_matrix(Y_test, pred_dt, normalize="true")
# Plot the confusion matrix
plt.subplots(figsize=(10,5))
sns.heatmap(conf_mat.T, annot=True, fmt=".0%", cmap="cividis", xticklabels=Target_Names, yticklabels=Target_Names,linewidths=1)
plt.xlabel("True label")
plt.ylabel("Predicted label")
plt.title('Random Forest Confusion Matrix')
print('Decision Tree test score',dt_Testscore*100)'''

        st.code(code,language='python')
        
        image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\180603.png')
        st.image(image)
    
        # get important features for decision tree clasifier
        importance = dt_clf.feature_importances_
        # summarize feature importance
        feature_importances = pd.Series(dt_clf.feature_importances_ , index=feature_names.columns)
        feature_importances.nlargest(20).plot(kind='barh',title='Decision Tree Feature Importance',color='purple',figsize=(10, 8))
        
        code = '''# get important features for decision tree clasifier
importance = dt_clf.feature_importances_
# summarize feature importance
feature_importances = pd.Series(dt_clf.feature_importances_ , index=feature_names.columns)
feature_importances.nlargest(20).plot(kind='barh',title='Decision Tree Feature Importance',color='purple',figsize=(10, 8))'''

        st.code(code,language='python')
    
        image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\TreeImportance.png')
        st.image(image)
    
        feature_cols=['age', 'biopsed', 'grew', 'region', 'pesticide', 'skin_cancer_history', 'cancer_history', 'background_father', 'drink', 'smoke', 'itch', 'bleed', 'has_sewage_system', 'elevation', 'fitspatrick','changed']
    
        #split dataset into train and test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=random_seed)
    
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import make_pipeline
        # Create a pipeline
        nb_clf = make_pipeline(MultinomialNB(alpha=1))
    
        # Fit the model with training set
        nb_clf.fit(X_train, Y_train)
        #Predict labels for the test set
        labels = nb_clf.predict(X_test)
        #Import scikit-learn metrics module for accuracy calculation
        from sklearn import metrics
        NB_Score=metrics.accuracy_score(Y_test, labels)
        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:",metrics.accuracy_score(Y_test, labels))
        
        st.subheader("""After fitting the model with the training set and predicting the labels for the test set,then plot the confusion matrix to evaluate the model result and check the percentage of True Vs False predictions.""")
        
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        # Create the confusion matrix
        conf_mat = confusion_matrix(Y_test, labels, normalize="true")
        # Plot the confusion matrix
        plt.subplots(figsize=(10,5))
        sns.heatmap(conf_mat.T, annot=True, fmt=".0%", cmap="cividis", xticklabels=Target_Names, yticklabels=Target_Names,linewidths=1)
        plt.xlabel("True label")
        plt.ylabel("Predicted label")
        plt.title('MultinomialNB Confusion Matrix')
        print('Test Accuracy score:',NB_Score*100)
        
        code = '''from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Create the confusion matrix
conf_mat = confusion_matrix(Y_test, labels, normalize="true")
# Plot the confusion matrix
plt.subplots(figsize=(10,5))
sns.heatmap(conf_mat.T, annot=True, fmt=".0%", cmap="cividis", xticklabels=Target_Names, yticklabels=Target_Names,linewidths=1)
plt.xlabel("True label")
plt.ylabel("Predicted label")
plt.title('MultinomialNB Confusion Matrix')
print('Test Accuracy score:',NB_Score*100)'''
        
        st.code(code,language='python')
        
        image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\Multino.png')
        st.image(image)
        
        st.subheader('Multinomial Logistic Regression')
        
        lm = LogisticRegression(multi_class='multinomial', solver='newton-cg', random_state=random_seed)
        lm.fit(X_train, Y_train)
        Y_pred=lm.predict(X_test)
        
        score=lm.score(X_test, Y_test)
        print('Test score is:',score)
        score=lm.score(X_train,Y_train)
        print('Train score is:',score)
        
        cr = classification_report(Y_test, Y_pred)
        print(cr)
        
        code = '''lm = LogisticRegression(multi_class='multinomial', solver='newton-cg', random_state=random_seed)
lm.fit(X_train, Y_train)
Y_pred=lm.predict(X_test)

score=lm.score(X_test, Y_test)
print('Test score is:',score)
score=lm.score(X_train,Y_train)
print('Train score is:',score)

cr = classification_report(Y_test, Y_pred)
print(cr)'''
        
        st.code(code,language='python')
        
        code = """              precision    recall  f1-score   support

         0.0       0.77      0.77      0.77       128
         1.0       0.69      0.96      0.80       184
         2.0       0.80      0.40      0.53        10
         3.0       0.78      0.69      0.73        45
         4.0       0.00      0.00      0.00        50
         5.0       0.70      0.49      0.58        43

    accuracy                           0.72       460
   macro avg       0.62      0.55      0.57       460
weighted avg       0.65      0.72      0.67       460"""

        st.code(code,language='python')
        
        conf_mat = confusion_matrix(Y_test, Y_pred, normalize="true")
        # Plot the confusion matrix
        plt.subplots(figsize=(10,5))
        sns.heatmap(conf_mat.T, annot=True, fmt=".0%", cmap="cividis", xticklabels=Target_Names, yticklabels=Target_Names,linewidths=1)
        plt.xlabel("True label")
        plt.ylabel("Predicted label")
        
        code = '''conf_mat = confusion_matrix(Y_test, Y_pred, normalize="true")
# Plot the confusion matrix
plt.subplots(figsize=(10,5))
sns.heatmap(conf_mat.T, annot=True, fmt=".0%", cmap="cividis", xticklabels=Target_Names, yticklabels=Target_Names,linewidths=1)
plt.xlabel("True label")
plt.ylabel("Predicted label")'''
        
        st.code(code,language='python')

        image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\Screenshot 2021-12-06 185657.png')
        st.image(image)
            
    if choice == 'Deep Learning':
        st.title("Deep Learning")
        
        st.subheader("EDA")
        metadata = pd.read_csv(r"C:\Users\marcu\Downloads\metadata.csv")
        
        code = '''metadata.head(10)'''
        st.code(code,language='python')
        st.write(metadata.head(10))
        
        count_nan_in_df = metadata.isnull().sum()
        print (count_nan_in_df)
        code = '''count_nan_in_df = metadata.isnull().sum()
print (count_nan_in_df)'''
        st.code(code,language='python')
        st.write(count_nan_in_df)
        
        st.subheader("List all image paths")
        code = '''skin_image_paths = glob(os.path.join(raw_data_path, '*.png')) #glob patterns specify sets of filenames with wildcard character.
skin_image_paths'''
        st.code(code,language='python')
        code = '''['/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_456_887_499.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_503_950_552.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_948_1801_721.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_162_252_773.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_122_185_957.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_691_1311_686.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_788_1503_541.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_212_327_532.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_432_851_560.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_442_863_795.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_53_82_657.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_101_1041_651.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_699_1320_901.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_31_42_680.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_822_1550_826.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_82_125_525.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_817_1537_643.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_660_1249_373.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_378_769_36.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_270_416_398.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_366_752_782.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_654_1244_16.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_74_112_729.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_774_1462_821.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_394_794_235.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_14_22_850.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_82_125_365.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_467_904_535.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_237_363_344.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_272_420_25.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_379_1531_17.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_126_191_304.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_923_1753_706.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_788_1503_168.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_535_1010_303.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_517_971_88.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_338_707_774.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_251_382_576.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_392_792_712.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_422_837_617.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_901_1716_497.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_877_1667_118.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_480_921_1.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_478_916_806.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_561_1068_557.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_211_318_942.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_192_295_164.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_185_285_862.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_460_894_242.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_876_1664_579.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_797_1511_805.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_876_1665_386.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_68_105_871.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_876_1665_569.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_523_988_900.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_827_1600_1000.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_830_1565_911.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_83_126_513.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_905_1721_327.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_477_915_389.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_32_44_211.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_864_1645_626.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_885_1687_204.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_326_690_797.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_290_445_686.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_810_1526_67.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_587_1117_880.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_567_1077_64.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_167_258_711.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_177_274_865.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_954_1808_415.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_568_1066_235.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_153_233_45.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_422_837_683.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_423_839_596.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_177_274_371.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_887_1688_436.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_16_24_691.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_256_394_449.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_682_1292_978.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_720_1362_700.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_200_304_139.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_237_362_52.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_53_1370_288.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_467_904_776.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_512_962_998.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_827_1598_85.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_313_669_908.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_455_885_115.png',
 '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_980_1849_295.png','''
        st.code(code,language='python')
        
        st.subheader("Create the image id and image path dictionary, so each image id has a corresponding image path, so we are able to find the exact location of each image by the image_id specified in the metadata tabular file:")
        code = '''image_id_and_path_dict = {os.path.basename(x): x for x in skin_image_paths}
image_id_and_path_dict'''
        st.code(code,language='python')
        code = '''{'PAT_456_887_499.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_456_887_499.png',
 'PAT_503_950_552.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_503_950_552.png',
 'PAT_948_1801_721.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_948_1801_721.png',
 'PAT_162_252_773.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_162_252_773.png',
 'PAT_122_185_957.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_122_185_957.png',
 'PAT_691_1311_686.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_691_1311_686.png',
 'PAT_788_1503_541.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_788_1503_541.png',
 'PAT_212_327_532.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_212_327_532.png',
 'PAT_432_851_560.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_432_851_560.png',
 'PAT_442_863_795.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_442_863_795.png',
 'PAT_53_82_657.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_53_82_657.png',
 'PAT_101_1041_651.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_101_1041_651.png',
 'PAT_699_1320_901.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_699_1320_901.png',
 'PAT_31_42_680.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_31_42_680.png',
 'PAT_822_1550_826.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_822_1550_826.png',
 'PAT_82_125_525.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_82_125_525.png',
 'PAT_817_1537_643.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_817_1537_643.png',
 'PAT_660_1249_373.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_660_1249_373.png',
 'PAT_378_769_36.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_378_769_36.png',
 'PAT_270_416_398.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_270_416_398.png',
 'PAT_366_752_782.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_366_752_782.png',
 'PAT_654_1244_16.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_654_1244_16.png',
 'PAT_74_112_729.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_74_112_729.png',
 'PAT_774_1462_821.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_774_1462_821.png',
 'PAT_394_794_235.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_394_794_235.png',
 'PAT_14_22_850.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_14_22_850.png',
 'PAT_82_125_365.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_82_125_365.png',
 'PAT_467_904_535.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_467_904_535.png',
 'PAT_237_363_344.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_237_363_344.png',
 'PAT_272_420_25.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_272_420_25.png',
 'PAT_379_1531_17.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_379_1531_17.png',
 'PAT_126_191_304.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_126_191_304.png',
 'PAT_923_1753_706.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_923_1753_706.png',
 'PAT_788_1503_168.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_788_1503_168.png',
 'PAT_535_1010_303.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_535_1010_303.png',
 'PAT_517_971_88.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_517_971_88.png',
 'PAT_338_707_774.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_338_707_774.png',
 'PAT_251_382_576.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_251_382_576.png',
 'PAT_392_792_712.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_392_792_712.png',
 'PAT_422_837_617.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_422_837_617.png',
 'PAT_901_1716_497.png': '/content/gdrive/My Drive/Data606/Project/Skin_Cancer_Analysis/data/raw_image_data/PAT_901_1716_497.png','''
        st.code(code,language='python')
        
        skin_lesion_abbreviations = metadata.diagnostic.unique()
        print("Skin lesions:", skin_lesion_abbreviations)
        code = '''skin_lesion_abbreviations = metadata.diagnostic.unique()
print("Skin lesions:", skin_lesion_abbreviations)'''
        st.code(code,language='python')
        st.write('Skin lession:',skin_lesion_abbreviations)
        
        st.subheader('Create the lesion type abbrevation and full name dictionary')
        code = '''lesion_type_dict = {
    'ACK': 'Actinic Keratosis',
    'BCC': 'Basal Cell Carcinoma',
    'MEL': 'Melanoma',
    'NEV': 'Nevus',
    'SCC': 'Squamous Cell Carcinoma',
    'SEK': 'Seborrheic Keratosis'
    
}
class_idx = {'ACK': 0, 'BCC': 1, 'MEL': 2, 'NEV': 3, 'SCC': 4, 'SEK': 5}'''
        st.code(code,language='python')
        
        st.subheader("""Add more columns in the metadata data frame for later data retrieving""")
        code = '''metadata['image_path'] = metadata['img_id'].map(image_id_and_path_dict.get)  # image path
metadata['cell_type'] = metadata['diagnostic'].map(lesion_type_dict.get)     # cell type
#metadata['cell_type_label'] = pd.Categorical(metaData_pd['cell_type']).codes    # cell type class label
metadata['cell_type_label'] = metadata['diagnostic'].map(class_idx.get)    # cell type class label
metadata'''
        st.code(code,language='python')
        st.write(metadata)
        
        st.subheader("""Feature Engineering""")
        st.caption('Drop NaN rows')
        code = '''no_nan_pd = metaData_pd.dropna()
no_nan_pd'''
        st.code(code,language='python')
        no_nan_pd = metadata.dropna()
        no_nan_pd
        
        st.subheader("""Use Torchvision to show sample images with class labels""")
        code = '''import torch
import torchvision
from torchvision import transforms

all_images = torchvision.datasets.ImageFolder(skin_categories_path)

train_size = int(len(all_images)*0.8)
test_size = len(all_images)-train_size

train_data, test_data = torch.utils.data.random_split(all_images, (train_size, test_size))
row = 5
col = 5

f, axarr = plt.subplots(row, col, figsize=(20,10))
for i in range(row):
    for j in range(col):
        x, y = test_data[i*4+j]
        axarr[i,j].imshow(x)
        axarr[i,j].text(0.0, 0.5, str(round(y,2)), dict(size=20, color='red'))'''
        st.code(code,language='python')
        image = Image.open(r'C:\Users\marcu\OneDrive\Pictures\Phase 4 screenshot\Data App pics\SkinLessionTest.png')
        st.image(image,caption='Figure 3: The total number of skin lession classes shown in bar format')
        
        st.subheader('DL_Image_Tabular_Combined_DenseNet121')
        
        st.code(code,language='python')
        
        



    
if __name__ == '__main__':
    	main()