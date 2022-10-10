# Heart-Stroke-Prediction-Analysis
This project is to predict early heart stroke based on the lifestyle of an individual using ML algorithms.

## Project Description:
This project is about predicting early heart strokes that helps the society to save human lives using Logistic Regression, Random Forest, KNN, Neural Networks and Ensemble Models. These ML alogorithms are applied on “Healthcare-Dataset-Stroke-Data.csv” Dataset from Kaggle, based on 5110 observations with 12 explanatory variables.
Dataset: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset/metadata)

## Goal: 
The project is aimed at implementing a model(s) to predict early heart strokes in an individual.
<br />•	Clean variables, build what is needed
<br />•	Models: Logistic Regression, KNN techniques, RandomForest, Ensemble Learning & Neural Networks
<br />•	Choose the best model having best accuracy.

## Business Problem:
Heart Stroke is one of the severe health hazards, According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. Therefore, early heart stroke prediction helps the society to save human lives. Many of these strokes can be avoided by adopting healthier lifestyle, and monitoring individuals who are most at risk can significantly improve the results.  This project focuses on identifying stroke risk factors and offers suggestions for how to avoid them.

## Data Exploration and Preprocessing:
The Dataset contains data of 5110 observations of an individual with 11 column variables, 9 of which are predictive to our outcome of stroke and one of which is an identification quantifier for our patients.
The data set include variables such as :
<br />• **id:**	Unique identifier
<br />• **gender:**	“Male”,” Female” or “Other”
<br />• **age:**	Age of the patient
<br />• **hypertension:**	0 if the patient doesn't have hypertension, 1 if the patient has hypertension
<br />• **heart_disease:**	0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
<br />• **ever_married:**	"No" or "Yes"
<br />• **work_type:** "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
<br />• **residence_type:**	"Rural" or "Urban"
<br />• **avg_glucose_level:**	average glucose level in blood
<br />• **BMI:**	body mass index
<br />• **smoking_status:** 	"formerly smoked", "never smoked", "smokes" or "Unknown"
<br />• **stroke:**  1 if the patient had a stroke or 0 if not

## Data Cleaning:
<br />• Replaced missing values with mean in column name BMI.
<br />• Dropped column ID which is irrelevant.
<br />• Checked for outliers, data entry errors.
<br />• One Hot Encoding
<br />• Preformed OverSampling using MWMOTE

As part of data pre-processing, performed one hot encoding on the dataset to convert categorical variables to machine readable format (numerical values). After encoding went ahead and checked for the imbalances in data and found asymmetry in data. Performed oversampling on the dataset using MWMOTE (Majority Weighted Minority Oversampling Technique) and balanced the data.  

## Models and their comparison:
Implemented the below models:
<br />• Logistic Regression
<br />• RandomForest
<br />• Classification using K-Nearest Neighbors
<br />• Neural Networks
<br />• Ensemble method

## Logistic Regression:
Since this is a classification problem and expected some linear relationships between variables, used a logistic regression model to classify the data.

The Logistic Regression model on the testing data gives an accuracy value of 87.07%. 

## Classification using K-Nearest Neighbors:
KNN stands for K-Nearest Neighbors. It is a supervised learning algorithm. It is often used as a benchmark for more complex classifiers such as Artificial Neural Networks (ANN) and Support Vector Machines (SVM). Used 14 independent features for KNN implementation. A robust implementation must consider feature engineering, data cleaning, and cross-validation.
<br />• K means clustering
<br />• K = 5
<br />• Sampling 80% of data for training the algorithms using random sampling

Implemented KNN with different optimal weights by changing k values and this time the accuracy achieved is 87.1%.

## Random Forest:
Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time.

Accuracy: 94.65%

## Neural Network:
Neural networks are a class of machine learning algorithms used for complex patterns in datasets using multiple hidden layers and non-linear activation functions. They are also known as artificial neural networks (ANNs) or simulated neural networks (SNNs).

Implemented in the scenario and the accuracy achieved for the testing set is 91.2%.

## Ensemble: Voting & Weighted
Ensemble methods are techniques that create multiple models and then combine them to produce improved results. Ensemble methods usually produces more accurate solutions than a single model would. Implemented ensemble techniques with three models: Logistic Regression, Neural Network and KNN in this project.
The accuracy attained is 94.38% for weighted average model.


Results:
Below is the accuracy for all the five models implemented in the project:

<br />• Logistic Regression	87.07 %
<br />• KNN	87.1%
<br />• Random Forest	94.65 %
<br />• Neural Networks	91.2 %
<br />• Ensemble	Weighted: 94.3 %


Random Forest performed the best with an accuracy of 94.6% followed by Ensemble and random forest with an accuracy of 94.3%.

## Conclusion:
Age is a major risk factor for stroke. As we get older, we are more at risk to suffer a stroke. Males and females both suffer stroke at a similar rate. However, females have been shown to suffer strokes at younger ages than males. Heart problems like hypertension and heart disease greatly increase the risk of stroke. People who have been married are at a higher risk of stroke. This may be due to higher levels of stress that occur during married life. We got better accuracy while using the Random Forest Machine learning model of 94.65 To improve variance and bias we used weighted average and got an accuracy of 94.3


