#Breast_Cancer_Detection_MachineLearning_Project
#Step: 1 Import relevant Libraries
​
import pandas as pd            ## this used for manipulation
import matplotlib.pyplot as plt          #this  used for visualization
import numpy as np                      # this is used for numeric calculation
import seaborn as sn           #this used for advanced visualization
​
#Step:2 Load_Breast_Cancer_Dataset
from sklearn.datasets import load_breast_cancer
cancer_dataset =  load_breast_cancer()
cancer_dataset    # This data is in the form of Dictionary

cancer_dataset.keys() # Here we check about keys in data


#Therefore the "Target" stores values of malignant or begnign tumors. and in output "Target value" shows their name.
cancer_dataset['target_names']

#So, 0 means Malignant Tumor--> Patient has not suffering from breast_cancer
#    1 means Benign Tumor-----> Patients have a breast_cancer


#Create DataFrame
cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],
                        columns = np.append(cancer_dataset['feature_names'],['target']))

# Create DataFrame by concate 'data' and 'target' together
cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],
                        columns = np.append(cancer_dataset['feature_names'],['target']))
​
cancer_df.to_csv("breast_cancer_dataframe.csv") #Just save dataset in csv file

#DATA_PREPROCESSING:
X = cancer_df.drop(['target'],axis = 1)
X.head(8)

# Import library for Train_Test_Split
# Now we have to split the data into Train and Test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Features scaling-->This is help to converting different units & magnitude in one unit.

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()     # Here, we take "sc" as a variable in which we stores value..
X_train_sc = sc.fit_transform(X_train)
X_test_sc  = sc.transform(X_test)


from sklearn.metrics import confusion_matrix  # this is describe the performance of a classification model(or"Classifier")
from sklearn.metrics import classification_report # for classification report of data
from sklearn.metrics import accuracy_score       # tell its good accuarcy score


# By Support_Vector_Machine

from sklearn.svm import SVC
svc_classifier = SVC()
svc_classifier.fit(X_train,y_train)
y_pred = svc_classifier.predict(X_test)
accuracy_score(y_test,y_pred)


# Same but with StandardScale to check accuracy..

svc_classifier2 = SVC()
svc_classifier2.fit(X_train_sc, y_train)
y_pred2_sc = svc_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred2_sc)


print(classification_report(y_test, y_pred2_sc))




import pickle
 
# save model
pickle.dump(svc_classifier2, open('breast_cancer_detector.pickle', 'wb'))
 
# load model
breast_cancer_detector_model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))
 
# predict the output
y_pred = breast_cancer_detector_model.predict(X_test)
 



















