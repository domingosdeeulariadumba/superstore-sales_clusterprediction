# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 02:52:43 2023

@author: domingosdeeularia
"""
# %%
""" IMPORTING LIBRARIES """
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Machine Learning Modules
from sklearn.model_selection import train_test_split as tts, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, \
    roc_curve, confusion_matrix

# For ignoring warnings
import warnings
warnings.filterwarnings('ignore')
# %%


# %%
"""" EXPLORATORY DATA ANALYSIS """

df = pd.read_csv('C:/Users/domingosdeeularia/Desktop/' \
                 'superstore_sales_cluster.csv')

    '''
    As this dataset was already used on my superstore cluster analysis, for this
    section we'll simply check again the info, the first and last five records,
    the statistical summary and the pair plot
    '''
# dataset info
df.info()

# First and last five records of the dataset
df.head()
df.tail()

# Statistical summary
df.describe(include = 'all')

# Presenting the pairplot
sb.pairplot(df, hue = 'Cluster ID', markers = '*')
plt.show()
# %%


# %%
"""" BUILDING THE PREDICTION MODEL """

    '''
    Since there are also categorical variables, we'll get dummies for them as
    we create two different sets for  predictors and dependent variable.
    We secondly assign the labels of the outcome to numeric so the model can
    understand them.
    And finally, it is created a function to get the metrics of the models we
    consider for this project (Logistic Regression Lasso and Ridge).
    '''

# Creating sets for explanators and dependent variable
x,y = pd.get_dummies(df.drop('Cluster ID', axis = 1),
                     drop_first = True),df['Cluster ID'].copy()

# Changing the dependent variable labels to numeric
n_classes = len(y.value_counts())  
y = y.apply(lambda id: (n_classes-1) if id == 'Cluster 2' else 0)

# Creating the function to get metrics of the models
def get_scores(model, x, y):
    
    X_scaled = pd.DataFrame(RobustScaler().fit_transform(x), columns=x.columns)
    X_train, X_test, y_train, y_test = tts(X_scaled, y, test_size=0.2, random_state=97)
    X_trainSM, y_trainSM = SMOTE(random_state=97).fit_resample(X_train, y_train)
    feature_scale = X_trainSM.std(axis=0)
    c_values = [0.01 * scale for scale in feature_scale]
    
    # Hyperparameter grid for GridSearchCV
    param_grid = {'C': c_values}
    
    # Recuesive Feature Eliminationv(RFE)
    X_train_RFE = None  
    X_test_RFE = None
    
    if model == 'Lasso':
        # Lasso (L1 regularization) with logistic regression
        grid_search = GridSearchCV(LogisticRegression(penalty='l1', \
                                                      solver='liblinear'), \
                                   param_grid, cv=5, scoring='accuracy')
        
        grid_search.fit(X_trainSM, y_trainSM)
        
        best_C = grid_search.best_params_['C']
        class_model = LogisticRegression(penalty='l1', C=best_C,
                                         solver='liblinear')
        class_model.fit(X_trainSM, y_trainSM)
        y_model_pred = class_model.predict(X_test)
    
    else:
        
        rfe = RFE(LogisticRegression(), n_features_to_select=None)
        rfe_fit = rfe.fit(X_scaled, y)
        X_train_RFE = X_trainSM[X_trainSM.columns[rfe_fit.support_.tolist()]]
        X_test_RFE = X_test[X_trainSM.columns[rfe_fit.support_.tolist()]]
        
        
        if model == 'Ridge':
            
            # Ridge (L2 regularization) with logistic regression
            grid_search = GridSearchCV( \
                LogisticRegression(penalty='l2', \
                                   solver='liblinear'), \
                    param_grid, cv=5, scoring='accuracy')
            
            grid_search.fit(X_train_RFE, y_trainSM)
                
            best_C = grid_search.best_params_['C']
            
            class_model = LogisticRegression(penalty='l2',
                                             C=best_C, solver='liblinear')
            
            class_model.fit(X_train_RFE, y_trainSM)
            
            y_model_pred = class_model.predict(X_test_RFE)
            
        else:
            
            grid_search = GridSearchCV(LogisticRegression(),
                                       param_grid, cv=5, scoring='accuracy')
            
            grid_search.fit(X_train_RFE, y_trainSM)
            
            best_C = grid_search.best_params_['C']
            
            class_model = LogisticRegression( C=best_C)
            
            class_model.fit(X_train_RFE, y_trainSM)
            
            y_model_pred = class_model.predict(X_test_RFE)
             
    report = classification_report(y_test, y_model_pred)
    
    cm = confusion_matrix(y_test, y_model_pred)
    
    ras = roc_auc_score(y_test, y_model_pred)
            
    return report, cm, ras, class_model, y_test, X_test_RFE

# Checking the metrics for the three models
for i in ['Lasso', 'Ridge', 'Logistic Regression']:
    report = get_scores(i, x, y)[0]
    print(f'{i}:\n{report}')

    '''
    The three models have presented considerably high performance. Being Lasso
    with 0.94, Ridge with 0.91 and Logistic Regression with 0.96. So we'll next
    present the AUC-ROC Curve for the latter. But, before this step, let us
    analyse its confusion matrix.
    '''
print('Confusion Matrix:\n', get_scores('other', x, y)[1])

    '''
    Printing the line above it was found that there is 93 out of 97 correct
    predictions. That is, 85 True Negatives, 8 True Posites, and 4 False Positives
    '''
# AUC-ROC Curve
auc_roc = get_scores('other', x, y)[2]
model = get_scores('other', x, y)[3]
y_test, X_test = get_scores('other', x, y)[4], get_scores('other', x, y)[5]
y_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, tresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, 'b-.', label = 'AUC = '+str(round(auc_roc,2)))
plt.plot([0,1], [0,1], 'k--')
plt.title('Receiver Operating Characterictic')
plt.ylabel('True Positive Rate')
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.xlim([0.0, 1.05])
plt.legend(loc = 'best')
# %%      
                  ________  ________   _______   ______ 
                 /_  __/ / / / ____/  / ____/ | / / __ \
                  / / / /_/ / __/    / __/ /  |/ / / / /
                 / / / __  / /___   / /___/ /|  / /_/ / 
                /_/ /_/ /_/_____/  /_____/_/ |_/_____/  

# %%                                     