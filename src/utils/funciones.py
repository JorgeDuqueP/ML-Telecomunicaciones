import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, mean_absolute_error
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.tree import DecisionTreeClassifier, plot_tree
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler 
from sklearn.datasets import make_classification
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
import os
import sys



os.getcwd()
root_path=os.path.dirname(os.getcwd())
root_path
sys.path.append(root_path)
sys.path

print('Comienza')
#Importamos el csv
def read_churn(path):
    pd.read_csv(path)
    return path

def tratamiento_data():
    
    churn = pd.read_csv('data\\row\\churn.csv')
    # Convertimos las columnas de categóricas a numéricas
    le = preprocessing.LabelEncoder()
    columns = ['gender', 'Dependents', 'Partner', 'CallService',
            'MultipleConnections', 'InternetConnection', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtectionService', 'TechnicalHelp', 'OnlineTV',
            'OnlineMovies', 'Agreement', 'PaymentMethod', 'TotalAmount', 'Churn']
    def numeric(columns):
        for i in columns:
            le.fit(churn[i])
            churn[i] = le.transform(churn[i])
        return columns
    numeric(columns)

# Hacemos la media de las columnas TotalAmount y tenure ya que hemos comprobado que mejoran las predicciones
    churn['Media'] = (churn['TotalAmount'] + churn['tenure']) / 2
# Borramos las columnas quedándonos con la media de las dos    
    churn.drop(['TotalAmount', 'tenure'], axis=1, inplace=True)
# Generamos la X y la y y las dividimos en train y test
    X = np.array(churn.drop(['Churn'], 1))
    y = np.array(churn['Churn'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)


# Generamos los modelos
def predicción_RForest():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
    forest_model = RandomForestClassifier(n_estimators=99, max_depth=44, criterion= 'entropy')
    forest_model.fit(X_train, y_train)
    y_pred_RFor3 = forest_model.predict(X_test)
    return y_pred_RFor3
