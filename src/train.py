from utils.funciones import *
import os
import sys

os.chdir(os.path.dirname(__file__))

read_churn('data\\row\\churn.csv')

tratamiento_data()

predicción_RForest()
