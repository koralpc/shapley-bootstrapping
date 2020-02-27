import pandas as pd
import numpy as np
import sys
import shap
from openml import datasets


#sys.path.append("..")

def returnDataset(idx):
    dataset_holder = [admissions(),amazon(),boston(),crime(),diabetes(),fish_weights(),insurance(),nhanes(),nye_airbnb(),student_grades(),liver_disorder()]
    X,y,name = dataset_holder[idx]
    X.reset_index(inplace = True)

    X.drop(['index'],axis = 1,inplace = True)
    if name == 'Nhanes':
        X.drop(['Unnamed: 0'],axis = 1,inplace = True)

    ## Dropping NA values
    drop_indexes = np.unique(np.where(X.isna())[0])
    X.dropna(inplace = True)
    mask = np.ones(len(y),dtype = bool)
    mask[drop_indexes] = False
    y = y[mask]

    return X,y,name

def boston():
    X,y = shap.datasets.boston()
    name = 'Boston'
    return X,y,name

def diabetes():
    X,y = shap.datasets.diabetes()
    name = 'Diabetes'
    return X,y,name

def crime():
    X,y = shap.datasets.communitiesandcrime()
    name = 'Crime'
    return X,y,name

def nhanes():
    X,y = shap.datasets.nhanesi()
    name = 'Nhanes'
    return X,y,name

def fish_weights():
    fish_df = pd.read_csv('../Data/Fish.csv')
    X = fish_df.drop('Weight',axis = 1)
    X_fac = pd.factorize(X['Species'])
    X['Species'] = X_fac[0]
    y = np.array(fish_df['Weight'])
    name = 'Fish_weights'
    return X,y,name

def nye_airbnb():
    nye_df = pd.read_csv('../Data/AB_NYC_2019.csv')
    nye_df.drop(['id','name','host_id','host_name','latitude','longitude','last_review'],axis = 1,inplace = True)
    nye_df['neighbourhood_group'] = pd.factorize(nye_df['neighbourhood_group'])[0]
    nye_df['neighbourhood'] = pd.factorize(nye_df['neighbourhood'])[0]
    nye_df['room_type'] = pd.factorize(nye_df['room_type'])[0]
    X = nye_df.drop('price',axis = 1)
    y = np.array(nye_df['price'])
    name = 'NYE_Airbnb'
    return X,y,name


def admissions():
    admission_df = pd.read_csv('../Data/Admission_Predict.csv')
    X = admission_df.drop(['Serial No.', 'Chance of Admit '],axis = 1)
    y = np.array(admission_df['Chance of Admit '])
    name = 'Admissions'
    return X,y,name


def student_grades():
    grades_df = pd.read_csv('../Data/student-mat.csv')
    grades_df['school'] = pd.factorize(grades_df['school'])[0]
    grades_df['sex'] = pd.factorize(grades_df['sex'])[0]
    grades_df['address'] = pd.factorize(grades_df['address'])[0]
    grades_df['famsize'] = pd.factorize(grades_df['famsize'])[0]
    grades_df['Pstatus'] = pd.factorize(grades_df['Pstatus'])[0]
    grades_df['Mjob'] = pd.factorize(grades_df['Mjob'])[0]
    grades_df['Fjob'] = pd.factorize(grades_df['Fjob'])[0]
    grades_df['reason'] = pd.factorize(grades_df['reason'])[0]
    grades_df['guardian'] = pd.factorize(grades_df['guardian'])[0]
    grades_df['schoolsup'] = pd.factorize(grades_df['schoolsup'])[0]
    grades_df['famsup'] = pd.factorize(grades_df['famsup'])[0]
    grades_df['paid'] = pd.factorize(grades_df['paid'])[0]
    grades_df['activities'] = pd.factorize(grades_df['activities'])[0]
    grades_df['nursery'] = pd.factorize(grades_df['nursery'])[0]
    grades_df['higher'] = pd.factorize(grades_df['higher'])[0]
    grades_df['internet'] = pd.factorize(grades_df['internet'])[0]
    grades_df['romantic'] = pd.factorize(grades_df['romantic'])[0]
    X = grades_df.drop('G3',axis = 1)
    y = np.array(grades_df['G3'])
    name = 'Student_grades'
    return X,y,name


def amazon():
    amazon_df = pd.read_csv('../Data/amazon.csv')
    amazon_df['state'] = pd.factorize(amazon_df['state'])[0]
    amazon_df['month'] = pd.factorize(amazon_df['month'])[0]
    X = amazon_df.drop(['date','number'],axis = 1)
    y = np.array(amazon_df['number'])
    name = 'Amazon'
    return X,y,name


def insurance():
    insurance_df = pd.read_csv('../Data/insurance.csv')
    insurance_df['sex'] = pd.factorize(insurance_df['sex'])[0]
    insurance_df['smoker'] = pd.factorize(insurance_df['smoker'])[0]
    insurance_df['region'] = pd.factorize(insurance_df['region'])[0]
    X = insurance_df.drop('charges',axis = 1)
    y = np.array(insurance_df['charges'])
    name = 'Insurance'
    return X,y,name


def liver_disorder():
    data,_,_,_ = datasets.get_dataset(8).get_data()
    X = data.iloc[:,:-1]
    y = np.array(data.iloc[:,-1])
    name = 'Liver_Disorder'
    return X,y,name
