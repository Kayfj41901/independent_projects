import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

def split_bc(df):
    '''
    Takes in a prepped breast cancer dataframe, splits it into train, validate and test subgroups stratifying it on the target of status
    and then returns those subgroups.

    Arguments: df - a cleaned pandas dataframe with the expected feature names and columns in the breast cancer dataset
    Return: train, validate, test - dataframes ready for the exploration and model phases. 
    ''' 
    train, test = train_test_split(df, train_size = 0.8, stratify = df.Status, random_state = 1234)
    train, validate = train_test_split(train, train_size = 0.7, stratify = train.Status, random_state = 1234)
    return train, validate, test

    