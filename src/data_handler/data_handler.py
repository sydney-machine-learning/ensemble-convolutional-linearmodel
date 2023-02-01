import os
from os.path import dirname
import pandas as pd
import numpy as np


class DataHandler:
    def __init__(self):
        pass

    def process_data(self):
        dataset_path = os.path.join(dirname(dirname(__file__)), "dataset", "titanic", "train.csv")
        df = pd.read_csv(dataset_path)
        df.loc[df['Age'].isnull(), 'Age'] = np.round(df['Age'].mean())
        df.loc[df['Embarked'].isnull(), 'Embarked'] = df['Embarked'].value_counts().index[0]

        df = df.join(pd.get_dummies(df['Sex']))
        df = df.drop('Sex', axis=1)

        df = df.join(pd.get_dummies(df['Embarked']))
        df = df.drop('Embarked', axis=1)

        features = ['Pclass', 'female', 'male', 'Age', 'SibSp', 'Parch', 'Fare', 'C', 'Q', 'S']
        nb_train = int(np.floor(0.9 * len(df)))
        df = df.sample(frac=1, random_state=42)

        X_train = df[features][:nb_train].to_numpy()
        y_train = df['Survived'][:nb_train].values
        X_test = df[features][nb_train:].to_numpy()
        y_test = df['Survived'][nb_train:].values
