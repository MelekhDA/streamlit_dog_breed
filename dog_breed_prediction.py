from io import TextIOWrapper
import pickle

import streamlit as st
import numpy as np
import pandas as pd

PATH_MLB = 'models/multi_label_binarizer.pkl'
PATH_LOGREG = 'models/ovr_logreg.pkl'
PATH_DF = 'models/df2top.csv'

ACTIVE_DOG = {
    1: 'Have not so much time to train dog',
    2: 'Ready to train each day for hours',
    3: 'Middle'
}

GOAL_IN_LIFE = {
    1: 'Good friend and companion',
    2: 'Clever helper',
    3: 'Partner for your active life'
}

WEIGHT_CATEGORY = {
    1: 'Tiny',
    2: 'Small',
    3: 'Large',
    4: 'Giant'
}


def load_model(path: str, mode: str = 'rb') -> TextIOWrapper:
    with open(path, mode=mode) as fio:
        return pickle.load(fio)


def min_max_proba(array: np.ndarray) -> None:
    array -= array.min()
    array /= array.max()


def norm_proba_to_predict(array: np.ndarray) -> None:
    array[array < 0.9] = 0
    array[array >= 0.9] = 1


class Classifier:

    def __init__(self):
        self.mlb = load_model(PATH_MLB)
        self.logreg = load_model(PATH_LOGREG)
        self.df = pd.read_csv(PATH_DF)

    def predict(self, data: list) -> np.ndarray:
        predict_proba = self.logreg.predict_proba([data])[0]
        min_max_proba(predict_proba)
        norm_proba_to_predict(predict_proba)

        return self.mlb.classes_[predict_proba == 1]

    def get_top(self, data, drop_breed_names):
        group_1, group_2, wt_category = data[0], data[0], data[2]

        breed_names = self.df[(self.df['Group1'] == group_1) &
                              (self.df['Group2'] == group_2) &
                              (self.df['MaleWtKgCategory'] == wt_category)]
        breed_names = breed_names.sort_values(by='PopularityUS2017')['BreedName'].values

        top_breed_names = []
        for breed_name in breed_names:
            if breed_name not in top_breed_names and breed_name not in drop_breed_names:
                top_breed_names.append(breed_name)
            if len(top_breed_names) >= 3:
                break

        return top_breed_names

    def predict_with_top(self, data: list) -> np.ndarray:
        predicts = self.predict(data)
        top_breed_names = self.get_top(data, predicts)

        return np.concatenate([predicts[:3], top_breed_names])


classifier = Classifier()

st.title("Find your dog :)")
st.sidebar.markdown("## About project\n")
st.sidebar.markdown("*** Many people would like to have a pet in their home. "
                    "But they often face various problems, ranging from the size of "
                    "the apartment and ending with the time that needs to be devoted to it. "
                    "Also important is the fact of the presence of a large "
                    "number of breeds and their features. ***")
st.sidebar.markdown("*** This project is aimed at simplifying the procedure for choosing a pet. ***")
st.sidebar.markdown("*** Dogs were considered as pets as an initial stage. "
                    "In the future, the project can be expanded. Please, try :) ***")

left_column, right_column = st.beta_columns(2)
group_1 = left_column.selectbox('Type of activity', [1, 2, 3], format_func=lambda x: ACTIVE_DOG[x], index=0)
group_2 = right_column.selectbox('What kind of dog would you like?', [1, 2, 3], format_func=lambda x: GOAL_IN_LIFE[x],
                                 index=2)

left_column, right_column = st.beta_columns(2)
male_wt_kg_category = left_column.selectbox('Weight category', [1, 2, 3, 4], format_func=lambda x: WEIGHT_CATEGORY[x],
                                            index=1)
watchdog = right_column.selectbox("Watchdog degree", [1, 2, 3, 4, 5, 6])

intelligence = st.select_slider("Intelligence", options=range(1, 81), value=40)
avg_pup_price = st.select_slider("Price", options=range(300, 3001, 100), value=1000)

data = [group_1, group_2, male_wt_kg_category, intelligence, avg_pup_price, watchdog]

dog_breed = classifier.predict_with_top(data)

if len(dog_breed) != 0:
    st.write(', '.join(dog_breed))
else:
    st.write('**No matching dog breed found**')
