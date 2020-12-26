from io import TextIOWrapper
from time import time
import pickle

import streamlit as st
import numpy as np

PATH_MLB = 'models/multi_label_binarizer.pkl'
PATH_LOGREG = 'models/ovr_logreg.pkl'


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

    def predict(self, data: list) -> np.ndarray:
        predict_proba = self.logreg.predict_proba([data])[0]
        min_max_proba(predict_proba)
        norm_proba_to_predict(predict_proba)

        return self.mlb.classes_[predict_proba == 1]


time_from = time()
classifier = Classifier()
st.write(f'classifier loaded in {round(time() - time_from, 4)} seconds')


group_1 = st.selectbox('Type of activity', [1, 2, 3])
group_2 = st.selectbox('Dog proposal', [1, 2, 3])
male_wt_kg_category = st.selectbox('Dog proposal', [1, 2, 3, 4])
intelligence = st.select_slider("Intelligence", options=range(1, 80))
avg_pup_price = st.select_slider("Price", options=range(350, 3001))
watchdog = st.selectbox("Watchdog", [1, 2, 3, 4, 5, 6])

data = [group_1, group_2, male_wt_kg_category, intelligence, avg_pup_price, watchdog]

dog_breed = classifier.predict(data)

if len(dog_breed) != 0:
    st.write(', '.join(dog_breed))
else:
    st.write('**No matching dog breed found**')
