from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.ensemble import GradientBoostingClassifier

def set_model():
	movie_df = pd.read_excel("종합.xlsx")

	X = movie_df[["긍정비율","개봉 이틀째 관객수", "2일차 스크린수","SF"]].values
	y = movie_df["흥행등급"].values

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

	global gbrt
	gbrt = GradientBoostingClassifier(random_state=0,max_depth=3,learning_rate=0.1)
	gbrt.fit(X_train, y_train)
	gbrt._make_predict_function()


def gbrt_predict(x):
	gbrt.predict(x)
