import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(data,ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ == '__main__': 

    # Read the Data 
    df = pd.read_csv(r"C:\Users\yash mohite\OneDrive\Desktop\corona virus solution\data.csv.csv")
    train, test = data_split(df, 0.2)
    x_train = train[["feaver", "bodypain", "age", "runnyNose", "diffbreath"]].to_numpy()
    x_test = test[["feaver", "bodypain", "age", "runnyNose", "diffbreath"]].to_numpy()

    y_train = train[["infectionprob"]].to_numpy().reshape(2204 ,)
    y_test = test[["infectionprob"]].to_numpy().reshape(550 ,)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # open a file, where you ant to store the data
    file = open("model.pkl", "wb")

    # dump information to that file
    pickle.dump(clf, file)
    file.close()

    
