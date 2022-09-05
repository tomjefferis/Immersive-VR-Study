import os
import pandas as pd

#load data from ../EEG folder, all csv files
def get_data():
    data = []
    scores = []
    for file in os.listdir("../EEG/"):
        if file.endswith(".csv"):
            filepath = "../EEG/" + file
            print(file)
            data.append(pd.read_csv(filepath))

        #get scores from file names, 1 = watching, 2 = normal, 3 = hard
        if "watching" in file:
            scores.append(1)
        elif "normal" in file or "correct" in file:
            scores.append(2)
        elif "hard" in file:
            scores.append(3)

    return data


dat = get_data()
print(dat[0].shape)