import pandas as pd
import numpy as np
from DT import ID3,iteration
from random import seed
from random import randint,sample
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# First: Read the csv file
DF = pd.read_csv("zoo.csv",
                 names=['animal_name', 'hair', 'feathers', 'eggs', 'milk',
                        'airbone', 'aquatic', 'predator', 'toothed', 'backbone',
                        'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize', 'class'])

DF = DF.drop("animal_name", axis=1)


def randomsplitdata(data):
    # seed random number generator
    Loc = sample([i for i in range(data.__len__())],int(data.__len__()*0.8))

    # generate some integers
    Selecteddata = data.loc[Loc]
    return Selecteddata


def randomforest(data, numoftree=30):
    Forest = []
    for i in range(numoftree):
        #split data
        splitdata = randomsplitdata(data)
        Forest.append(ID3(splitdata,splitdata,splitdata.columns[:-1],random=True,numfeature=2))
    return Forest


def testrf(forest,testdata):

    # change Dataframe to dict
    queries = testdata.iloc[:, :-1].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["predicted"])

    for i in range(len(testdata)):
        prd = []
        for j in range(len(forest)):
            prd.append(iteration(forest[j],queries[i]))

        predicted.loc[i, "predicted"] = np.unique(prd)[np.argmax(np.unique(prd,return_counts=True)[1])]

    accuracy = np.sum(predicted["predicted"] == testdata["class"]) * 100 / len(testdata)
    print("accuracy is:", np.sum(predicted["predicted"] == testdata["class"]) * 100 / len(testdata), "%")
    return accuracy

training_data = DF.iloc[:60].reset_index(drop=True)
train = DF.iloc[:60].reset_index(drop=True)
Y_train = training_data["class"]
training_data.drop(["class"],axis=1,inplace=True)
test_data = DF.iloc[60:].reset_index(drop=True)
test = DF.iloc[60:].reset_index(drop=True)
Y_test = test_data["class"]
test_data.drop(["class"],axis=1,inplace=True)
accuracy = []

Fr = randomforest(train, 50)
print(len(Fr))
testrf(Fr,test)
# for i in range(10,20):
#
#     Fr = randomforest(training_data, i)
#     accuracy.append(testrf(Fr,test_data))
#
# x = np.arange(10,20,1)
# plt.plot(x,accuracy)
# plt.show()

model = RandomForestClassifier(n_estimators=50, random_state=0)

model.fit(training_data, Y_train)
preds = model.predict(test_data)

print("accuracy is:", np.sum(preds == Y_test) * 100 / len(Y_test), "%")