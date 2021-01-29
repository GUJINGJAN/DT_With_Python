import pandas as pd
import numpy as np
from random import seed
from random import randint,sample

# First: Read the csv file
DF = pd.read_csv("zoo.csv",
                 names=['animal_name', 'hair', 'feathers', 'eggs', 'milk',
                        'airbone', 'aquatic', 'predator', 'toothed', 'backbone',
                        'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize', 'class'])

DF = DF.drop("animal_name", axis=1)


def entropy(target_col):
    """
    :describe Entropy only focus on Target_col
    :param traget_col:
    :return: Entropy of target col
    """
    elements, counts = np.unique(target_col, return_counts=True)

    Entropy = np.sum([(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts))
                      for i in range(len(elements))])

    return Entropy


def Info_Gain(data, split_attribute, target_name="class"):
    """
    g(data,split_attribute) = H(data) -H(D|A)
    H(data): Entropy of the whole input dataset,use target to calculate
    H(D|A): Condition Entropy D give to A

    :param data: Data that we want find the IG.
    :param split_attribute: Feature name we need split ,for example, eggs "1" means yes animal has eggs
    "0" means this animal has no egg.
    :param target_name: In this Data is the class we want to devide
    :return: Info Gain We want
    """

    H_data = entropy(data[target_name])
    keys, values = np.unique(data[split_attribute], return_counts=True)

    H_DA = np.sum(
        [(values[i] / np.sum(values)) * entropy(data.where(data[split_attribute] == keys[i]).dropna()[target_name])
                   for i in range(len(keys))])

    g_info = H_data - H_DA

    return g_info


def ID3(Data, orginaldata, features, target_attribute_name="class", parent_node_class=None, random=False, numfeature=2):
    """
    Build a decision Tree from training Data , use iteration to generate the TREE.

    :param Data: All training data
    :param orginaldata: Data from the last iteration
    :param features: columns name of Data exclude target name
    :param target_attribute_name: for classify task it's "class"
    :param parent_node_class: parent node class,in case we can't divide data with feature.
    :return: Tree structure
    """

    # set stop criteria
    # Case1: Data Entropy is zero,no more split is needed,return it's own class name
    if len(np.unique(Data[target_attribute_name])) <= 1:
        return np.unique(Data[target_attribute_name])[0]

    # Case2: Data is empty,just return class from last iteration
    elif len(Data) == 0:
        return np.unique(orginaldata[target_attribute_name])[
            np.argmax(np.unique(orginaldata[target_attribute_name], return_counts=True)[1])]

    # Case3: There is no more features for splitting
    elif len(features) == 0:
        return parent_node_class
    else:
        # set parent_node_class
        parent_node_class = np.unique(Data[target_attribute_name])[
            np.argmax(np.unique(Data[target_attribute_name], return_counts=True)[1])]

        # Calculate the InfoGain and find the best Feature to split the data
        if random:
            seed(1)

            if len(features) >= numfeature:

                ind = sample([i for i in range(len(features))],numfeature)

                choosefeature = [features[loc] for loc in ind]


                list_info_gain = [Info_Gain(Data, feature) for feature in choosefeature]
                print(list_info_gain)

                best_feature_index = np.argmax(list_info_gain)
                best_feature = features[ind[best_feature_index]]

            else:
                random = False

        if not random:

            list_info_gain = [Info_Gain(Data, feature) for feature in features]
            best_feature_index = np.argmax(list_info_gain)
            best_feature = features[best_feature_index]


        # Define the structure of Tree
        tree = {best_feature: {}}

        # Remove the Feature we have used
        features = [i for i in features if i != best_feature]



        # Do iteration
        for boolvalue in np.unique(Data[best_feature]):
            sub_data = Data.where(Data[best_feature] == boolvalue).dropna()

            subtree = ID3(sub_data, Data, features, target_attribute_name, parent_node_class,random=random,numfeature=numfeature)

            tree[best_feature][boolvalue] = subtree

        return tree


def iteration(tree, test_data):
    """
    Use iteration to do a test
    :param tree: Tree we have trained
    :param test_data: Valid data
    :return: prediction class
    """
    global prd

    # if tree is a number stop iteration and return the number
    if isinstance(tree, (int, float)): return tree

    for key in tree.keys():
        # new tree
        new = tree[key]
        # if we can't find suitable Branch just return default 1

        if test_data[key] not in new.keys(): return 1
        # find suitable Branch and generate new tree
        new1 = new[test_data[key]]

        # Do iteration
        prd = iteration(new1, test_data)

    return prd


def test(data, tree):
    """
    Do test
    :param data: test data
    :param tree: TREE
    :return: print accuracy of prediction
    """
    # change Dataframe to dict
    queries = data.iloc[:, :-1].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["predicted"])

    for i in range(len(data)):
        predicted.loc[i, "predicted"] = iteration(tree, queries[i])
    print("accuracy is:", np.sum(predicted["predicted"] == data["class"]) * 100 / len(data), "%")

# training_data = DF.iloc[:80].reset_index(drop=True)
# test_data = DF.iloc[80:].reset_index(drop=True)
# trees = ID3(training_data, training_data, training_data.columns[:-1])
# test(test_data, trees)
