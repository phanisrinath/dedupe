import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import joblib


def train_model(data_to_build_model, prefixes_to_use, clf=None):
    """ 
    Function to build a model. The model currently is a sklearn model can be changed
    to any deep learning model in further iterations

    :type data_to_build_model: str
    :param data_to_build_model: path to read data for model building

    :type prefixes_to_use: list
    :param prefixes_to_use: list of column prefixes i.e just pass ["book", "author"] 
    even though the actual columns being ["book1","book2","author1","author2"].

    :type clf: sklearn.model
    :param clf: Any sklearn model to just fit and dump. If this is provided then no
    CV is done. Instead just fit and dumped

    """
    train_data = pd.read_csv(data_to_build_model)

    cols_to_keep = list()
    for i in prefixes_to_use:
        cols = train_data.columns
        cols_to_keep.extend(cols[cols.str.startswith(f"{i}_")])

    cols_to_keep.append("label")
    train_data = train_data[cols_to_keep]

    y = train_data["label"]
    X = train_data.drop(["label"], axis=1)

    if clf:
        print("Model provided. Not peforming any CV. Directly fitting and dumping.")
    else:
        clf = RandomForestClassifier()
        scores = cross_val_score(clf, X, y, scoring="f1", cv=4, n_jobs=-1)
        print(f"Cross validation scores: {str(scores)}. Mean: {scores.mean()}")

    # Finally building the new model
    clf.fit(X, y)
    print("Model built successfully.")

    if len(prefixes_to_use)==1:
        model_path_to_write = f"models/model_{prefixes_to_use[0]}.pkl"
    else:
        model_path_to_write = f"models/model.pkl"
    with open(model_path_to_write, "wb") as f:
        joblib.dump(clf, f)

    print(f"Model dumped : {model_path_to_write}")

    return "Done"


if __name__ == "__main__":
    # train_model("output/prepared_data.csv",
    #             ["book", "author"])
    train_model("output/prepared_data_2.csv",
            ["title"])
