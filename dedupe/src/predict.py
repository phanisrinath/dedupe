from sklearn.externals import joblib
import pandas as pd
from tqdm import tqdm

from prepare_data import add_features, read_clean_data

def generate_predictions(data_for_classif_path, prefixes_to_use, model_path=None, app_mode=False, clf=None):
    
    """ Function to Clean, Prepare and Generate Predictions
    :type model_path: str
    :param model_path: path of the model trained. If None then clf has to be passed

    :type data_for_classif_path: str
    :param data_for_classif_path: path of the data generated from blocking step

    :type prefixes_to_use: list
    :param prefixes_to_use: list of column prefixes i.e just pass ["book", "author"] 
    even though the actual columns being ["book1","book2","author1","author2"].
    
    :type app_mode: bool
    :param app_mode: if True then the clf needs to be passed as a parameter else the model will be read from the disk. 
    If this is True then model_path is ignored and clf  cannot be None

    :type clf: sklearn model
    :param clf: model for predictions
    
    :rtype: pd.DataFrame which is like the data generated from bloking but with one additional column of Probability
    """
    if app_mode:
        df_cleaned = read_clean_data(path_=None, df=data_for_classif_path, prefixes_to_clean=prefixes_to_use)
    else:
        if len(prefixes_to_use)==1:
            model_path_to_read = f"models/model_{prefixes_to_use[0]}.pkl"
        else:
            model_path_to_read = f"models/model.pkl"
            
        with open(model_path_to_read, "rb") as f:
            clf = joblib.load(f)
        
        df_cleaned = read_clean_data(path_=data_for_classif_path, prefixes_to_clean=prefixes_to_use)
        
    features = add_features(prefixes_to_extract=prefixes_to_use,df=df_cleaned)
    
    preds = clf.predict_proba(features)[:,1]
    
    del features
    df_cleaned["Probability"] = preds
    print(f"Predictions generated for test set.")
    
    return df_cleaned

def reconcile_duplicates(df_predicted, threshold=0.45):
    
    combos = df_predicted[df_predicted["Probability"]>=threshold]["comb_key"].tolist()
    combos = [set(i.split("_")) for i in combos]
    
    final_dupes = []
    flattened = []
    for i in tqdm(combos):
        if len(final_dupes) == 0:
            final_dupes.append(i)
            flattened.extend(list(i))
        else:
            if len(i.intersection(set(flattened)))>0:
                for e, j in enumerate(final_dupes):
                    if len(j.intersection(i))>0:
                        final_dupes[e] = final_dupes[e].union(i)
                        flattened.extend(list(j))
                    else:
                        pass
            else:
                final_dupes.append(i)
                flattened.extend(list(i))
    
    return final_dupes


def prepare_final_df_to_return(final_dupes, path_for_dedupe, app_mode= False):

    """ Function to create the final data frame after genereating predictions and reconciling them
    :type final_dupes: list of sets
    :param final_dupes: list of duplicate index sets
    
    :type path_for_dedupe: 'str' or 'pd.Dataframe
    :param path_for_dedupe: If app_mode = True then this is a pandas data frame of actual data else path to read the actual file to join results

    :type app_mode: bool
    :param app_mode: to run this function locally set this app_model to False

    :raises:

    :rtype:
    """
    if app_mode:
        finalDf = path_for_dedupe
    else:
        finalDf = pd.read_csv(path_for_dedupe)
    finalDf["Duplicate"] = "unique"
    for e, i in enumerate(final_dupes):
        i = [int(j) for j in i]
        finalDf.Duplicate[finalDf["id"].isin(i)] = f"cluster_{e}"
    if not app_mode:
        finalDf.to_csv('final_result.csv',index=False)
        return "Done"
    else:
        print("Deduplication Completed! File written.")
        return finalDf
    

if __name__ == "__main__":
    df_predicted = generate_predictions(model_path="models/model_book.pkl",
                            data_for_classif_path="output/data_for_classif_tmp.csv",
                            prefixes_to_use=["book"]
                            )
    final_dupes = reconcile_duplicates(df_predicted, threshold=0.45)
    prepare_final_df_to_return(final_dupes, "input/data_to_dedupe.csv")
