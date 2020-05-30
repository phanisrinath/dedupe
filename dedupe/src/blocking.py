import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from gensim.parsing.preprocessing import (strip_multiple_whitespaces,
                                          strip_non_alphanum)
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import os


def read_clean_inference_data(path_=None, prefixes_to_clean=None, app_mode=False, df=None):
    """ Description
    :type path_: string
    :param path_: (optional) path to the file to read

    :type prefixes_to_clean: list
    :param prefixes_to_clean: list of column prefixes i.e just pass ["book", "author"] 
    even though the actual columns being ["book1","book2","author1","author2"].

    :type app_mode: bool
    :param app_mode: (optional) If true then path_ is not used and df should be passed

    :type df: pd.DataFrame
    :param df: pandas dataframe to clearn the dataframe with the prefixes mentioned.

    :type path_: string
    :param path_: (optional) path to the file to read
    
    :rtype: pd.DataFrame
    """
    if app_mode:
        df = df
    else:
        df = pd.read_csv(path_)

    cols = list()
    for i in prefixes_to_clean:
        cols.extend([j for j in df.columns if i in j])

    for i in cols:
        df[i] = df[i].apply(lambda x: (strip_multiple_whitespaces(
            strip_non_alphanum(x.lower().strip()))))
    return df


def create_tfidf_matrix(df, prefixes_to_clean):
    """ Function to create a tfidf matrix. Which is of the shape
    [num_of_recs_to_dedupe,vocab_of_both_prefixes]

    :type df: pd.DataFrame
    :param df: Pandas data frame of cleaned data.

    :type prefixes_to_clean: list
    :param prefixes_to_clean: list of column prefixes i.e just pass ["book", "author"] 
    even though the actual columns being ["book1","book2","author1","author2"].

    :rtype: numpy array of size [num_of_recs_in_df,vocab_of_both_prefixes]

    """
    finalDf = None
    for i in tqdm(prefixes_to_clean):
        corpus = list(df[i])
        cv = CountVectorizer()
        corpus2 = cv.fit_transform(corpus).toarray()
        if finalDf is None:
            finalDf = corpus2
        else:
            finalDf = np.concatenate([finalDf, corpus2], axis=1)
    print(f"Data prepared for passing into Annoy. Shape {str(finalDf.shape)}")
    return finalDf


def create_save_annoy_index(data_to_nearest_neighbor, num_of_trees=10, sim_recs_to_extract=3):
    """ Function to create an annoy index and extract the nearest neighbours of each record.
    This will act as the blocking step for the complete algorithm during inference time
    The reason for using Annoy is to try and keep the solution as fast as possible 

    :type data_to_nearest_neighbor: np.array()
    :param data_to_nearest_neighbor:  Tfidf matrix of size [num_of_recs_in_df,vocab_of_both_prefixes]

    :type num_of_trees: int
    :param num_of_trees: # Axis parallel splits to be made on the space. Similar to LSH hyperplane slicing

    :type sim_recs_to_extract: int
    :param sim_recs_to_extract: number of similar records to extract. Increasing this number could potentially 
    increase the inference time

    :rtype: list of lists where each list represents the indexes of similar records
    """
    f = data_to_nearest_neighbor.shape[1]  # Num of Columns
    t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
    for e, i in enumerate(data_to_nearest_neighbor):
        t.add_item(e, i)

    # 10 trees This needs to be tuned to find the best result
    t.build(num_of_trees)
    # t.save("path") # This index can be saved for faster processing and for parallelizing across
    # other processes

    sims = []
    for i in tqdm(range(data_to_nearest_neighbor.shape[0])):
        # will find the 1000 nearest neighbors
        sims.append(t.get_nns_by_item(i, sim_recs_to_extract))
    print(f"Blocking Performed! Only potential matches identified for further classification.")

    return sims


def prepare_data_for_classification(sims_list, df, prefixes_to_use):
    """ Function to prepare data for classification. 
    This is a pariwise dataset created post blocking.

    :type sims_list: list of lists
    :param sims_list: Each list represents the indexes of similar records

    :type df: pd.DataFrame
    :param df: original data to dedupe of the specific format metioned.

    :type prefixes_to_use: list 
    :param prefixes_to_use: list of column prefixes i.e just pass ["book", "author"] 
    even though the actual columns being ["book1","book2","author1","author2"].

    :raises:

    :rtype: pandas dataframe ready for feature extraction and classification
    """
    data_for_classif = pd.DataFrame()

    def concat_ids(x, y):
        lst = [str(x), str(y)]
        lst.sort()
        return "_".join(lst)

    for e, i in tqdm(enumerate(sims_list)):
        try:
            i.remove(e)  # Removing its ownself as a potential match.
        except ValueError:
            pass
        sim_df = df.loc[i]
        curr_rec = pd.DataFrame(df.loc[e]).transpose()

        cols = prefixes_to_use+["id"]
        curr_rec.columns = [
            i+"1" if i in cols else i for i in curr_rec.columns]
        sim_df.columns = [i+"2" if i in cols else i for i in sim_df.columns]

        sim_df["key"] = e
        curr_rec["key"] = e
        merged = pd.merge(curr_rec, sim_df, on="key")
        merged["comb_key"] = merged.apply(
            lambda x: concat_ids(x["id1"], x["id2"]), axis=1)
        data_for_classif = data_for_classif.append(merged)

    data_for_classif.drop_duplicates(subset=["comb_key"], inplace=True)
    data_for_classif.reset_index(inplace=True, drop=True)
    del data_for_classif["key"]

    print(
        f"Data Preparation Completed. Dataset of size shape{str(data_for_classif.shape)} created.")

    return data_for_classif


if __name__ == "__main__":
    df = read_clean_inference_data(path_="input/data_to_dedupe.csv",
                                   prefixes_to_clean=["book"])
    data_to_nearest_neighbor = create_tfidf_matrix(
        df, prefixes_to_clean=["book"])
    sims_list = create_save_annoy_index(
        data_to_nearest_neighbor, num_of_trees=10, sim_recs_to_extract=3)
    data_for_classif = prepare_data_for_classification(
        sims_list, df, prefixes_to_use=["book"])
    data_for_classif.to_csv("output/data_for_classif_tmp.csv", index=False)
    
