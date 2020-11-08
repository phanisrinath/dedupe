import pandas as pd
import numpy as np
from gensim.parsing.preprocessing import strip_non_alphanum, strip_multiple_whitespaces
import textdistance
from tqdm import tqdm


def read_clean_data(path_, df=None, prefixes_to_clean=None, fill_missing=''):
    """ Description
    :type path_: string
    :param path_: (optional) path to the file to read

    :type df: pd.DataFrame
    :param df: (optional) dataframe for cleansing

    :type prefixes_to_clean: list
    :param prefixes_to_clean: list of column prefixes i.e just pass ["book", "author"] 
    even though the actual columns being ["book1","book2","author1","author2"].

    :raises:

    :rtype:
    """
    if path_ is not None:
        df = pd.read_csv(path_)
    
    if fill_missing is not None:
        df.fillna(value='', inplace=True)
    
    cols = list()
    for i in prefixes_to_clean:
        cols.extend([j for j in df.columns if i in j])

    for i in cols:
        df[i] = df[i].apply(lambda x: (strip_multiple_whitespaces(
            strip_non_alphanum(x.lower().strip()))))
    return df


def compute_edit_distance_row(str1, str2, prefix):
    """ Function to create the edit distanc measures for a given row. 
    These distance measures will be used as features for the classfication model
    reference-https://pypi.org/project/textdistance/ . This function also includes distance measures
    Open refine. N Gram Fingerprint distances (Levenshtein on Fingerprints) https://github.com/OpenRefine/OpenRefine/wiki/Clustering-In-Depth

    :type str1: string
    :param str1: First string to compare

    :type str2: string
    :param str2: Second string to compare

    :type prefix: string
    :param prefix: prefix to be added to the new columns generated

    :rtype: pd.DataFrame: which is a single row dataframe that has all the edit distance mesures added
    """
    tmp_dict = dict()
    tmp_dict[prefix +
             "_hamming_distance"] = textdistance.hamming.distance(str1, str2)
    tmp_dict[prefix +
             "_hamming_norm_similarity"] = textdistance.hamming.normalized_similarity(str1, str2)
    tmp_dict[prefix +
             "_levenshtein_norm_distance"] = textdistance.levenshtein.normalized_distance(str1, str2)
    tmp_dict[prefix +
             "_levenshtein_distance"] = textdistance.levenshtein.distance(str1, str2)
    tmp_dict[prefix +
             "_jaro_winkler_similarity"] = textdistance.jaro_winkler.similarity(str1, str2)
    tmp_dict[prefix +
             "_jaro_winkler_norm_distance"] = textdistance.jaro_winkler.normalized_distance(str1, str2)
    tmp_dict[prefix +
             "_jaccard_similarity"] = textdistance.jaccard.similarity(str1, str2)
    tmp_dict[prefix +
             "_jaccard_norm_distance"] = textdistance.jaccard.normalized_distance(str1, str2)
    tmp_dict[prefix +
             "_sw_norm_similarity"] = textdistance.smith_waterman.normalized_similarity(str1, str2)
    tmp_dict[prefix +
             "_sw_similarity"] = textdistance.smith_waterman.similarity(str1, str2)
    tmp_dict[prefix +
             "_cosine_similarity"] = textdistance.cosine.similarity(str1, str2)
    tmp_dict[prefix +
             "_bag_norm_similarity"] = textdistance.bag.normalized_similarity(str1, str2)
    tmp_dict[prefix +
             "_lcsseq_norm_similarity"] = textdistance.lcsseq.normalized_similarity(str1, str2)
    tmp_dict[prefix +
             "_uni_gram_fingerprint_distance"] = n_gram_fingerprint_distance(str1, str2, n=1)
    tmp_dict[prefix +
             "_bi_gram_fingerprint_distance"] = n_gram_fingerprint_distance(str1, str2, n=2)
    return pd.DataFrame(tmp_dict, index=[0])


def compute_edit_distances(col1, col2, prefix):
    """ Function to create the edit distance measures for a given data frame. This internally uses
    compute_edit_distances function

    :type col1: np.array or List
    :param col1: First array for string comparison

    :type col2: np.array or List
    :param col2: Second array for string comparison

    :type prefix: string
    :param prefix: prefix to be added to the new columns generated

    :rtype: pd.DataFrame with new columns added giving the edit distances for each row
    """
    tmpdf = pd.DataFrame({"col1": col1,
                          "col2": col2})
    finalDf = pd.DataFrame()
    for i in tqdm(range(tmpdf.shape[0])):
        finalDf = finalDf.append(compute_edit_distance_row(str1=tmpdf['col1'][i],
                                                           str2=tmpdf['col2'][i],
                                                           prefix=prefix)
                                 )
    return finalDf


def n_gram_fingerprint_distance(str1, str2, n=1):
    """ Function to compute n-gram fingerprint distance
    :type str1: string
    :param str1: string1

    :type str2: string
    :param str2: string2

    :type n: int
    :param n: if n==1 then unigram fingerprintm; if n==2 then bigram

    :rtype: float
    """
    res = []

    for i in [str1, str2]:
        i = strip_non_alphanum(i.lower().strip()).replace(" ", "")
        if n == 2:
            bigrams = []
            for j in range(len(i)):
                try:
                    bigrams.append(i[j]+i[j+1])
                except IndexError:
                    break
            i = list(set(bigrams))
        else:
            i = list(set(list(i)))
        i.sort()
        i = "".join(i)
        res.append(i)

    return textdistance.levenshtein.normalized_similarity(res[0], res[1])


def add_features(prefixes_to_extract, df):
    """ Function to generate all the features

    :type prefixes_to_extract: list of strings
    :param prefixes_to_extract: The list should contain the prefixes of columns for which 
    feature engineering is performed. Like edit measures, embeddings, openrefine fingerprints etc.

    :type df: pd.DataFrame
    :param df: dataframe that has the columns as per the prefixes. Meaning if the prefix has "book" the dataframe should have 
    two columns, "book1" and "book2" to compare and compute the  

    :rtype: pd.DataFrame
    """
    print(
        f"Feature Generation initiated for prefix {str(prefixes_to_extract)}.")
    final_feature_df = pd.DataFrame()
    for i in prefixes_to_extract:
        print(f"Feature Generation in progress for prefix: {i}.")
        feature_df = compute_edit_distances(df[i+"1"], df[i+"2"], prefix=i)
        # TODO:
        # Add other feature engineering steps like soundex hash comparison
        # Word Embeddings in the same format as above line.
        final_feature_df = pd.concat([final_feature_df, feature_df], axis=1)

    print(f"Feature Generation Completed.")
    final_feature_df.reset_index(drop=True, inplace=True)
    return final_feature_df


if __name__ == "__main__":
    # df = read_clean_data(path_="input/train-joined-id.csv",
    #                      prefixes_to_clean=["book", "author"])
    # df_features = add_features(prefixes_to_extract=["book", "author"], df=df)
    # df = pd.concat([df, df_features], axis=1)
    # df.to_csv("output/prepared_data.csv", index=False)

    df = read_clean_data(path_="input/train-joined-id_2.csv",
                        prefixes_to_clean=["title"])
    df_features = add_features(prefixes_to_extract=["title"], df=df)
    df = pd.concat([df, df_features], axis=1)
    df.to_csv("output/prepared_data_2_1.csv", index=False)