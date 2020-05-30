import pandas as pd
from flask import Flask, request
from sklearn.externals import joblib

from blocking import (create_save_annoy_index, create_tfidf_matrix,
                      prepare_data_for_classification,
                      read_clean_inference_data)
from predict import (generate_predictions, prepare_final_df_to_return,
                     reconcile_duplicates)

app = Flask(__name__)

# Loading all the models
with open("models/model.pkl", "rb") as f:
    clf_model = joblib.load(f)

with open("models/model_book.pkl", "rb") as f:
    clf_model_book = joblib.load(f)

with open("models/model_author.pkl", "rb") as f:
    clf_model_author = joblib.load(f)


@app.route("/api/v1/dedupe", methods=["POST"])
def dedupe():
    """ Method to process a post request: 
    input json should be able to be transformed into a df of 3 columns
    :rtype: json
    """
    json_ = request.get_json(force=True)
    prefixes_to_use = json_["dedupe_by"]
    input_request = pd.DataFrame(json_["data"])

    if len(prefixes_to_use) == 1:
        if prefixes_to_use[0] == "book":
            clf = clf_model_book
        elif prefixes_to_use[0] == "author":
            clf = clf_model_author
        else:
            raise KeyError
    elif len(prefixes_to_use) == 2:
        clf = clf_model

    # Implementing Blocking First, The following section is all about blocking.
    # Clean -> Create a Dense Matrix -> Annoy Index Creation -> Find Similar records for each input records -> Prepare data for classif
    df = read_clean_inference_data(path_=None,
                                   prefixes_to_clean=prefixes_to_use,
                                   app_mode=True,
                                   df=input_request)
    data_to_nearest_neighbor = create_tfidf_matrix(
        df, prefixes_to_clean=prefixes_to_use)
    sims_list = create_save_annoy_index(
        data_to_nearest_neighbor, num_of_trees=10, sim_recs_to_extract=3)
    data_for_classif = prepare_data_for_classification(
        sims_list, df, prefixes_to_use=prefixes_to_use)

    # Generate Predictions
    # Data for classif -> Generate Predictions -> Reconcile Results to create clusters -> create final df to return
    df_predicted = generate_predictions(data_for_classif,
                                        prefixes_to_use,
                                        model_path=None,
                                        app_mode=True,
                                        clf=clf)
    final_dupes = reconcile_duplicates(df_predicted, threshold=0.45)
    res_df = prepare_final_df_to_return(
        final_dupes, path_for_dedupe=input_request, app_mode=True)

    return res_df.to_json(orient="records")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8890)
