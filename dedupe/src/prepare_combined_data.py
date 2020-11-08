import pandas as pd

mapping = pd.read_csv(r"C:\Users\phani\Downloads\Amazon-GoogleProducts\Amzon_GoogleProducts_perfectMapping.csv")
googledata = pd.read_csv(r"C:\Users\phani\Downloads\Amazon-GoogleProducts\GoogleProducts.csv", encoding = "ISO-8859-1")
amazondata = pd.read_csv(r"C:\Users\phani\Downloads\Amazon-GoogleProducts\Amazon.csv", encoding = "ISO-8859-1")


googledata.rename({"name":"title"}, axis=1, inplace=True)

amazondata.rename({"id":"id1","title":"title1", "manufacturer":"manufacturer1","description":"description1", "price":"price1"}, axis=1, inplace=True)
googledata.rename({"id":"id2","title":"title2", "manufacturer":"manufacturer2","description":"description2", "price":"price2"}, axis=1, inplace=True)

amazondata["key_1"] = 1
googledata["key_2"] = 1

merged_data = amazondata.merge(googledata, left_on="key_1",right_on="key_2" )
merged_data.drop(["key_1", "key_2"], axis=1, inplace=True)
merged_data["id"] = merged_data.apply(lambda x: x["id1"]+str(abs(hash(x["id2"]))), axis=1)

mapping["id"] = mapping.apply(lambda x: x["idAmazon"]+str(abs(hash(x["idGoogleBase"]))), axis=1)
mapping["label"] = 1
merged_data = merged_data.merge(mapping[["id", "label"]], on="id", how="left")
merged_data.label.fillna(0, inplace=True)
merged_data.label.value_counts()

del mapping, amazondata, googledata

merged_data_1 = merged_data.loc[merged_data.label==1]
merged_data_0 = merged_data.loc[merged_data.label==0]
merged_data_0 = merged_data_0.sample(n=merged_data_1.shape[0]*50, replace=True, random_state=42)
merged_data = merged_data_1.append(merged_data_0)
merged_data = merged_data.sample(frac=1, random_state=100)
merged_data.reset_index(drop=True, inplace=True)
merged_data.to_csv("input/train-joined-id_2.csv", index=False)