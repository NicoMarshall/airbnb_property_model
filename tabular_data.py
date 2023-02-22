import pandas as pd
import numpy as np


def remove_rows_with_missing_ratings(df: pd.DataFrame):
    df.dropna(subset=["Location_rating","Cleanliness_rating","Accuracy_rating","Value_rating","Check-in_rating","Communication_rating"], inplace=True)
    return df

def combine_description_strings( df: pd.DataFrame):
    df.dropna(subset=["Description"],inplace=True)
    df["Description"] = df["Description"].apply(lambda x:list(map(str, x[1:-1].split(',')))) 
    df["Description"] = df["Description"].apply(lambda x:[n.strip() for n in x])
    df["Description"] = df["Description"].apply(lambda x: ''.join(x)) 
    df["Description"] = df["Description"].str.replace("About this space",'')
    df["Description"] = df["Description"].map(lambda x: x.strip("'''")) 
    return df
   
def set__default_feature_values(df:pd.DataFrame):
    df.loc[:,["bedrooms","bathrooms","beds","guests"]] =  df.loc[:,["bedrooms","bathrooms","beds","guests"]].replace({np.nan:int(1)})
    return df
    
def clean_tabular_data(data):
    tab_df = pd.read_csv(data)
    remove_rows_with_missing_ratings(tab_df)
    combine_description_strings(tab_df)
    set__default_feature_values(tab_df)
    return tab_df

if __name__ == "__main__":
    clean_tabular_data("listing.csv").to_csv("clean_tabular_data.csv")
    