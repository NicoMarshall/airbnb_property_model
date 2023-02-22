import pandas as pd
import numpy as np
import ast
tab_df = pd.read_csv("listing.csv")

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
   

