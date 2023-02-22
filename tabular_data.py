import pandas as pd
import numpy as np

tab_df = pd.read_csv("listing.csv")

def remove_rows_with_missing_ratings(df: pd.DataFrame):
    df.dropna(subset=["Location_rating","Cleanliness_rating","Accuracy_rating","Value_rating","Check-in_rating","Communication_rating"], inplace=True)
    return df

