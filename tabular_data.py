import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def remove_rows_with_missing_ratings(df: pd.DataFrame):
    df.dropna(subset=["Location_rating","Cleanliness_rating","Accuracy_rating","Value_rating","Check-in_rating","Communication_rating","bedrooms"], inplace=True)
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
    df.loc[:,["bedrooms","bathrooms","beds","guests"]] =  df.loc[:,["bedrooms","bathrooms","beds","guests"]].replace({np.nan:int(1),r'Somerford Keynes England United Kingdom':int(1)})
    df[["bedrooms","guests"]] = df[["bedrooms","guests"]].apply(pd.to_numeric,errors="coerce")
    return df
    
def clean_tabular_data(data):
    tab_df = pd.read_csv(data)
    combine_description_strings(tab_df)
    set__default_feature_values(tab_df)
    remove_rows_with_missing_ratings(tab_df)
    return tab_df

def load_airbnb(data, label: str):
    tab_df = pd.read_csv(data)
    labels = tab_df[label]
    tab_df = tab_df.drop(label, axis = 1)
    text_cols = tab_df.select_dtypes(include = object)
    tab_df = tab_df.drop(text_cols,axis=1).drop('Unnamed: 0',axis=1).drop('Unnamed: 19',axis=1)
    
    return tab_df, labels
   
def plot_histograms(data: pd.DataFrame):
    fig, ax = plt.subplots(nrows=3, ncols=2)
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 6))  # define the figure and subplots
    axes = axes.ravel()  # array to 1D
    cols = f.columns
    colors = ['tab:blue', 'tab:orange', 'tab:green']*4  # list of colors for each subplot, otherwise all subplots will be one color

    for col, color, ax in zip(cols, colors, axes):
        f[col].plot(kind='hist', ax=ax, color=color, label=col, title=col)
        #ax.legend()    
    fig.delaxes(axes[11])  # delete the empty subplot
    fig.tight_layout()
    plt.show()
        
    

if __name__ == "__main__":
    clean_tabular_data("listing.csv").to_csv("clean_tabular_data.csv")
    features, labels = load_airbnb("clean_tabular_data.csv", "Price_Night")
    plot_histograms(features)
   
    