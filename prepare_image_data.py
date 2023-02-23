import pandas as pd
import os
import numpy as np
import cv2
import glob
import shutil

clean_df = pd.read_csv("clean_tabular_data.csv")
for id in clean_df.loc[1:3,"ID"]:
    try:
        os.mkdir(f"C:\\Users\\nicom\\OneDrive\\Υπολογιστής\\airbnb_property_model\\processed_images\\{id}")
    except FileExistsError:
        pass
    src_dir = f"C:\\Users\\nicom\\OneDrive\\Υπολογιστής\\airbnb_property_model\\images\\{id}"
    dst_dir = f"C:\\Users\\nicom\\OneDrive\\Υπολογιστής\\airbnb_property_model\\processed_images\\{id}"
    #print(src_dir)
    try:
        file_names = [file for file in os.listdir(src_dir) if file.endswith(".png")]
        for file_name in file_names:
            try:
                shutil.copy(os.path.join(src_dir, file_name), dst_dir)
            except FileExistsError:
                pass
    except FileNotFoundError:
        pass
    
    
    
    
