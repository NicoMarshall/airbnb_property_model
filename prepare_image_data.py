import pandas as pd
import os
import numpy as np
import cv2
import shutil


clean_df = pd.read_csv("clean_tabular_data.csv")
resize_factor = 2

def copy_resize(clean_df):
    for id in clean_df.loc[1:2,"ID"]:
        try:
            os.mkdir(f"C:\\Users\\nicom\\OneDrive\\Υπολογιστής\\airbnb_property_model\\processed_images\\{id}")
        except FileExistsError:
            pass
        src_dir = f"C:\\Users\\nicom\\OneDrive\\Υπολογιστής\\airbnb_property_model\\images\\{id}"
        dst_dir = f"C:\\Users\\nicom\\OneDrive\\Υπολογιστής\\airbnb_property_model\\processed_images\\{id}"
        try:
            file_names = [file for file in os.listdir(src_dir) if file.endswith(".png")]
            for file_name in file_names:
                try:
                    shutil.copy(os.path.join(src_dir, file_name), dst_dir)
                    os.chdir(dst_dir)
                    img = cv2.imread(file_name)
                    h,w = img.shape[:2]
                    h,w = int(h/resize_factor),int(w/resize_factor)
                    resizeImg = cv2.resize(img, (w, h))
                    img = resizeImg
                    os.chdir(f"C:\\Users\\nicom\\OneDrive\\Υπολογιστής\\airbnb_property_model")
                except FileExistsError:
                    pass
        except FileNotFoundError:
            pass
    
    
    
    
