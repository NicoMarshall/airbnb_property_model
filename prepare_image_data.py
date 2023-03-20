import pandas as pd
import os
import numpy as np
import cv2
import shutil


clean_df = pd.read_csv("clean_tabular_data.csv")

def copy_resize(df:pd.DataFrame):
    min_height = 155
    for id in df.loc[1:20,"ID"]:
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
                    if h < min_height:
                        min_height = h
                    else:
                        pass    
                    resize_factor = min_height/h
                    h,w = int(h*resize_factor),int(w*resize_factor)
                    img = cv2.resize(img, (w, h))
                    print(h,w)
                    os.chdir(f"C:\\Users\\nicom\\OneDrive\\Υπολογιστής\\airbnb_property_model")
                except FileExistsError:
                    pass
        except FileNotFoundError:
            pass
    
    
if __name__ =="__main__":
    copy_resize(clean_df)
    