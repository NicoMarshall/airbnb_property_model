# airbnb_property_model
Data science project to create and evauluate different machine learning models that predict nightly prices of airbnb property listings. Written in Python, using pandas and scikit learn for data processing and analysis.

# Milestone 1:
The first stage was cleaning and pre-processing the raw data to get it into the right format for ML. The raw data in csv format was first loaded into a pandas dataframe. Then a script was written inside tabular_data.py to remove rows where ratings were missing, format description strings, set default values and split data into features, labels tuples. Image data resized inside prepare_image_data.py, so that all images have the same height.
