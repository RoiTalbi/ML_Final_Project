import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def preprocess_data2(df):

    # remove the first column (sha256)
    df.drop(df.columns[0], axis=1, inplace=True)

    # remove coloumns 'A' and 'B' and 'C'
    df.drop(['A', 'B', 'C'], axis=1, inplace=True)



def preprocess_data(original_df):

    out_df = original_df.copy()

    # drop the frist column (sha256)
    out_df.drop(out_df.columns[0], axis=1, inplace=True)

    # OneHotEncoder for 'C' column and drop the original column
    encoder = OneHotEncoder()
    encoded_df = encoder.fit_transform(out_df[['C']])
    encoded_df = pd.DataFrame(encoded_df.toarray(), columns=encoder.get_feature_names())

    for col in encoded_df.columns:
        if encoded_df[col].sum() < 1000:
            encoded_df.drop(col, axis=1, inplace=True)

    print(f"added cloumns for feature C: {len(encoded_df.columns)} ")

    out_df = pd.concat([out_df, encoded_df], axis=1)
    out_df.drop('C', axis=1, inplace=True)
    out_df[encoded_df.columns] = out_df[encoded_df.columns].astype(bool)

    # OneHotEncoder for colomn 'file_type_trid' 
    encoded_df = encoder.fit_transform(out_df[['file_type_trid']])
    encoded_df = pd.DataFrame(encoded_df.toarray(), columns=encoder.get_feature_names())

    # drop the coloumns that have less than 1000 values that are not 0 
    for col in encoded_df.columns:
        if encoded_df[col].sum() < 1000:
            encoded_df.drop(col, axis=1, inplace=True)

    print(f"added cloumns for feature file_type_trid: {len(encoded_df.columns)} ")

    out_df.drop('file_type_trid', axis=1, inplace=True)
    out_df = pd.concat([out_df, encoded_df], axis=1)
    out_df[encoded_df.columns] = out_df[encoded_df.columns].astype(bool)

    # replace the file_type_trid column with the new values
    #df['file_type_trid'] = df['file_type_trid'].apply(object_to_numeric)
    #df['C'] = df['C'].apply(object_to_numeric)

    # convert all boolean columns to type bool
    boolean_columns = out_df.filter(regex='^has_').columns
    out_df[boolean_columns] = out_df[boolean_columns].astype(bool)

    # get all other columns 
    numeric_columns = out_df.columns.difference(boolean_columns)

    # fill every NaN value with the most common value in the column in the boolean columns
    for col in boolean_columns:
        most_common_value = out_df[col].mode()[0]
        out_df[col].fillna(most_common_value, inplace=True)

    # fill every NaN value with the mean value in the column
    for col in numeric_columns:
        out_df[col].fillna(out_df[col].mean(), inplace=True)

    return out_df

    