import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import sys
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.compose import ColumnTransformer


# decorator for printing original and new data frames
def print_original_and_new_dataframes(func):
    def wrapper(*args, **kwargs):
        print("Original data frame:")
        print(args[0].head())

        new_df = func(*args, **kwargs)

        print(".\n" * 3)
        print("Preprocessed data frame:")
        print(new_df.head())

        return new_df
    return wrapper




def preprocess_categorial_features(original_df, threshold_to_drop=50):

    PREFIX = "OHE"
    # create an instance of OneHotEncoder
    encoder = OneHotEncoder()

    # create a column transformer object.
    # Columns: 'C' , 'file_type_trid' are categorial
    ct = ColumnTransformer(
        transformers=[
            (PREFIX, encoder, [col for col in original_df.columns if original_df[col].dtype == 'object'])
        ],
        remainder='drop'
        #remainder='passthrough'
    )

    # fit and transform the data
    encoded_data = ct.fit_transform(original_df)

    # convert the sparse matrix to a dense array
    dense_array = encoded_data.toarray()

    # create a new DataFrame with the encoded data
    encoded_df = pd.DataFrame(dense_array, columns=ct.get_feature_names_out())

    # now drop all the columns that have less than 'threshold_to_drop' values of that category
    for col in encoded_df.columns:
        if encoded_df[col].sum() < threshold_to_drop:
            encoded_df.drop(col, axis=1, inplace=True)
    
    # rename all columns that their name starts with 'remainder' to their original name
    # encoded_df.rename(columns={col: col.split("_")[1] for col in encoded_df.columns if col.startswith("remainder")}, inplace=True)

    # convert all boolean columns to type bool
    added_columns = encoded_df.filter(regex=f'^{PREFIX}').columns
    encoded_df[added_columns] = encoded_df[added_columns].astype(bool)

    # drop object column in original df and contact the encoded data with the original data
    original_df.drop(columns=[col for col in original_df.columns if original_df[col].dtype == 'object'], inplace=True)
    encoded_df = pd.concat([original_df, encoded_df], axis=1)

    return encoded_df



def remove_outliers(original_df, threshold=3):

    df = original_df.copy()

    # for each numeric column, replace outlier's values with median
    # print the number of rows removed
    for column in df.select_dtypes(include=np.number).columns:
        z_scores = np.abs(stats.zscore(df[column]))
        
        # Count the number of outliers replaced
        num_outliers_replaced = np.sum(z_scores > threshold)
        
        # Replace outliers with the median value of the column
        df[column] = np.where(z_scores > threshold, df[column].median(), df[column])
        
        # Print the number of outliers replaced for the current column
        print(f"Number of outliers replaced in column '{column}': {num_outliers_replaced}")
    
    return df



def fill_missing_categorial_features(df):
    # fill with most common value the missing values in the 'C' column
    most_common_value = df['C'].mode()[0]
    df['C'].fillna(most_common_value, inplace=True)

    # fill with most common value the missing values in the 'file_type_trid' column
    most_common_value = df['file_type_trid'].mode()[0]
    df['file_type_trid'].fillna(most_common_value, inplace=True)


# @print_original_and_new_dataframes
def preprocess_data(original_df):

    df = original_df.copy()

    # drop the frist column (sha256)
    df.drop(df.columns[0], axis=1, inplace=True)

    fill_missing_categorial_features(df)

    # Preprocess the 'file_type_trid' column using the 'file_type_prob_trid' column 
    # for every value in file_type_prob_trid column, if it's less than 30.0 change the matching value in file_type_trid to the most common value
    most_common_file_type_trid = df['file_type_trid'].mode()[0]
    df.loc[df['file_type_prob_trid'] < 30.0, 'file_type_trid'] = most_common_file_type_trid
    df.drop('file_type_prob_trid', axis=1, inplace=True)
    
    # preprocess categorial features
    df = preprocess_categorial_features(df)

    # convert all boolean columns to type bool
    boolean_columns = df.filter(regex='^has_').columns
    df[boolean_columns] = df[boolean_columns].astype(bool)

    # fill every NaN value with the most common value in the column in the boolean columns
    # and with the mean in the other (numeric) columns
    for col in df.columns:
        # get column's type 
        col_type = df[col].dtype

        if col_type == bool:
            most_common_value = df[col].mode()[0]
            df[col].fillna(most_common_value, inplace=True)

        else:
            df[col].fillna(df[col].median(), inplace=True)


    # remove outliers
    df = remove_outliers(df)

    # print the number of missing values
    print(f"Number of missing values: {df.isnull().sum().sum()}")

    return df
    