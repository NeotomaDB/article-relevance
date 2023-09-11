import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class NeotomaOneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
        self.encoder = OneHotEncoder(handle_unknown='ignore')  # Handle unknown categories
    
    def fit(self, X, y=None):
        # Fit the encoder on the specified column
        self.encoder.fit(X[self.column_name].apply(pd.Series).stack())
        return self
    
    def transform(self, X):
        # Use the fitted encoder to transform the specified column
        dummies = self.encoder.transform(X[self.column_name].apply(pd.Series).stack())
        
        # Convert to a DataFrame and set column names
        dummies_df = pd.DataFrame(dummies, columns=self.encoder.get_feature_names_out([self.column_name]))
        
        # Group by index and sum the one-hot encoded columns
        dummies_df = dummies_df.groupby(level=0).sum()
        
        # Remove the extra unnamed column (the first column)
        dummies_df = dummies_df.iloc[:, 1:]
        
        # Concatenate the one-hot encoded columns with the original DataFrame
        result_df = pd.concat([X, dummies_df], axis=1)
        
        # Drop the original column
        result_df = result_df.drop(columns=[self.column_name])
        
        return result_df
