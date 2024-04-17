import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class NeotomaOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, min_count=3):
        self.min_count = min_count
        self.categories = {}
        self.removed_rows = []

    def fit(self, X, y=None):
        # Consider subjects which apperance > min_count
        for col in X.columns:
            value_counts = X[col].apply(pd.Series).stack().value_counts()
            self.categories[col] = value_counts[value_counts >= self.min_count].index.tolist()
        return self

    def transform(self, X):
        transformed_dfs = []
        for col in self.categories:
            categories = self.categories.get(col, [])

            # Ensure that all categories from the fit phase are present in the transform phase
            for category in self.categories[col]:
                if category not in X[col].apply(pd.Series).stack().unique():
                    X[category] = 0

            # Handling empty lists
            X.loc[:, col] = X[col].apply(lambda x: ['None'] if len(x) == 0 else x)

            # Leave the categories that exist and convert them to a stack
            transformed_df = pd.get_dummies(X[col].apply(pd.Series).stack())

            # If there is a category (say 'math') that has created a new column in 
            # self.encoder but that 'math' does not exist in the new df, build it and
            # encode with 0s
            for category in categories:
                if category not in transformed_df.columns:
                    transformed_df[category] = 0

            transformed_df = transformed_df.groupby(level=0).max().fillna(0)

            # Change T/F to 0/1
            transformed_df = transformed_df.astype(int)
            transformed_df = transformed_df[categories]
            transformed_dfs.append(transformed_df)

            # Keep track of removed rows for debugging
            removed_indices = set(X.index) - set(transformed_df.index)
            self.removed_rows.extend(list(removed_indices))
        
        result = pd.concat(transformed_dfs, axis=1)
        
        # Drop column where there were no subjects from a query
        if 'None' in result.columns:
            result = result.drop(columns=['None'])
        
        return result