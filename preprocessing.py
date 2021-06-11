import numpy as np
import pandas as pd
import statistics
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, LabelEncoder

class PreProcessing:
    def __init__(self, num_of_bins, struct):
        if num_of_bins < 2:
            raise Exception('Number of bins should be larger or equal to 2.')
        self.num_of_bins = num_of_bins
        self.struct = struct
        self.le = LabelEncoder()
        self.oe = OrdinalEncoder()
    def fill_missing(self, df):
        dfs = []
        for v in self.struct['class']:
            dfs.append(df.loc[df['class'] == v])

        for d in dfs:
            for attr in df:
                if d[attr].isnull().values.any():
                    if self.struct[attr] == 'NUMERIC':
                        d[attr].fillna(d[attr].mean(), inplace=True)
                    else:
                        d[attr].fillna(statistics.mode(df[attr]), inplace=True)

        final_df = pd.concat(dfs)
        return final_df


    def discretize(self, df):
        numeric_feature_names = []
        for k in self.struct:
            if self.struct[k] == 'NUMERIC':
                numeric_feature_names.append(k)

        numeric_df = df.loc[:,numeric_feature_names]
        est = KBinsDiscretizer(n_bins=self.num_of_bins, encode='ordinal', strategy='uniform')
        # est.fit(numeric_df)
        disc_bin_df = pd.DataFrame(est.fit_transform(numeric_df))
        disc_bin_df.columns = numeric_df.columns
        disc_bin_df.index = numeric_df.index

        other_df = df.drop(numeric_feature_names, axis=1)

        # oe = OrdinalEncoder()
        # le = LabelEncoder()
        target = pd.DataFrame(other_df['class'])
        other_df.drop(columns=['class'], inplace=True)
        labeled_cat_df = pd.DataFrame(self.oe.fit_transform(other_df))
        labeled_target = pd.DataFrame(self.le.fit_transform(target['class']))

        labeled_cat_df.columns = other_df.columns
        labeled_cat_df.index = other_df.index
        labeled_target.columns = target.columns
        labeled_target.index = target.index
        other_df = labeled_cat_df.merge(labeled_target, left_index=True, right_index=True)

        final_df = other_df.merge(disc_bin_df, left_index=True, right_index=True)

        return final_df

    def inverse_pred(self, predictions):
        return self.le.inverse_transform(predictions)
