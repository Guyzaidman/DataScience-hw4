from sklearn.naive_bayes import CategoricalNB

class NaiveBayes:
    def __init__(self, train_df):
        X, y = train_df.drop(columns=['class']), train_df['class']
        self.model = CategoricalNB(alpha=2)
        self.model.fit(X,y)

    def predict(self, test_df):
        test_df.drop(columns=['class'], inplace=True)
        pred = self.model.predict(test_df)
        return pred