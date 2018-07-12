import pandas as pd, numpy as np
from base import Transformer, squash, unsquash

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
class Select(Transformer):
    #Extract specified columns from a pandas df or numpy array
    def __init__(self, columns=0, to_np=True):
        self.columns = columns
        self.to_np = to_np
        self.res = 0

    def get_params(self, deep=True):
        return dict(columns=self.columns, to_np=self.to_np)

    def fit(self, X, y=None, **fit_params):
    #def fit(self, X):
        return self

    def transform(self, X, y=None, **transform_params):
    #def transform(self, X):
        if isinstance(X, pd.DataFrame):
            allint = isinstance(self.columns, int) or (isinstance(self.columns, list) and all([isinstance(x, int) for x in self.columns]))
            if allint:
                res = X.ix[:, self.columns]
            elif all([isinstance(x, str) for x in self.columns]):
                res = X[self.columns]
            else:
                print "Select error: mixed or wrong column type."
                res = X

            if self.to_np:
                res = unsquash(res.values)
        else:
            res = unsquash(X[:, self.columns])

        #store the resultant
        self.res = res
        return res

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
class Squash(Transformer):
    def __init__(self):
        self.res = 0
    def transform(self, X, y=None, **transform_params):
    #def transform(self, X):
        self.res = squash(X)
        return squash(X)
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
class Stringify(Transformer):
    def __init__(self):
        self.res = 0
        
    def transform(self, X, y=None, **transform_params):
    #def transform(self, X):
        self.res = X
        return np.array(map(str, X))