import numpy as np
from sklearn.base import TransformerMixin

# Helpers
# ------------------------------------------------------------------------------------------------
def unsquash(X):
    ''' Transform vector of dim (n,) into (n,1) '''
    if len(X.shape) == 1 or X.shape[0] == 1:
        return np.asarray(X).reshape((len(X), 1))
    else:
        return X


def squash(X):
    ''' Transform vector of dim (n,1) into (n,) '''
    return np.squeeze(np.asarray(X))

# ------------------------------------------------------------------------------------------------
# Transformers
# ------------------------------------------------------------------------------------------------
class Transformer(TransformerMixin):
    ''' Base class for pure transformers that don't need a fit method '''

    def fit(self, X, y=None, **fit_params):
    #def fit(self, X):
        return self

    def transform(self, X, y=None, **transform_params):
    #def transform(self, X):
        return X

    def get_params(self, deep=True):
        return dict()
