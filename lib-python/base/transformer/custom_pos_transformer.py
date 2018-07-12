import nltk
nltk.download('all')
import pandas as pd, numpy as np
from nltk.tokenize import word_tokenize
from base import Transformer, squash, unsquash
from custom_string_transformer import get_unicode, get_string

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
class POSTransformer(Transformer):
    #Extract specified columns from a pandas df or numpy array
    def __init__(self, columns=0, pos_of_interest=0, to_np=False):
        self.columns = columns
        self.to_np = to_np
        self.res = 0
        self.pos_of_interest = pos_of_interest

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        if isinstance(X, pd.DataFrame):
            if all([isinstance(x, str) for x in self.columns]):
                res = pd.DataFrame()
                res['consolidated_cleaned'] = X[self.columns].apply(lambda x: ' '.join(x), axis=1)                
                res['tokenized_consolidated_cleaned'] = res.apply(lambda row: nltk.word_tokenize(get_unicode(row['consolidated_cleaned']).lower()), axis=1)
                res['pos_tagged_sent_consolidated_cleaned'] = map(lambda sent: sent, res['tokenized_consolidated_cleaned'].apply(nltk.pos_tag))
                res['pos_tagged_sent_consolidated_cleaned_alt'] = map(lambda entry: map(lambda entry: entry[0] if(entry[1] in self.pos_of_interest) else '', entry), res['pos_tagged_sent_consolidated_cleaned'])
                res['pos_tagged_sent_consolidated_cleaned_alt'] = res['pos_tagged_sent_consolidated_cleaned_alt'].apply(lambda x: ' '.join(x))
            else:
                print "Select error: mixed or wrong column type."
                res = X

            if self.to_np:
                res = unsquash(res['pos_tagged_sent_consolidated_cleaned_alt'].values)
            else:
                res = res['pos_tagged_sent_consolidated_cleaned_alt']
        else:
            #res = unsquash(X[:, self.columns])
            res = unsquash(X[:, 'pos_tagged_sent_consolidated_cleaned_alt'])

        #store the resultant
        self.res = res
        return res