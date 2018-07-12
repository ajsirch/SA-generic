from base import Transformer, squash, unsquash
from pandas.io.json import json_normalize
import json
from jq import jq
import requests as req
import pandas as pd

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
class SciBiteTransformer(Transformer):
    #Extract specified columns from a pandas df or numpy array
    def __init__(self, columns=0, ner_entity_interest=0, ner_terms_interest=0, ner_extract_terms_interest=0, root_url=0):
        self.columns = columns
        self.res = 0
        self.ner_entity_interest = ner_entity_interest
        self.ner_terms_interest = ner_terms_interest
        self.ner_extract_terms_interest = ner_extract_terms_interest
        self.root_url = root_url
        
    # -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
    #Create a request
    def request_fn(self, entry):
        resp = req.post(self.root_url, {"text": entry, "format":"txt", "output":"json"})
        return resp.json()['RESP_PAYLOAD']
    # -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
    def convert_unicode(self, entry):
        return json.dumps(entry)
    # -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE        
    #Parse the json
    def parseReponse(self, entry):    
        return jq(".[] | .[] | {entityType, name, frag_vector_array, realSynList}").transform(text=entry, multiple_output=True)

    # -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
    def ner_aggregate(self, entry, colname):
        #print 'Col name : ', colname
        #print 'Entry : ', entry, '\n\n'
        agg_entry = {}
        for x in self.ner_terms_interest:
            agg_entry[x] = ','.join(map(lambda z: ','.join(z[x]) if isinstance(z[x], list) and z['entityType']==colname and z[x] is not '' else ''
                                        + ''.join(z[x]) if not isinstance(z[x], list) and z['entityType']==colname and z[x] is not '' else '', entry))
            agg_entry[x] = ' '.join(map(lambda x: x if x is not '' else '', set(list(agg_entry[x].split(',')))))
        return agg_entry

        def fit(self, X, y=None, **fit_params):
            return self

    def transform(self, X, y=None, **transform_params):
        if isinstance(X, pd.DataFrame):
            if all([isinstance(x, str) for x in self.columns]):
                res = pd.DataFrame()
                if len(self.columns) > 1:
                    res['consolidated_cleaned'] = X[self.columns].apply(lambda x: ' '.join(x), axis=1)
                else:
                    res['consolidated_cleaned'] = X[self.columns]
                print 'Execute SciBite calls'
                res['consolidated_ner'] = res['consolidated_cleaned'].apply(self.request_fn)
                res['consolidated_ner'] = res['consolidated_ner'].apply(self.convert_unicode)
                print 'Execute response parsing after NER'
                res['consolidated_ner'] = res['consolidated_ner'].apply(self.parseReponse)
                
                ner_extract = map(lambda x: map(lambda y: self.ner_aggregate(x, y), self.ner_entity_interest), res['consolidated_ner'])
                for z in self.ner_entity_interest:
                    res[z] = list(map(lambda d: d.replace('|', ' ').strip().lower(), map(lambda a: '| '.join(map(lambda b: b['name'] + ' ' + b['realSynList'] if b['entityType'].strip()==z else '', a)), ner_extract)))

            else:
                print "Select error: mixed or wrong column type."
                res = X
        else:
            res = unsquash(X[:, 'pos_tagged_sent_consolidated_cleaned_alt'])

        #store the resultant
        self.res = res
        #print res
        return res