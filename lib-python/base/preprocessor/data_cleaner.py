from base import Transformer, squash, unsquash
import pandas as pd
import re as regex


#General data cleansing class for handling all sorts of specical situations
#with data
class DataCleaner(Transformer):
    def __init__(self, columns=0, to_np=True):
        self.columns = columns
        self.to_np = to_np
        self.res = 0

    def iterate(self):
        for cleanup_method in [self.remove_urls
            , self.remove_usernames
            #, self.remove_special_chars
            , self.remove_special_chars_non_medical
            , self.remove_non_ascii
            , self.remove_na
            #,self.remove_numbers
            ,self.remove_numbers_non_medical
                               ]:
            yield cleanup_method

    @staticmethod
    def remove_by_regex(entries, regexp):
        if(entries is not None):
            entries.replace(regexp, " ", inplace=True)
        return entries

    def remove_urls(self, entries):
        if(entries is not None):
            return self.remove_by_regex(entries, regex.compile(r"http.?://[^\s]+[\s]?"))
        else:
            return entries

    def remove_na(self, entries):
        return entries.astype(object).where(pd.notnull(entries),'None')

    def remove_special_chars(self, entries):
        for remove in map(lambda r: regex.compile(regex.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$",
                                                                     "@", "%", "^", "*", "(", ")", "{", "}",
                                                                     "[", "]", "|", "/", "\\", ">", "<",
                                                                     "!", "?", ".", "'", "-",
                                                                     "--", "---", "#"]):
            if(entries is not None):
                entries.replace(remove, " ", inplace=True)
        return entries

    def remove_special_chars_non_medical(self, entries):
        for remove in map(lambda r: regex.compile(regex.escape(r)), [",", ":", "\"", ";", "$",
                                                                     "@", "\n", "\t",
                                                                     "|", "/", "\\",
                                                                     "!", "?", ".", "'"]):
            if(entries is not None):
                entries.replace(remove, " ", inplace=True)
        return entries

    def remove_non_ascii(self, entries):
        if(entries is not None):
            return self.remove_by_regex(entries, regex.compile(r"[^A-Za-z0-9\s]+"))
        else:
            return entries

    def remove_usernames(self, entries):
        if(entries is not None):
            return self.remove_by_regex(entries, regex.compile(r"@[^\s]+[\s]?"))
        else:
            return entries

    def remove_numbers(self, entries):
        if(entries is not None):
            return self.remove_by_regex(entries, regex.compile(r"\s?[0-9]+\.?[0-9]*"))
        else:
            return entries

    def remove_numbers_non_medical(self, entries):
        if(entries is not None):
            return self.remove_by_regex(entries, regex.compile(r"\s? [0-9]+\.? [0-9]*"))
        else:
            return entries

    #the cleanup method is carried out here - extending the base transformer class
    def fit(self, X, y=None, **fit_params):
    #def fit(self, X):
        return self

    #def transform(self, X):
    def transform(self, X, y=None, **transform_params):
        if isinstance(X, pd.DataFrame):
            res = X
        else:
            res = pd.DataFrame(X)
            
        for cleanup_method in self.iterate():
            res = cleanup_method(res)        
        
        if(self.to_np):
            if isinstance(X, pd.DataFrame):
                res = unsquash(res.values)
            else:
                res = unsquash(X[:, self.columns])
                
        self.res = res 
        return res