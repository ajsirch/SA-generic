# -*- coding: utf-8 -*-
#These methods are explained : http://www.carlosble.com/2010/12/understanding-python-and-unicode/
#https://nedbatchelder.com/text/unipain.html
    
def __if_number_get_string(number):
    converted_str = number
    if isinstance(number, int) or \
       isinstance(number, float):
        converted_str = str(number)
    return converted_str
 
def get_unicode(strOrUnicode, encoding='utf-8'):
    strOrUnicode = __if_number_get_string(strOrUnicode)
    if isinstance(strOrUnicode, unicode):
        return strOrUnicode
    return unicode(strOrUnicode, encoding, errors='ignore')
 
def get_string(strOrUnicode, encoding='utf-8'):
    strOrUnicode = __if_number_get_string(strOrUnicode)
    if isinstance(strOrUnicode, unicode):
        return strOrUnicode.encode(encoding)
    return strOrUnicode