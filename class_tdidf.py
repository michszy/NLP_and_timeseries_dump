import math
import re
import os
import pandas as pd
import numpy as np
class class_tfidf():


    def __init__(self):
        pass



    def word_frequency(txt):
        document = txt.split(' ')
        r = dict()
        for i in document:
            if i not in r:
                r.update({i : int(1)})
            else:
                r[i] += int(1)
        return r


    def string_to_dataframe(txt):
        df = pd.DataFrame.from_dict(txt, orient='index')
        df.columns = ['freq']
        df = df['freq'].sort_values(ascending=False)
        words = list(df.index)
        vals = list(df.values)
        df = pd.DataFrame.from_dict(dict(zip(words, vals)), orient='index')
        df.columns = ['freq']
        df['words'] = df.index
        df.index = np.arange(0, len(df['words']), 1)
        return df


    def cosine_similarity(vector1, vector2):
        dot_product = sum(p * q for p, q in zip(vector1, vector2))
        magnitude = math.sqrt(sum([val ** 2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
        return dot_product / magnitude

    def idf(word, dataset):
        found_word = dataset[dataset['words'].str.match(word)]
        found_word = found_word['freq'].iloc[0]
        sum_of_the_words = dataset['freq'].sum()
        result = found_word / sum_of_the_words
        return result


    def sample_to_dataframe(self, sample, token=False):
        if type(sample) == type(' '):
            splited_sample = sample.split(' ')
            print('The sample is a string')
            df = self.string_to_dataframe(splited_sample)
        elif type(sample) == type(os.path.normpath(sample)):
            with open(sample, 'r') as file:
                raw_text = file.read().replace('\n', '')
                df = self.string_to_dataframe(raw_text)
        elif type(sample) == type(dict()):
            df = sample
            print('The sample is a dataframe')
        else:
            print('Lose')
        print(type(sample))
        print (df)
