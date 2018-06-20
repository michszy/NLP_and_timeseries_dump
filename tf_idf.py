from __future__ import division
import numpy as np
import string
import math
import matplotlib.pyplot as plt
import seaborn as sns
import operator
import pandas as pd

def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)

def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val ** 2 for val in vec1]))
    return dot_product / magnitude

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    r = len(intersection)/ len(union)
    return r

def word_frequency(document):
    document = document.split(" ")
    r = dict()
    for i in document:
        if i not in r:
            r.update({i : int(1)})
        else:
            r[i] += int(1)
#    print(r)
    return r

def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)

def sublinear_term_frequency(term, tokenized_document):
    return  1 + math.log(tokenized_document.count(term))

def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document)) / max_count))




tokenize = lambda doc: doc.lower().split(" ")


document_0 = "China has China a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_6 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"


all_documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]
tokenized_documents = [tokenize(d) for d in all_documents]
all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])

"""
print(all_tokens_set)
print(tokenized_documents)
print("always" in all_tokens_set)
"""




word_frequency(document_3)

jaccard_similarity(document_0, document_0)

vec1 = [1,2,3]
vec2 = [1,4,6]
vec3 = [2,18,30]
cosine_similarity(vec1,vec2)

r = set(tokenized_documents[2]).intersection(set(tokenized_documents[4]))
print(r)




print(term_frequency("China", document_0))
print("sublinear_term_frequency " + str(sublinear_term_frequency('China', document_0)))
print("augmented_term_frequency " + str(augmented_term_frequency("China", document_0)))

print('_________')

with open('prince.txt', 'r') as file:
    data = file.read().replace('\n', '')


prince_data_freq = word_frequency(data)


#creating the dataframe
df = pd.DataFrame.from_dict(prince_data_freq, orient='index')



#while the words are the index, we create a frequencies columns
df.columns = ['freq']



#the freq are sorted from the higtest to the lowest
df= df['freq'].sort_values(ascending=False)



words = list(df.index)
vals = list(df.values)
df = pd.DataFrame.from_dict(dict(zip(words, vals)), orient='index')
df.columns = ['freq']
df['words'] = df.index
df.index = np.arange(0, len(df['words']), 1)

print(df.head())
plt.figure(figsize=(15,6))
#plt.bar(df['words'][:50], df['freq'][:50])
#plt.show()

def idf(word,dataset):
    found_word = dataset[dataset['words'].str.match(word)]
    found_word = found_word['freq'].iloc[0]
    sum_of_the_words = dataset['freq'].sum()
    result = found_word / sum_of_the_words
    return result

print(idf('de', df))











