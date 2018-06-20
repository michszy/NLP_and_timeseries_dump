"""
Natural language processing
séparer le texte en  phrase et en mots
tokenisation => extraire les mots d'un document
s = text
s.plit(" ")


"""
import re
s = "Ce canton était organisé autour de Saint-Hilaire dans l'arrondissement de Limoux. Son altitude variait de 133 m (Verzeille) à 857 m (Villardebelle) pour une altitude moyenne de 282 m."


r = s.split(' ')
r = re.split("[ ]", s)
#print(r)


"""
Stemming
Couper les mots de manière bourinne en ésperant trouveer la racine
mot du 3eme groupe ---
suffixe ou prefixe +++
"""



import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


token = nltk.word_tokenize(s)
print('________')
print(token[:8])
print('________')
tagged = nltk.pos_tag(token)
print(tagged[:8])
print('________')

import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')

for token in doc[:2]:
    print(token.text)
    print(token.lemma)
    print(token.pos_)
    print(token.tag_)
    print(token.dep)
    print(token.shape)
    print(token.is_alpha)
    print(token.is_stop)
    print('____________')

print('____________')

for ent in doc.ents:
    print(ent.text)
    print( ent.start_char)
    print(ent.end_char)
    print(ent.label_)
    print('____________')

nlp = spacy.load('en_core_web_md')
tokens = nlp(u'dog cat banana')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


"""
Name entity recognotion
"""

nlp = spacy.load('en_core_web_md')
tokens = nlp(u'dog cat banana afskfsd')

"""
for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
"""

"""
tf idf
WORD2VEC
"""


"""
cos similarity
terme frequency

"""

######_____#######

