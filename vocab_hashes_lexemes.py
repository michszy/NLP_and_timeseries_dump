import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab

nlp = spacy.load('en_core_web_sm')
doc = nlp(u'I love coffee a')
print(doc.vocab.strings[u'coffee'])
print(doc.vocab.strings[3197928453018144401])

for word in doc:
    lexeme = doc.vocab[word.text]
    print(lexeme.text, lexeme.orth, lexeme.shape_, lexeme.prefix,\
          lexeme.is_alpha, lexeme.is_digit, lexeme.is_title, lexeme.lang_)


empty_doc = Doc(Vocab())
#print(empty_doc.vocab.strings[3197928453018144401])
empty_doc.vocab.strings.add(u'coffee')
print("_________")
print(empty_doc.vocab.strings[3197928453018144401])
new_doc = Doc(doc.vocab)
print(new_doc.vocab.strings[3197928453018144401])

print("ok")
text = open('prince.txt', 'r').read()
#doc = nlp(text)


print("ok")
print("____")
print(doc[5:])

for word in doc[5:]:
    lexeme = doc.vocab[word.text]
    print(lexeme.text, lexeme.orth, lexeme.shape_, lexeme.prefix, \
          lexeme.is_alpha, lexeme.is_digit, lexeme.is_title, lexeme.lang_)


"""Calculer pour chacun document le tf idf"""





