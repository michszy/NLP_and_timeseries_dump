from class_tdidf import class_tfidf as cl
import pytest



def f():
    random_text = 'Hello Words Hello Hello'
    r_t_word_frequency = cl.word_frequency(random_text)
    df = cl.string_to_dataframe(r_t_word_frequency)
    r = cl.idf('Hello', df)
    return r

def test_function():
    assert f() == 0.75
