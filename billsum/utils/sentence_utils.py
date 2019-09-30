'''
We store sentence data in a custom format, because SPACY
objects take up too much space.

This utility helps handle them
'''
from collections import namedtuple

# The sentence object - set up to match spacy syntex
Word = namedtuple('Word', ['text', 'i', 'lemma_', 'ent_type_',
    'ent_iob_', 'pos_', 'dep_', 'head'])

class Sent:

    def __init__(self, words):
        self.words = words
        self.text = ' '.join(w.text for w in words)

    def __iter__(self):
        return iter(self.words)

class Doc:

    def __init__(self, sents):
        self.sents = sents

    def __iter__(self):
        return iter(self.sents)

def spacy_to_tuple(doc):
    text_feats = [(w.string, w.i, w.lemma_, w.ent_type_, w.ent_iob_, w.pos_, w.dep_, w.head.i)
                                    for w in doc]
    return text_feats


def list_to_doc(input_sents):
    '''
    Takes in a list of sentence data and wraps everything in the classes

    Input: [('Expressing ', 0, 'express', '', 'O', 'VERB', 'ROOT', 0),..]

    Output:
        Doc of Sents of Words 
        Doc(Sents(Words(text=expressing, i=0..)))))
    '''

    all_sents = []

    for sent_data in input_sents:
        all_words = []

        for word_data in sent_data:
            w = Word(*word_data)
            all_words.append(w)

        all_sents.append(Sent(all_words))
    return Doc(all_sents)


    