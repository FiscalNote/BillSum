"""
Bill features related to "similarity" between different aspects of the bill.
"""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

from fn_tldr.summarize.features.basic_features import GenericFeature
from fn_tldr.summarize.features.text_transformers import SpacyTfidfWrapper
from fn_tldr.utils.matrix_utils import cosine_sim_sparse

class SimWithFirstF(GenericFeature):
    """
    Calculate cosine similarity between the sentence and the first sentence 
        in the bill. Uses text vectors computer that are passed in, so is 
        independent of a specific representation
    """
    def __init__(self):
        self.is_sparse = False

        self.text_transformer = SpacyTfidfWrapper(tfidf_args={'use_idf': False, 'min_df': 1, 'max_df': 1.0, 'binary': True})

    def prepare_doc(self, doc, *args, **kwargs):
        # Fit each sentence as a document
        self.text_transformer.fit([doc], sent_as_doc=True)
        self.sent_vecs = self.text_transformer.transform_by_sent([doc])       

        # Find first full sentence 
        i = 0
        for sent in doc.sents:
            text = ''.join(s.text for s in sent)
            if 'SECTION-HEADER' in text:
                i += 1
                continue
            break
        self.first_sentence = i

    def make_features(self, i, sent):
        # Compute or get-precomputed vectors
        v1 = self.sent_vecs[self.first_sentence]
        v2 = self.sent_vecs[i]
        return cosine_similarity(v1,v2)[0]

class SimWithTitletF(GenericFeature):

    def __init__(self):
        self.is_sparse = False

        self.text_transformer = SpacyTfidfWrapper(tfidf_args={'use_idf': False, 'min_df': 1, 'max_df': 1.0})

    def prepare_doc(self, doc, title='', **kwargs):
        # Fit each sentence as a document add in title to the mock document 
        # We want to fit both ways to know what we miss 

        self.text_transformer.fit([doc, title], sent_as_doc=True)
        self.sent_vecs = self.text_transformer.transform_by_sent([doc])    

        # Double wrap because I need to clean up    
        self.title_vec = self.text_transformer.transform([title])   

    def make_features(self, i, sent):
        # Compute or get-precomputed vectors
        v1 = self.title_vec
        v2 = self.sent_vecs[i]
        return cosine_similarity(v1,v2)[0]