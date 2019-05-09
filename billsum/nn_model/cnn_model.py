from billsum.cnn_model.word_embeddings import to_index_vectors, parse_glove_embeddings

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras.layers import Embedding, Input, SpatialDropout1D, Reshape, Dropout, Dense, Flatten, MaxPooling2D, Conv2D, Concatenate
from keras.models import Sequential, Model

from keras.optimizers import Adamax


import numpy as np


def mask_aware_mean(x):
    # Special method to only average non-zero embeddings and avoid padding
    mask = K.not_equal(K.sum(K.abs(x), axis=2, keepdims=True), 0)

    # number of that rows are not all zeros
    n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=False)

    # compute mask-aware mean of x
    x_mean = K.sum(x, axis=1, keepdims=False) / n

    return x_mean

def get_cnn_model(num_word_embed=5001, k_word=50, n_word=50,  
                             weights=None, dropout_bill=0.3, dropout_cnn=0.3,
                             layer_prefix='text', filter_size=4, num_maps=600,
                             **kwargs):
    '''
    Simple bill embedding that uses an CNN. 
    Returns the final layer's output in this sequence
    
    bill_input: An input/previous layer for text.

    vocab_size: Number of distinct possible words in input
    k_word: dimension of word embeddings
    n_word: length of input bills in words 
    weights: pre-defined weights for embedding layer 
    layer_prefix: prefix to use for all bill layers (relevant when many texts present)
    filter_size: size of the cnn filter 
    num_maps: number of filters to use
    dropout_bill: Dropout to use after bill later
    '''

    bill_input = Input(shape=(n_word,), dtype="int32", name="bill_input")

    if weights is not None:
        bill_embed = Embedding(input_dim=weights.shape[0],
                output_dim=k_word,
                input_length=n_word,
                weights=[weights],
                trainable=True,
                name='text_embed'
                )(bill_input)
    else:
        bill_embed = Embedding(input_dim=num_word_embed,
                output_dim=k_word,
                input_length=n_word,
                trainable=True,
                embeddings_initializer='glorot_uniform',
                name='text_embed'
                )(bill_input)

    bill_embed = SpatialDropout1D(dropout_bill, name='bill_dropout')(bill_embed)
    # # Add additional dimensions (needed to work with cnn)
    bill_embed = Reshape((n_word, k_word, 1))(bill_embed)

    # # The CNN Magic
    cnn_bill_embed = Conv2D(5, (3, k_word),  padding='valid', 
                            activation='relu', kernel_initializer='normal',
                            name='text_cnn3')(bill_embed)


    cnn_bill_embed = MaxPooling2D((n_word - 3 + 1, 1), 
                                    name='text_cnn_maxpool3')(cnn_bill_embed)


    cnn_bill_embed2 = Conv2D(5, (2, k_word),  padding='valid', 
                            activation='relu', kernel_initializer='normal',
                            name='text_cnn2')(bill_embed)


    cnn_bill_embed2 = MaxPooling2D((n_word - 2 + 1, 1), 
                                    name='text_cnn_maxpool2')(cnn_bill_embed2)

    final_bill_embed = Flatten()(cnn_bill_embed)
    final_bill_embed2 = Flatten()(cnn_bill_embed2)

    final_final = Concatenate()([final_bill_embed, final_bill_embed2])

    #final_bill_embed = Dropout(final_bill_embed, name='text_cnn_dropout')(final_bill_embed)

    main_output = Dense(1, activation='softmax', name='main_output')(final_final)

    model = Model(inputs=[bill_input], outputs=[main_output])

    return model

def prep_sent(sent, lemmatize=True):
    
    final_words = []
    for word in sent:
        if lemmatize:
            final_words.append(word[2])
        else:
            final_words.append(word[0])

    return " ".join(final_words)

class CnnWrapper:

    def __init__(self,score=('rouge-2', 'p')):

        self.model = None

        self.score = score

        self.score_threshold=0.1

        self.n_word = 30

    def train(self, train_docs):
        print(train_docs[0])
        all_sents = [prep_sent(s[1]) for sents in train_docs for s in sents]

        X_train, vectorizer = to_index_vectors(all_sents, max_words=self.n_word)

        X_train = np.array(X_train)

        print("Total Data:", len(X_train))

        self.vectorizer = vectorizer

        word_embeddings = parse_glove_embeddings(vectorizer.vocabulary_)
        
        self.model = get_cnn_model(weights=word_embeddings, n_word=self.n_word)

        print(self.model.summary())

        rtype, mtype = self.score

        y_train = np.array([y[2][rtype][mtype] for sents in train_docs for y in sents ])
        y_train = y_train > self.score_threshold
        
        print("Baseline", y_train.mean())

        checkpoint = ModelCheckpoint('weights.{epoch:03d}', verbose=1, save_best_only=False, mode='auto')

        self.model.compile(optimizer=Adamax(lr=0.1), loss='binary_crossentropy', metrics=['accuracy'])

        self.model.fit(X_train, y_train, callbacks=[checkpoint], epochs=5)


    def score(self, test_doc):

        sents = [prep_sent(s) for s in test_doc['doc']]

        X = to_index_vectors(sents, vectorizer=self.vectorizer)



if __name__ == '__main__':

    import pickle

    data = pickle.load(open('/Users/anastassiakornilova/BSDATA/sent_data/us_test_sent_scores.pkl', 'rb'))

    model = CnnWrapper()

    model.train(list(data.values()))





