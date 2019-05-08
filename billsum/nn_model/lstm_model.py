from keras.layers import Input, Dense, Embedding, LSTM
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import keras.backend as K

import numpy as np
import pickle


GLOVE_PATH = '/Users/anastassiakornilova/FN-Votes-V2/data/glove.6B.50d.txt'

def parse_glove_embeddings(word_dict, k_word=50, embed_file=GLOVE_PATH):
    '''
    Map words to pre-trained embeddings. OOV words will be initialized to 0s 

    word_dict: dictionary mapping from words to indicies.
    embed_file: Link to Glove embeddings file.
    
    Returns matrix, where each row corresponds to an embedding.
    '''

    print("Parsing Embedding Matrix")
    embeddings = np.zeros((len(word_dict) + 1, k_word)) 
    included_words = {}
    with open(embed_file, 'r') as f:
        content = f.read().split('\n')
        for line in content:
            data = line.split(' ')
            if data[0] in word_dict:
                embeddings[word_dict[data[0]] + 1] = list(map(float, data[1:]))
                included_words[data[0]] = 1
    i = 0
    for word in word_dict:
        if word not in included_words:
            #print(word)
            i += 1
            embeddings[word_dict[word] + 1] = np.zeros(k_word) 
    
    print(embeddings.shape, "Missing", i)

    return np.array(embeddings, dtype=np.float)



def get_lstm_model(num_word_embed, n_word=50, k_word=50, weights=None):

	bill_input = Input(shape=(n_word,), dtype="int32", name="bill_input")

	if weights is not None:
	    bill_embed = Embedding(input_dim=num_word_embed,
	            output_dim=k_word,
	            weights=[weights],
	            trainable=True,
	            name='text_embed'
	            )(bill_input)
	else:
	    bill_embed = Embedding(input_dim=num_word_embed,
	            output_dim=k_word,
	            trainable=True,
	            embeddings_initializer='glorot_uniform',
	            name='text_embed'
	            )(bill_input)
	        
	lstm_output = LSTM(16, name='lstm')(bill_embed)

	dense_output = Dense(16, activation='relu', name='dense1')(lstm_output)

	main_output = Dense(1, activation='sigmoid', name='main_output')(dense_output)

	model = Model(inputs=[bill_input], outputs=[main_output])

	print(model.summary())

	return model


def prepare_texts(texts, n_word=50, tok=None):
	
	if tok == None:
		tok = Tokenizer()
		tok.fit_on_texts(texts)

	X = tok.texts_to_sequences(texts)

	X2 = pad_sequences(X, padding='post', value=0, maxlen=n_word)

	return X2, tok


def prep_sent(sent, lemmatize=True):
    
    final_words = []

    for word in sent:
        if lemmatize:
            final_words.append(word[2])
        else:
            final_words.append(word[0])

    return " ".join(final_words)


class LSTMWrapper:

    def __init__(self, label=('rouge-2', 'p')):

        self.model = None

        self.label = label

        self.score_threshold=0.1

        self.n_word = 50

    def train(self, train_docs):

        tdocs = [d['doc'] for d in train_docs]

        print(tdocs[0])

        all_sents = [prep_sent(s) for sents in tdocs for s in sents]

        X_train, tokenizer  = prepare_texts(all_sents)
        
        self.tokenizer = tokenizer

        print("Total Data:", len(X_train))

        #word_embeddings = parse_glove_embeddings(vectorizer.vocabulary_)
        
        self.model = get_lstm_model(len(tokenizer.word_index) + 1, n_word=self.n_word)

        print(self.model.summary())

        rtype, mtype = self.label

        y_train = np.array([y[rtype][mtype] for doc in train_docs for y in doc['scores'] ])
        
        y_train = y_train > self.score_threshold
        
        print("Baseline", y_train.mean())

        #checkpoint = ModelCheckpoint('weights.{epoch:03d}', verbose=1, save_best_only=False, mode='auto')

        self.model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])

        self.model.fit(X_train, y_train,  epochs=5)


    def score(self, test_doc):

        sents = [prep_sent(s) for s in test_doc['doc']]

        X, _ = prepare_texts(sents, n_word=self.n_word, tok=self.tokenizer)

        ys = model.predict(X)[:,0]

        return ys



if __name__ == '__main__':

	data = pickle.load(open('/Users/anastassiakornilova/BSDATA/sent_data/us_test_sent_scores.pkl', 'rb'))

	texts = []
	y = []

	for v in data.values():
	    for sent in v:
	        s = ' '.join(w[2] for w in sent[1])
	        texts.append(s)
	        y.append(sent[2]['rouge-2']['p'] > 0.1)

	XText, tokenizer = prepare_texts(texts)

	from sklearn.model_selection import train_test_split

	X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(XText, y, texts, test_size=0.1)

	model = get_lstm_model(len(tokenizer.word_index) + 1)

	model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])

	model.fit(np.array(X_train), y_train,  epochs=5, verbose=1)









