import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

GLOVE_PATH = '/Users/anastassiakornilova/FN-Votes-V2/data/glove.6B.50d.txt'


def to_index_vectors(texts, max_words=50, vectorizer=None, unique_words=True, vocab_size=20000):

    if vectorizer is None:
        # Fit vectorizer if not passed in
        vectorizer = CountVectorizer(stop_words='english', binary=True, max_features=vocab_size)
        vectorizer.fit(texts)

    id_map = vectorizer.vocabulary_

    text_vectors = []
    for text in texts:

        text = text.replace('This measure has not been amended since it was introduced.', '')

        # TEMP: use set of words to get more representative sample
        if unique_words:
            words = set(text.split())
        else:
            words = text.split()
        # If word in vectorizer, get its index
        # Offset by 1 to allow for padding
        vec = [id_map[w] + 1 for w in words if w in id_map]

        # Add padding if necessary
        if len(vec) < max_words:
            vec = vec + [0] * (max_words - len(vec))

        # Only keep max words of the words
        # TODO: pick better subset here
        text_vectors.append(vec[:max_words])
    
    #print(text_vectors[:10])
    #print(sum([len(np.array(x).nonzero()[0]) for x in text_vectors]) / len(text_vectors), len(text_vectors))
    return text_vectors, vectorizer


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
    for word in word_dict:
        if word not in included_words:
            # print word
            embeddings[word_dict[word] + 1] = np.zeros(k_word)
    print(embeddings.shape)
    return np.array(embeddings, dtype=np.float)
