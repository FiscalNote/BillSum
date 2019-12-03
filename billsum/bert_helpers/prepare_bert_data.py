'''
Take in data from cleaned sentence files and write them into simple tsvs for BERT
'''
import os
import pandas as pd 
import pickle


prefix = os.environ['BILLSUM_PREFIX']

if not prefix.endswith('/'):
    prefix += '/'

# Create additional data dirs
if not os.path.isdir(prefix + 'bert_data'):
	os.mkdir(prefix + 'bert_data')

#### Process training data ####
sent_data = pickle.load(open(prefix + 'sent_data/us_train_sent_scores.pkl', 'rb'))

# Establish an ordering for scoring after BERT runs
doc_order = sorted(sent_data.keys())

to_save = []
for key in doc_order:
	doc = sent_data[key]
	for sent in doc:
		y = int(sent[2]['rouge-2']['p'] > 0.1) # Our label 
		to_save.append([sent[0], y])

# Save in tsv format
final = pd.DataFrame(to_save)
final.to_csv(sep='\t', index=None)

# Save in tsv format
final = pd.DataFrame(to_save)
final.to_csv(prefix + 'bert_data/train.tsv', sep='\t', index=None)

# Store just the sentences for pretraining
#final[0].to_csv(prefix + 'bert_data/all_texts_us_train.txt', index=None)


###### Repeat for both test files ######

sent_data = pickle.load(open(prefix + 'sent_data/us_test_sent_scores.pkl', 'rb'))

doc_order = sorted(sent_data.keys())

to_save = []
for key in doc_order:
	doc = sent_data[key]
	for sent in doc:
		y = int(sent[2]['rouge-2']['p'] > 0.1) # Our label 
		to_save.append([sent[0], y])

# Save in tsv format
final = pd.DataFrame(to_save)
final.to_csv(prefix + 'bert_data/test.tsv', sep='\t', index=None)


# CA
sent_data = pickle.load(open(prefix + 'sent_data/ca_test_sent_scores.pkl', 'rb'))

# Establish an ordering for scoring after BERT runs
doc_order = sorted(sent_data.keys())

to_save = []
for key in doc_order:
	doc = sent_data[key]
	for sent in doc:
		y = int(sent[2]['rouge-2']['p'] > 0.1) # Our label 
		to_save.append([sent[0], y])

# Save in tsv format
final = pd.DataFrame(to_save)
final.to_csv(prefix + 'bert_data/ca_test.tsv', sep='\t', index=None)

