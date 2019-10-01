import os
import pandas as pd 
import pickle

from rouge import Rouge
rouge = Rouge()

prefix = os.environ['BILLSUM_PREFIX']

if not prefix.endswith('/'):
    prefix += '/'


# Load in reference summaries
for locality in ['us', 'ca']:


	# Load in generated summaries
	# UPDATE ME WITH YOUR LOCATION
	new_sums = pd.read_json("new_summaries_{}.jsonl".format(locality), lines=True)

	docs = pd.read_json(prefix + 'clean_final/{}_test_data_final.jsonl'.format(locality), lines=True)
	docs.set_index('bill_id', inplace=True)

	all_rscores = []
	for _, row in new_sums.iterrows():
		if len(row['my_sum']) > 2000:
			raise ValueError("Your summary is too long for bill {}".format(row['bill_id']))
		# Find relevant summary in docs and compute rouge
		rscore = rouge.get_scores([docs.loc[row['bill_id']].clean_summary], [row['my_sum']])[0]

		all_rscores.append(rscore)


	# Aggregate the stats
	value_data = {}
	for k, ds in enumerate(all_rscores):
	    for v1, inner in ds.items():
	        for v2, score in inner.items():
	            value_data[(k, v1, v2)] = [score]


	data = pd.DataFrame(value_data)
	d2 = data.stack(0)
	data = d2.droplevel(0)
	print("LOCALITY", locality)
	data = data[[('rouge-1', 'f'), ('rouge-2', 'f'), ('rouge-l', 'f')]]
	print(data.describe())
	print('---')

