'''
Aggregate scores from individual bills
'''
import os
import pandas as pd
import pickle

prefix = os.path.expanduser('~/BSDATA/')


for file in os.listdir(prefix + 'score_data'):

	if file == 'baseline_scores':
		continue

	all_scores = pickle.load(open(prefix + 'score_data/' + file, 'rb'))
	value_data = {}
	for k, ds in all_scores.items():
	    for v1, inner in ds.items():
	        for v2, score in inner.items():
	            value_data[(k, v1, v2)] = [score]


	data = pd.DataFrame(value_data)
	d2 = data.stack(0)
	data = d2.droplevel(0)
	print(file)
	data = data[[('rouge-1', 'f'), ('rouge-2', 'f'), ('rouge-l', 'f')]]
	print(data.describe())
	print('---')