# Methods to create final summaries froms cored sentences
import numpy as np 
import operator
from sklearn.metrics.pairwise import cosine_similarity

MAX_SUMMARY_LENGTH = 2000

def greedy_summarize(sent_texts, weights, threshold=15, return_idx=False,
                        summary_len = MAX_SUMMARY_LENGTH):

    # Sort indicies of weights
    sweights = np.argsort(weights)
    
    sent_lens = [len(s) for s in sent_texts]

    # See how many we can add until we reach limit 
    top_idx = [] 
    total_chars = 0

    for i in reversed(sweights):

        mylen = sent_lens[i]
        
        # If sentence is too long - skip it
        if total_chars + mylen > summary_len:
            continue
        
        # if len(sent_texts[i].split()) < 8:
        #     continue # 2 short

        top_idx.append(i)
        total_chars += mylen

    # Put sentences into original document order
    final_idx = sorted(top_idx)
    
    if return_idx:
        return final_idx
        
    return np.array(sent_texts)[final_idx]

