from collections import defaultdict
import json
import jsonlines
import numpy as np
import os
import pickle
import re
import xml.etree.ElementTree as ET


house_order = ['ih', 'rth', 'rh', 'rfh','rch', 'eh', 'ath', 'pcs', 'hds', 'rds', 'rfs', 'rs', 'rs2', 'rcs', 'es', 'ats', 'cps', 'enr', 'pp']
senate_order = [ 'is',  'rds', 'rfs', 'rs', 'rs2', 'rcs', 'es', 'ats', 'cps', 'ih', 'rth', 'rh', 'rfh','rch', 'eh', 'ath', 'enr', 'pp']

DATA_PREFIX = '../usbills/data/'

def text_from_xml(node):
    """Create text iterator.
    The iterator loops over the element and all subelements in document
    order, returning all inner text.
    """
    tag = node.tag
    if not isinstance(tag, str) and tag is not None:
        return
    t = node.text
    
    # We may or may not want this?
    if t and tag not in ['enum', 'header']:
        yield t
    for e in node:
        yield text_from_xml(e)
        t = e.tail
        if t:
            yield t

def extract_summary_xml(bill_data):
    '''
    Find the text of the latest summary - from xml. This structure is flexible 
    to handle formatting changes between sessions.

    bill_data: XML structure

    Returns text or None (if no summary)
    '''

def extract_summary_json(data):
    '''
    Extract latest summary from a bill data json.
    data: JSON/Dicy

    Returns text or None (if no summary)
    '''

    if data.get('summary') is None or data['summary'].get('text') is None:
        return None

    return data['summary']['text']
            
def process_old_session(session):
    '''
    Transform pre-113 sources into a jsonlines format with key details only. 
    '''

    path = '{}/{}/bills/'.format(DATA_PREFIX, session)

    final_data  = []

    # Iterate over all bill types/bills looking for ones with text versions
    for (dirpath, dirnames, filenames) in os.walk(path):
    
        # Found bill with texts
        if dirpath.endswith('text-versions'):
            
            # Find the matching bill info file 
            temp = dirpath.replace('text-versions', 'data.json')
            f = json.load(open(temp))
            
            # If there is no summary, skip this bill
            if f.get('summary') is None or f['summary'].get('text') is None:
                continue
            
            summary = f['summary']['text']

            # Reformat billid to match newer docs (hr100-115 => 115hr100)
            parts = f['bill_id'].split('-')
            billid = parts[1] + parts[0]

            # Get most recent title
            title = f['titles'][0]['title']

            # There are multiple bill versions - pick most recent one
            cur_order = house_order if 'bills/h' in dirpath else senate_order
            idxs = [cur_order.index(d) if d in cur_order else -1 for d in dirnames]
            best = np.argmax(idxs)

            # Get text if it exists
            p1 = os.path.join(dirpath, dirnames[best], 'document.html')
            if os.path.exists(p1):
                with open(p1) as f2:
                    text = f2.read()

                    bill_data = {'bill_id': billid, 'summary': summary,
                                'title': title, 'text': text}
                    final_data.append(bill_data)          

    print("Final bill count for {}: {}".format(session, len(final_data)))
    return final_data

def process_new_session(session):

    # First we need to collect all sumamries and bill versions
    summaries = {}
    bill_versions = defaultdict(list)
    
    # Regex to find the bill ids and statuses
    billid_r = re.compile('[0-9]+[a-z]+[0-9]+')
    statuscode_r = re.compile('[a-z]+[0-9]*(?=.pretty)')
    
    path = '/Users/anastassia.kornilova/usbills/data/{}/'.format(session)
    bad = 0
    i = 0

    # Collect both the summaries and texts at the same time
    for (dirpath, dirnames, filenames) in os.walk(path):
        for file in filenames:
            i += 1
            if i % 100 == 0:
                print(i, file)

            # Get summary from metadata file
            if 'BILLSTATUS' in file and '.zip' not in file:
                billid = file.split('-')[1].split('.xml')[0]

                bill_data = ET.parse(os.path.join(dirpath , file)).getroot()
                
                sums = list(bill_data.iter('billSummaries'))[0]
                
                # Keep the latest summary only
                if len(sums) > 0:
                    s = sums[-1]

                    sum_text = None

                    # Get the text field out
                    for child in s:
                        if child.tag == 'text':
                            sum_text = child.text

                    if sum_text is not None:
                        summaries[billid] = sum_text

            # Record text
            if 'BILLS-' in file and '.xml' in file:
                billid = billid_r.findall(file)[0]
                status = statuscode_r.findall(file)[0]

                with open(os.path.join(dirpath, file)) as f:
                    text = f.read()
                    bill_versions[billid].append((status, text))

    print("Could not find summary for", bad)
    # Next combine final text with final status data
    house_r = re.compile('\A[0-9]+h')
    senate_r = re.compile('\A[0-9]+s')

    final_data = {}

    for billid, version in bill_versions.items():
        if house_r.match(billid):
            my_order = house_order
        elif senate_r.match(billid):
            my_order = senate_order
        else:
            print("BAD", billid)
            
        idx = [my_order.index(v[0]) for v in version if v[0] in my_order]
        # Weird edge case
        if len(idx) == 0:
            best = 0
        else:  
            best = np.argmax(idx)
        final_text = version[best][1]
        #final_text = bpp.preprocess(final_text)

        status = statuses.get(billid)
        if status:
            final_data[billid] = (status, final_text)
    return final_data


newform = {'115': '20172018r', '114': '20152016r', '113': '20132014r', '111': '20092010r', '112': '20112012r'}


for session in range(103, 113):
   data = process_old_session(session)

   path = DATA_PREFIX + 'final_data/bills_{}.jsonl'.format(session)
   
   with jsonlines.open(path, mode='w') as writer:
        writer.write_all(data)
  

# for session in range(111, 113):
#     path = os.path.expanduser('~/usbills/data/{}/bills/'.format(session))

#     for (dirpath, dirnames, filenames) in os.walk(path):
#         if 'package.zip' in filenames:
#             mypath = dirpath + '/package.zip'
#             zip_ref = zipfile.ZipFile(mypath, 'r')
#             zip_ref.extractall(dirpath)

         
# for session in [111, 112]:
#     #prettify_old_session(session)
#     data = process_old_session(session)
#     ses2 = newform[str(session)]
#     pickle.dump(data, open('../new_data/all_data_{}.pkl'.format(ses2), 'wb'))

# for session in [115]:
#     #prettify_new_session(session)
#     data  = process_new_session(session)
#     ses2 = newform[str(session)]
#     pickle.dump(data, open('../new_data/all_data_{}.pkl'.format(ses2), 'wb'))


