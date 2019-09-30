import json
import jsonlines
import numpy as np
import re
import os
import xml.etree.ElementTree as ET

# Text Length Cut-offs for the dataset
MIN_TEXT_LENGTH = 2000
MAX_TEXT_LENGTH = 20000


cleanr = re.compile('<.*?>')
def clean_html(raw_html):
  cleantext = re.sub(cleanr, '', raw_html)
  # Additional noise in docs
  cleantext = cleantext.replace('&lt;all&gt;', '')
  return cleantext

def clean_summary(text):
    '''
    Get rid of html and the filler text 
    '''
    text = clean_html(text)

    # Procedural text - not relevant 
    text = text.replace('(This measure has not been amended since it was introduced', '')
    text = text.replace('The expanded summary of the House reported version is repeated here.)', '')
    text = text.replace('The summary has been expanded because action occurred on the measure.)', '')
    text = text.replace('The summary of that version is repeated here.)', '')

    return text

def extract_data_xml(bill_data):
    '''
    Find the text of the latest summary - from xml. This structure is flexible 
    to handle formatting changes between sessions.

    bill_data: XML structure

    Returns dict of title and summary
    '''

    if bill_data.tag == 'billStatus':
        bill_data = bill_data.find('bill')

    # First find the title 
    latest_title = bill_data.find('titles').getchildren()[-1]
    if latest_title.tag == 'item':

        title = latest_title.find('title').text
    else:
        title = latest_title.text

    summary = None
    # Two possible structures - we check which one applies
    if bill_data.find('summary') is not None:
        summary =  bill_data.find('summary').text

    elif bill_data.find('summaries'):

        head = bill_data.find('summaries').find('billSummaries')

        # Get latest of all summaries
        all_summaries = head.findall('item')
        if len(all_summaries) > 0:
            latest = all_summaries[-1]
            summary = latest.find('text').text 

    # Clean up summary 
    if summary is not None:
        summary = clean_summary(summary)

    return {'title': title, 'summary': summary}


def extract_data_json(data):
    '''
    Extract latest summary from a bill data json.
    data: JSON/Dict

    Returns text or None (if no summary)
    '''

    titles = data.get('titles')
    if len(titles) == 0:
        title = None
    else:
        title = titles[-1].get('title')

    if data.get('summary') is None:
        summary = None
    else:
        summary = data.get('summary').get('text')

    return {'title': title, 'summary': summary}


def find_latest_text(bill_dir):
    '''
    Pick the latest bill version from the text versions.

    bill_dir: path to directory for bill (text_versions is subdir)

    Returns path to best subdir
    '''

    house_order = ['ih', 'rth', 'rh', 'rfh','rch', 'eh', 'ath', 'pcs', 'hds', 'rds', 'rfs', 'rs', 'rs2', 'rcs', 'es', 'ats', 'cps', 'enr', 'pp']
    senate_order = [ 'is',  'rds', 'rfs', 'rs', 'rs2', 'rcs', 'es', 'ats', 'cps', 'ih', 'rth', 'rh', 'rfh','rch', 'eh', 'ath', 'enr', 'pp']
    
    versions = os.listdir(os.path.join(bill_dir, 'text-versions'))

    cur_order = house_order if 'bills/h' in bill_dir else senate_order
    idxs = [cur_order.index(d) if d in cur_order else -1 for d in versions]
    best = np.argmax(idxs)

    return os.path.join(bill_dir, 'text-versions', versions[best])

def prepare_html_text(html_text):
    '''
    Takes in bill html and prepares into final text block.

    Cleans up html and removes metadata included in the file 
    '''

    text = clean_html(html_text)

    # Get rid of metadata - using known cues 
    phrase = 'United States of America in Congress assembled,'
    
    if phrase in text:
        i = text.index(phrase)
        text = text[i + len(phrase):]
    if 'RESOLUTION' in text:
        i = text.index('RESOLUTION')
        text = text[i + len('RESOLUTION'):]
    if 'Concurrent Resolution' in text:
        i = text.index('Concurrent Resolution')
        text = text[i + len('Concurrent Resolution'):]
    if 'Joint Resolution' in text:
        i = text.index('Joint Resolution')
        text = text[i + len('Joint Resolution'):]

    if 'Joint Resolution' in text:
        i = text.index('Joint Resolution')
        text = text[i + len('Joint Resolution'):]

    # Clean off white space from both ends
    text = text.strip()

    # Weird ending
    if 'Union Calendar' in text:
        i = text.index('Union Calendar')
        text = text[:i]

    return text

def prepare_bill(bill_dir, session):
    '''
    Take in a bill directory with all bill data and return a dict with the 
    bill_id, title, summary, text.

    Takes in session to make a more specific billid
    '''
    print(bill_dir)
    # # Extract basic bill data
    # if os.path.isfile(os.path.join(bill_dir, 'data.json')):
    #     data = json.load(open(os.path.join(bill_dir, 'data.json')))
    #     final_data = extract_data_json(data)

    # elif os.path.isfile(os.path.join(bill_dir, 'data.xml')):
    #     f = os.path.join(bill_dir, 'data.xml')
    #     bill_data = ET.parse(f).getroot()
    #     final_data = extract_data_xml(bill_data)
    # else:
    #     raise ValueError('No data file for bill ')

    final_data = {}

    # Extract basic bill data
    if os.path.isfile(os.path.join(bill_dir, 'data.json')):
        data = json.load(open(os.path.join(bill_dir, 'data.json')))
        final_data = extract_data_json(data)
    elif os.path.isfile(os.path.join(bill_dir, 'data.xml')):
        f = os.path.join(bill_dir, 'data.xml')
        bill_data = ET.parse(f).getroot()
        final_data = extract_data_xml(bill_data)
    else:
        raise ValueError('No data file for bill ')

    # Get the bill id from the path - its the final subfolder
    billid = os.path.basename(os.path.normpath(bill_dir))
    final_data['bill_id'] = str(session) + '_' + billid 

    # Next get the text 
    f = os.path.join(bill_dir, 'text-versions')

    if not os.path.isdir(f):
        raise ValueError('No text for bill')

    # Figure out which text version is the latest
    latest_version = find_latest_text(bill_dir)
    text_file = os.path.join(latest_version, 'document.html')
    
    if not os.path.isfile(text_file):
        raise ValueError('No text for bill')

    with open(text_file) as reader:
        t = reader.read()
        text = prepare_html_text(t)

        # If to short or too long, dont return the text
        # if len(text) < MIN_TEXT_LENGTH or len(text) > MAX_TEXT_LENGTH:

        #     text = None

        final_data['text'] = text

    return final_data


if __name__ == '__main__':
    for ses in range(107, 113):

        # Change to your prefix
        path = '/data/final_data/congress/{}/bills/'.format(ses)

        final_dataset = []
        i = 0
        j = 0
        z = 0
        for btype in os.listdir(path):
            if '.DS_Store' in btype:
                continue
            subpath = os.path.join(path, btype)
            if 'res' in btype:
                continue

            for file in os.listdir(subpath):
                
                billpath = os.path.join(subpath, file)

                if  os.path.isdir(billpath):
                    try:
                        bd = prepare_bill(billpath, ses)
                        if bd.get('summary') is not None and bd.get('text') is not None:
                            final_dataset.append(bd)
                            i += 1

                        else:
                            j += 1
                    except ValueError as e:
                        print(e, file)
                        z += 1

        print(j, i, z)

        with open('/data/final_data/final/final_data_{}.jsonl'.format(ses), 'w') as f:

            writer = jsonlines.Writer(f)
            writer.write_all(final_dataset)


                    
