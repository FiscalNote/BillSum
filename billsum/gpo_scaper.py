'''
Download recent bills statuses and format to match older structure/text file structure.
A lot of regex tricks in this file, so change with caution.
'''
import requests
import os
import re
import zipfile

def get_json(url):
    accept = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'

    useragent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"

    header = {"user-agent": useragent, "accept": accept,
                "cookie": "HttpOnly; HttpOnly; HttpOnly; HttpOnly; HttpOnly; gsScrollPos-1514522618=; HttpOnly; has_js=1; gsScrollPos-1514522492=; _pk_ref.1.4253=%5B%22%22%2C%22%22%2C1542218240%2C%22https%3A%2F%2Fwww.gpo.gov%2Ffdsys%2F%22%5D; _pk_ses.1.4253=*; _pk_id.1.4253=3cd04d2c0055765c.1542133090.7.1542218245.1542218240."}

    r = requests.get(url, headers=header)

    return r

local_path = '../usbills/data/'

i = 0
prefix = 'BILLSTATUS'
base = 'https://www.govinfo.gov/bulkdata/json/' + prefix + '/'

for session in range(113, 116):

    print(session)

    url = base + str(session)
    to_explore = [url]

    while len(to_explore) > 0:

        url = to_explore.pop()
        r = get_json(url)

        if 'json' in r.headers.get('Content-Type'):
            # Directory with subdirectories leading to data 

            to_explore += [l['link'] for l in r.json()['files'] if 'xml' not in l['link']]

            start = url.find(prefix) + len(prefix)
            path = os.path.join(local_path, url[start:])
            
            # Hack to avoid splitting into "subsessions"
            path = re.sub(r'/[12](?:$|/)', '/', path)

        elif 'zip' in url:
            # Actual data files
           
            data = r.content

            # Save output - create local path
            start = url.find(prefix) + len(prefix) + 1
            path = os.path.join(local_path, url[start:])
            
            ### We want to match the govinfo scraper format ###
            ### Hence all the stromg formatting magic below ###

            # Alter path to add the bills subdirectory (consistency)
            path = path.replace('{}/'.format(session), '{}/bills/'.format(session))

            # Point to final bill id folder
            directory = os.path.dirname(path)

            if not os.path.exists(directory):
               os.makedirs(directory)

            filename = os.path.basename(path)
            final_path = os.path.join(directory, filename)
            
            # Finally save the zip file
            with open(path, 'wb') as f:
               f.write(data)

            # Extract from zipfile
            zip_ref = zipfile.ZipFile(path, 'r')
            zip_ref.extractall(directory)
     
            # Move each status file into the appropriate bills folder
            for file in os.listdir(directory):
                if 'BILLSTATUS' in file and '.zip' not in file:
                    # Extract the bill id
                    billid = re.findall('BILLSTATUS-([0-9]{3}[a-z]+[0-9]+)', file)[0]
                    # Fix formatting
                    billid = re.sub(r'[0-9]{3}([a-z]+[0-9]+)', r'\1', billid)

                    # Move to corresponding bills file
                    filename = os.path.join(directory, file)
                    newdir = os.path.join(directory, billid)
                    newname = os.path.join(newdir, 'data.xml')

                    if not os.path.exists(newdir):
                        os.makedirs(newdir)

                    os.rename(filename, newname)





