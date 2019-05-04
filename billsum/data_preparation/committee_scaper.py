from bs4 import BeautifulSoup
import requests
import time

base = requests.get('https://www.govinfo.gov/sitemap/CHRG_2018_sitemap.xml')

soup = BeautifulSoup(base.text)

i = 0
for ch in soup.find_all('loc'):
	print(i)
	i += 1
	if i % 100 == 0:
		time.sleep(30)
	url = ch.text 

	cid = url.split('/')[-1]

	final_url = 'https://www.govinfo.gov/content/pkg/{}/html/{}.htm'.format(cid, cid)

	print(final_url)
	data = requests.get(final_url).text 

	with open('usdata/committee_hearings/2018/{}.htm'.format(cid), 'w') as f:
		f.write(data)

