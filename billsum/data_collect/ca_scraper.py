import pickle
from requests_html import HTMLSession
import time

session = HTMLSession()


base = "http://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=201520160"


data = []

try:
	for i in range(1, 1482):
		try:
			if i % 10 == 0:
				print(i, len(data))
				time.sleep(5)

			url = base + "SB{}".format(i)

			r = session.get(url)

			if '<strike/>' in str(r.html.find('#bill', first=True).raw_html):
				print('too fancy')
				continue

			title = r.html.find('#title', first=True).text 

			summary = r.html.find('#digesttext', first=True).text 

			text = r.html.find('#bill', first=True).text

			text = text.replace(u'\xa0', u' ')

			data.append({'summary': summary, 'text': text, 'title': title, 'external_id': 'SB {}'.format(i)})
		except KeyboardInterrupt:
			raise KeyboardInterrupt
		except:
			print("bad bill", i)


	for i in range(1, 2916):
		try:
			if i % 10 == 0:
				print(i, len(data))
				time.sleep(5)

			url = base + "AB{}".format(i)

			r = session.get(url)

			if '<strike/>' in str(r.html.find('#bill', first=True).raw_html):
				print('too fancy')
				continue

			title = r.html.find('#title', first=True).text 

			summary = r.html.find('#digesttext', first=True).text 

			text = r.html.find('#bill', first=True).text

			text = text.replace(u'\xa0', u' ')

			data.append({'summary': summary, 'text': text, 'title': title, 'external_id': 'AB {}'.format(i)})

		except KeyboardInterrupt:
			raise KeyboardInterrupt
		except:
			print("bad bill", i)

finally:
	pickle.dump(data, open('ca_senate_20152016.pkl', 'wb'))