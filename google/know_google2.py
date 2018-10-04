import json
import urllib.request
import urllib.parse

api_key = 'enter-api-key'
service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
def extract(filename):
	count = 0
	count1 = 0
	output = open("google/test_google_2words.txt",'w')
	with open(filename) as file:
		prev = '	'
		prev_2 = '	'
		for line in file:
			count1 = count1 + 1
			if count1 > 0:
				if line != '\n':
					data = prev.split("\t")
					next_word = line.split("\t")[0]
					next_pos = line.split("\t")[1]
					prev = line
					if (data[1] == 'NNP' or data[1] == 'NNPS') and (next_pos == 'NNP' or next_pos == 'NNPS'):
						query = data[0] + " " + next_word
						params = {'query': query,'limit': 1,'indent': True,'key': api_key}
						url = service_url + '?' + urllib.parse.urlencode(params)
						response = json.loads(urllib.request.urlopen(url).read())
						if len(response['itemListElement']) > 0:
							if '@type' in response['itemListElement'][0]['result']:
								type = response['itemListElement'][0]['result']['@type']
								type.remove('Thing')
								print(count)
								print(query)
								print(type)
								output.write(str(count) + "\t" + str(query) + "\t")
								for item in type:
									output.write("%s " % item)
								output.write("\n")
					count = count + 1
				else:
					count = count + 2
					prev = next(file,'False')		
	output.close()
	file.close()

if __name__=='__main__':
	extract('data/CONLL_test.pos-chunk')
