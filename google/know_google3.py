import json
import urllib.request
import urllib.parse

api_key = 'enter-api-key'
service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
def extract(filename):
    count = 0
    output = open("google/dev_google_3words.txt",'w')
    with open(filename) as file:
        prev = '	'
        prev_2 = '	'
        for line in file:
            count = count + 1
            if count > 0:
                if line != '\n':
                    data = prev.split("\t")
                    data_2 = prev_2.split("\t")
                    next_word = line.split("\t")[0]
                    next_pos = line.split("\t")[1]
                    prev_2 = prev
                    prev = line
                    if (data_2[1] == 'NNP' or data_2[1] == 'NNPS') and (data[1] == 'NNP' or data[1] == 'NNPS') and (next_pos == 'NNP' or next_pos == 'NNPS'):
                        query = data_2[0] + " " + data[0] + " " + next_word
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
                                output.write(query + "\t")
                                for item in type:
                                    output.write("%s " % item)
                                output.write("\n")
									
    output.close()
    file.close()

if __name__=='__main__':
    extract('data/CONLL_dev.pos-chunk')
