import json
import urllib.request
import urllib.parse

api_key = 'enter_api_key'
service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
def extract(filename):
    count = 0
    output = open("google/test_google.txt",'w')
    with open(filename) as file:
        for line in file:
            count = count + 1
            if count > 0:
                if line != '\n':
                    data = line.split("\t")
                    if data[1] == 'NNP' or data[1] == 'NNPS' or (data[0][0].isupper() and data[1] in ['NN','NNS']):
                        query = data[0]
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
					
    output.close()
    file.close()

if __name__=='__main__':
    extract('data/CONLL_test.pos-chunk')
