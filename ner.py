import nltk
import sys
from nltk.classify import maxent
import numpy as np
from nltk.corpus import wordnet as wn
import re
import pickle
import json
import urllib.request
import urllib.parse
import copy

l = []
np.seterr(all='ignore')
tags = ['I-PER','I-ORG','I-LOC','I-MISC','O','NaN','B-LOC','B-MISC','B-ORG','B-PER']
organization_common_words = ['Medical','Pharmaceuticals','Communications','Organization','Corporation','Campaign','Inc','Corp','Co','&','International','Treasuries','Treasury','Federal','Institute','Associates','Ltd','Committee','National','Agency','Council','Exchange']
TP = {}
for tag in tags:
	TP[tag] = {}
	for totag in tags:
		TP[tag][totag] = 0

def loadGloveModel(gloveFile):
	f = open(gloveFile,'r')
	model = {}
	for line in f:
		splitLine = line.split()
		word = splitLine[0]
		embedding = np.array([float(val) for val in splitLine[1:]])
		l.append(embedding)
		model[word] = embedding
	return model

embeddings = loadGloveModel('glove/glove.6B.50d.txt')
l = np.asarray(l)

def binarize_embeddings():
	uplus = {}
	bminus = {}
	uarr = np.zeros(l.shape)
	barr = np.zeros(l.shape)
	col_mean = l.mean(axis=0)
	for j,column in enumerate(l.T):
		for i,value in enumerate(column):
			if value >= col_mean[j]:
				uarr[i][j] = 1
			elif value <= col_mean[j]:
				barr[i][j] = 1
	i = 0
	for word in embeddings.keys():
		uplus[word] = uarr[i]
		bminus[word] = barr[i]
		i = i + 1
	return uplus,bminus

uplus,bminus = binarize_embeddings()

#function to use google knowledge graph api to get categories for proper nouns
def google_lookup(query):
	type = ''
	api_key = 'AIzaSyAWM-84q4_vcwWymfoSboj5F6vcQREDFCU'
	service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
	params = {
    		'query': query,
    		'limit': 1,
    		'indent': True,
    		'key': api_key,
	}
	url = service_url + '?' + urllib.parse.urlencode(params)
	response = json.loads(urllib.request.urlopen(url).read())
	
	if len(response['itemListElement']) > 0:
		type = response['itemListElement'][0]['result']['@type']
		type.remove('Thing')
	return type

#function to clean up google categories
def google_cleanup(google_category):
	google_category = google_category.strip("\n")
	if 'Place' in google_category:
		google_category = 'Place'
	if 'Organization' in google_category:
		google_category = 'Organization'
	google_category = google_category.strip()
	return google_category

def extract_training_data(labeled_data,filename,google_filename1,google_filename2,google_filename3):
	count = 1
	flag_2 = 'False'
	train = 'True'
	hyphen_flag = 'False'
	google_category = ''
	google_category2 = ''
	google_category3 = ''
	f = open(google_filename1, 'r')
	f1 = open(google_filename2,'r')
	f2 = open(google_filename3,'r')
	with open(filename) as file:
		line1 = f.readline()
		line2 = f1.readline()
		line3 = f2.readline()
		prev_tag = 'NaN'
		prev_word = 'BoS'
		prev_pos = 'NaN'
		prev_chunk = 'NaN'
		prev = next(file)
		for line in file:
			#extracting data from the google category files to populate features
			data = prev.split("\t")
			TP[prev_tag][data[3].strip()] += 1
			if flag_2 == 'True':
				google_category2 = google_cleanup(type_2[2])
				flag_2 = 'False'
				line2 = f1.readline()
			
			type = line1.split("\t")
			type_2 = line2.split("\t")
			type_3 = line3.split("\t")
			if line3:
				while int(type_3[0]) < count:
					line3 = f2.readline()
					type_3 = line3.split("\t")
			
			if type[0] == str(count):
				line1 = f.readline()
				google_category = google_cleanup(type[2])

			
			if type_2[0] == str(count):
				flag_2 = 'True'
				google_category2 = google_cleanup(type_2[2])

			
			if type_3[0] == str(count):
				line3 = f2.readline()
				google_category3 = google_cleanup(type_3[2])

			
			if line != '\n':
				next_word = line.split("\t")[0]
				next_pos = line.split("\t")[1]
				next_chunk = line.split("\t")[2].strip("\n")
				prev = line
				count += 1
			else:
				next_word = 'EoS'
				next_pos = 'NaN'
				next_chunk = 'NaN'
				prev = next(file,False)
				count += 2
			if '-' in data[0] and prev_word != 'BoS' and data[1] == 'JJ':
				word = data[0].split("-")
				if word[0] and word[1]:
					if word[0][0].isupper() or word[1][0].isupper():
						hyphen_flag = 'True'
								
			labeled_data.append((prev_tag,prev_word,prev_pos,prev_chunk,data[0],data[1],data[2],data[3].strip("\n"),next_word,next_pos,next_chunk,google_category,google_category2,google_category3,train))
			google_category = ''
			google_category2 = ''
			google_category3 = ''
			
			
			if line != '\n':
				prev_word = data[0]
				prev_pos = data[1]
				prev_chunk = data[2]
				prev_tag = data[3].strip("\n")
			else:
				prev_word = 'BoS'
				prev_pos = 'NaN'
				prev_chunk = 'NaN'
				prev_tag = 'NaN'
				
	return labeled_data

def viterbi():
	for tag in tags:
		tag_count1 = sum(TP[tag].values())
		for totag in tags:
			if tag_count1 > 0:
				TP[tag][totag] = TP[tag][totag] / tag_count1

def build_train_set(labeled_data):
	train_set = [(feature_builder(prev_tag,prev_word,prev_pos,prev_chunk,token,pos,chunk,next_word,next_pos,next_chunk,google_category,google_category2,google_category3,train),tag) for (prev_tag,prev_word,prev_pos,prev_chunk,token,pos,chunk,tag,next_word,next_pos,next_chunk,google_category,google_category2,google_category3,train) in labeled_data]
	return train_set

def feature_builder(prev_tag,prev_word,prev_pos,prev_chunk,token,pos,chunk,next_word,next_pos,next_chunk,google_category,google_category2,google_category3,train):
	features = {}
	hyphen_flag = 'False'
	word = []
	features["token"] = token
	if '-' in token and prev_word != 'BoS' and pos == 'JJ':
		word = token.split("-")
		if word[0] and word[1]:
			if word[0][0].isupper() or word[1][0].isupper():
				hyphen_flag = 'True'
	
	if pos == 'NNP' or pos == 'NNPS' or hyphen_flag == 'True' or (token[0].isupper() and pos in ['NN','NNS','JJ'] and '-' not in token):
		hyphen_flag = 'False'
		features['chunk'] = chunk
		features['pos'] = pos
		features['prev_chunk'] = prev_chunk
		features['prev_pos'] = prev_pos
		features['next_chunk'] = next_chunk
		features['next_pos'] = next_pos
		features["prefix_1"] = token[0:1]
		features["prefix_2"] = token[0:2]
		features["prefix_3"] = token[0:3]
		features["prefix_4"] = token[0:4]
		features["prefix_5"] = token[0:5]
		features["prefix_6"] = token[0:6]
		features["suffix_1"] = token[-1].lower()
		features["suffix_2"] = token[-2:].lower()
		features["suffix_3"] = token[-3:].lower()
		features["suffix_4"] = token[-4:].lower()
		features["suffix_5"] = token[-5:].lower()
		features["suffix_6"] = token[-6:].lower()
		features['upper'] = token.upper()
		features['title'] = token.title()
		
		if prev_word == 'BoS':
			features['BoS'] = 1
		if next_word == 'EoS':
			features['EoS'] = 1
		if prev_word in organization_common_words or next_word in organization_common_words or token in organization_common_words:
			features['Organization'] = 1
		
		if token == '&':
			features['ampersand'] = 1
		
		if '-' in token:
			features['hyphen'] = 1
			features['hyphenated_word'] = token
		
		if token.isupper():
			features['upper'] = 1
		
		shape = re.sub('[A-Z]','X',token)
		shape = re.sub('[a-z]','x',shape)
		shape = re.sub('[0-9]','d',shape)
		
		next_shape = re.sub('[A-Z]','X',next_word)
		next_shape = re.sub('[a-z]','x',next_shape)
		next_shape = re.sub('[0-9]','d',next_shape)
		
		prev_shape = re.sub('[A-Z]','X',prev_word)
		prev_shape = re.sub('[a-z]','x',prev_shape)
		prev_shape = re.sub('[0-9]','d',prev_shape)
		
		sshape = re.sub('[a-z]+','x',token)
		sshape = re.sub('[A-Z]+','X',sshape)
		sshape = re.sub('[0-9]+','d',sshape)
		
		prev_sshape = re.sub('[a-z]+','x',prev_word)
		prev_sshape = re.sub('[A-Z]+','X',prev_sshape)
		prev_sshape = re.sub('[0-9]+','d',prev_sshape)
		
		next_sshape = re.sub('[a-z]+','x',next_word)
		next_sshape = re.sub('[A-Z]+','X',next_sshape)
		next_sshape = re.sub('[0-9]+','d',next_sshape)
			
		
		#features["sshape"] = sshape
		#features["prev_sshape"] = prev_sshape
		#features["next_sshape"] = next_sshape
		features["shape"] = shape
		features['prev_tag'] = prev_tag
		
		features["next_word"] = next_word
		features["next_shape"] = next_shape
		
		features["prev_word"] = prev_word
		features["prev_shape"] = prev_shape
		
		#features["prev_current_sshape"] = features["prev_sshape"],features["sshape"]
		#features["current_next_sshape"] = features["sshape"],features["next_sshape"]
		features["prev_current"] = features["prev_shape"] + " " + features["shape"]
		features["current_next"] = features["shape"] + " " + features["next_shape"]
		features["prev_current_next"] = features["prev_shape"] + " " + features["shape"] + " " + features["next_shape"]
		features['prev_current_pos'] = prev_pos + " " + pos
		features['prev_current_tokens'] = prev_word + " " + token
		features['current_next_pos'] = pos + " " + next_pos
		features['current_next_tokens'] = token + " " + next_word
		features['current_next_chunk'] = chunk + " " + next_chunk
		features['prev_current_chunk'] = prev_chunk + " " + chunk
		
		if '-' in token and pos == 'JJ' and prev_word != 'BoS':
			if word[0] and word[1]:
				t = list(wn.synsets(word[0]))
				t1 = list(wn.synsets(word[1]))
				if t and word[0][0].isupper():
					features['wordnet1'] = wn.synset(t[0].name()).lexname()
				elif t1 and word[1][0].isupper():
					features['wordnet1'] = wn.synset(t1[0].name()).lexname()
		else:
			t = list(wn.synsets(token))
			if t:
				features['wordnet1'] = wn.synset(t[0].name()).lexname()
		
		word = token + "_" + next_word
		t = list(wn.synsets(word))
		if t:
			features['wordnet2'] = wn.synset(t[0].name()).lexname()
		
		word = prev_word + "_" + token + "_" + next_word
		t = list(wn.synsets(word))
		if t:
			features['wordnet3'] = wn.synset(t[0].name()).lexname()
		
		
		if google_category != '':
			features["google1"] = google_category
		if google_category2 != '':
			features["google2"] = google_category2
		if google_category3 != '':
			features["google3"] = google_category3
		
		if (prev_pos == 'NNP' or prev_pos == 'NNPS') and (next_pos == 'NNP' or next_pos == 'NNPS'):
			features['prev_current_next_pos'] = prev_pos + " " + pos + " " + next_pos
			features['prev_current_next_tokens'] = prev_word + " " + token + " " + next_word
			features['prev_current_next_chunks'] = prev_chunk + " " + chunk + " " + next_chunk
			
		if token.lower() in uplus:
			udimensions = uplus[token.lower()]
			bdimensions = bminus[token.lower()]
		else:
			udimensions = uplus['<unk>']
			bdimensions = bminus['<unk>']
		for i,d in enumerate(udimensions):
			ukey = 'u' + str(i)
			bkey = 'b' + str(i)
			features[ukey] = udimensions[i]
			features[bkey] = bdimensions[i]

	
	if prev_tag == 'I-ORG' and (next_pos == 'NNP' or next_pos == 'NNPS') and token in ['of','for','and']:
			
		features['prev_current_next_pos'] = prev_pos + " " + pos + " " + next_pos
		features['prev_current_next_tokens'] = prev_word + " " + token + " " + next_word
		features['prev_current_next_chunks'] = prev_chunk + " " + chunk + " " + next_chunk
		shape = re.sub('[A-Z]','X',token)
		shape = re.sub('[a-z]','x',shape)
		shape = re.sub('[0-9]','d',shape)
		
		next_shape = re.sub('[A-Z]','X',next_word)
		next_shape = re.sub('[a-z]','x',next_shape)
		next_shape = re.sub('[0-9]','d',next_shape)
		
		prev_shape = re.sub('[A-Z]','X',prev_word)
		prev_shape = re.sub('[a-z]','x',prev_shape)
		prev_shape = re.sub('[0-9]','d',prev_shape)
		features["shape"] = shape
		features["prev_shape"] = prev_shape
		features["next_shape"] = next_shape
		features["prev_current_next"] = prev_shape + " " + shape + " " + next_shape
		features['prev_tag'] = prev_tag
		features["prev_word"] = prev_word
		features["next_word"] = next_word
		features['chunk'] = chunk
		features['pos'] = pos
		features['prev_chunk'] = prev_chunk
		features['prev_pos'] = prev_pos
		features['next_chunk'] = next_chunk
		features['next_pos'] = next_pos
		features["prefix_1"] = token[0:1]
		features["prefix_2"] = token[0:2]
		features["prefix_3"] = token[0:3]
		features["prefix_4"] = token[0:4]
		features["prefix_5"] = token[0:5]
		features["prefix_6"] = token[0:6]
		features["suffix_1"] = token[-1].lower()
		features["suffix_2"] = token[-2:].lower()
		features["suffix_3"] = token[-3:].lower()
		features["suffix_4"] = token[-4:].lower()
		features["suffix_5"] = token[-5:].lower()
		features["suffix_6"] = token[-6:].lower()
		features['upper'] = token.upper()
		features['title'] = token.title()
		if prev_word == 'BoS':
			features['BoS'] = 1
		if next_word == 'EoS':
			features['EoS'] = 1
		
		if prev_word in organization_common_words or next_word in organization_common_words or token in organization_common_words:
			features['Organization'] = 1
		
		if token == '&':
			features['ampersand'] = 1
		
		if token.isupper():
			features['upper'] = 1
		
		if google_category != '':
			features["google1"] = google_category
		if google_category2 != '':
			features["google2"] = google_category2
		if google_category3 != '':
			features["google3"] = google_category3
		
		word = token + "_" + next_word
		t = list(wn.synsets(word))
		if t:
			features['wordnet2'] = wn.synset(t[0].name()).lexname()
		
		word = prev_word + "_" + token + "_" + next_word
		t = list(wn.synsets(word))
		if t:
			features['wordnet3'] = wn.synset(t[0].name()).lexname()
			
		if token.lower() in uplus:
			udimensions = uplus[token.lower()]
			bdimensions = bminus[token.lower()]
		else:
			udimensions = uplus['<unk>']
			bdimensions = bminus['<unk>']
		for i,d in enumerate(udimensions):
			ukey = 'u' + str(i)
			bkey = 'b' + str(i)
			features[ukey] = udimensions[i]
			features[bkey] = bdimensions[i]
	return features

def process_test_data(classifier,filename,google_filename1,google_filename2,google_filename3,google_filename4):
	i = 0
	output = open("data/response.name",'w')
	last_ditch = open(google_filename4,'r')
	line4 = last_ditch.readline()
	prev_tags = []
	for i in range(25):
		prev_tags.append((-1,-1))
	count = 1
	flag_2 = 'False'
	train = 'False'
	google_category = ''
	google_category2 = ''
	google_category3 = ''
	f = open(google_filename1, 'r')
	f1 = open(google_filename2,'r')
	f2 = open(google_filename3,'r')
	with open(filename) as file:
		line1 = f.readline()
		line2 = f1.readline()
		line3 = f2.readline()
		prev = next(file)
		prev_tag = 'O'
		prev_word = 'BoS'
		prev_pos = 'NaN'
		prev_chunk = 'NaN'
		for line in file:
			data = prev.split("\t")
			#extracting category information from google category files for test and dev set feature population
			if flag_2 == 'True':
				google_category2 = google_cleanup(type_2[2])
				flag_2 = 'False'
				line2 = f1.readline()
			
			type = line1.split("\t")
			type_2 = line2.split("\t")
			type_3 = line3.split("\t")
			if line3:
				while int(type_3[0]) < count:
					line3 = f2.readline()
					type_3 = line3.split("\t")
			
			if type[0] == str(count):
				line1 = f.readline()
				google_category = google_cleanup(type[2])

			
			if type_2[0] == str(count):
				flag_2 = 'True'
				google_category2 = google_cleanup(type_2[2])

			
			if type_3[0] == str(count):
				line3 = f2.readline()
				google_category3 = google_cleanup(type_3[2])
			
			if line != '\n':
				next_word = line.split("\t")[0]
				next_pos = line.split("\t")[1]
				next_chunk = line.split("\t")[2].strip("\n")
				prev = line
				count += 1
			else:
				next_word = 'EoS'
				next_pos = 'NaN'
				next_chunk = 'NaN'
				prev = next(file,False)
				count += 2
			featureset = feature_builder(prev_tag,prev_word,prev_pos,prev_chunk,data[0],data[1],data[2].strip("\n"),next_word,next_pos,next_chunk,google_category,google_category2,google_category3,train)
			label = classifier.classify(featureset)
			m = classifier.prob_classify(featureset)
			max = 0
			#for tag in tags:
			#	if max < TP[prev_tag][tag] * m.prob(tag):
			#		max = TP[prev_tag][tag] * m.prob(tag)
			#		tagmax = tag
			#if tagmax != label and tagmax != 'O':
			#	label = tagmax
			
			google_category = ''
			google_category2 = ''
			google_category3 = ''

			
			if prev_tag == 'I-PER' and data[1] == 'NNP' and label == 'I-ORG':
				label = 'I-PER'
			
			if re.search('[0-9]+',data[0]) != None:
				label = 'O'
			
			if '-' in data[0] and data[1] == 'JJ' and prev_word != 'BoS':
				word = data[0].split('-')
				if word[0] and word[1]:
					t = list(wn.synsets(word[0]))
					t1 = list(wn.synsets(word[1]))
					if t and word[0][0].isupper():
						if wn.synset(t[0].name()).lexname() in ['noun.location']:
							label = 'I-MISC'
					if t1 and word[1][0].isupper():
						if wn.synset(t1[0].name()).lexname() in ['noun.location']:
							label = 'I-MISC'
			
			if data[0] in ['of','for'] and prev_tag == 'I-ORG' and next_pos in ['NNP','NNPS']:
				label = 'I-ORG'
			if prev_word in ['of','for'] and prev_tag == 'I-ORG' and (data[1] == 'NNP' or data[1] == 'NNPS'):
				label = 'I-ORG'
			if prev_tag == 'I-ORG' and data[0] in organization_common_words:
				label = 'I-ORG'
			if data[1] in ['NNP','NNPS'] and next_word in organization_common_words:
				label = 'I-ORG'
			
			#looking into memory of last 25 labels
			item = [item for item in prev_tags if item[0] == data[0]]
			if item:
				if label != item[0][1] and label in ['I-PER','I-ORG'] and item[0][1] in ['I-PER','I-ORG']:
					label = item[0][1]
			
			#last ditch efforts to get a label
			t = list(wn.synsets(data[0]))
			if not t and data[0][0].isupper() and data[1] not in ['NNP','NNPS','IN','DT','PRP','WP','PRP$','TO','WRB','WDT','WP$','POS','PDT','LS','EX','CC','CD'] and len(data[0]) > 3 and not data[0].isupper() and '-' not in data[0]:
				#if 'Person' in google_lookup(data[0]):
				#	label = 'I-PER'
				#	last_ditch.write(data[0] + "\t" + label + "\n")
				#elif 'Organization' in google_lookup(data[0]):
				#	label = 'I-ORG'
				#	last_ditch.write(data[0] + "\t" + label + "\n")
				data1 = line4.split("\t")
				if data[0] == data1[0]:
					label = google_cleanup(data1[1])
					line4 = last_ditch.readline()
			
			if prev_tag == 'I-PER' and data[1] == 'NNP' and label == 'I-ORG':
				label = 'I-PER'
			
			output.write(data[0] + "\t" + label + "\n")
			if line != '\n':
				prev_word = data[0]
				prev_pos = data[1]
				prev_chunk = data[2].strip("\n")
				prev_tag = label
				
			
			else:
				prev_word = 'BoS'
				prev_pos = 'NaN'
				prev_chunk = 'NaN'
				prev_tag = 'NaN'
				output.write("\n")
			
			prev_tags[i] = (data[0],label)
			
			i = i + 1
			if i == 25:
				i = 0
							
			
			#print(featureset)
	output.close()
	file.close()


if __name__=='__main__':	
	#loading the classifier from a pickle file
	f = open('models/classifier-binarizer-glove-train-dev.pickle','rb')
	classifier = pickle.load(f)
	
	#extracting training data, adding features and creating a training data set
	labeled_data = extract_training_data([],"data/CONLL_train.pos-chunk-name","google/train_google.txt","google/train_google_2words.txt","google/train_google_3words.txt")
	
	labeled_data = extract_training_data(labeled_data,"data/CONLL_dev.pos-chunk-name","google/dev_google.txt","google/dev_google_2words.txt","google/dev_google_3words.txt")
	train_set = build_train_set(labeled_data)
	#classifier = nltk.MaxentClassifier.train(train_set,trace = 3,max_iter = 100)
	#pickle.dump(classifier , f)
	f.close()
	#process_test_data(classifier,'CONLL_dev.pos-chunk',"dev_google.txt","dev_google_2words.txt","dev_google_3words.txt","dev_google_last_ditch.txt")
	process_test_data(classifier,'data/CONLL_test.pos-chunk',"google/test_google.txt","google/test_google_2words.txt","google/test_google_3words.txt","test_google_last_ditch.txt")
	
