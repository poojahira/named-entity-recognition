def score (keyFileName, responseFileName):
	keyFile = open(keyFileName, 'r')
	key = keyFile.readlines()
	responseFile = open(responseFileName, 'r')
	response = responseFile.readlines()
	if len(key) != len(response):
		print("length mismatch between key and submitted file")
		sys.exit(0)
	correct = 0
	incorrect = 0
	keyGroupCount = 0
	keyStart = 0
	responseGroupCount = 0
	responseStart = 0
	correctGroupCount = 0
	for i in range(len(key)):
		key[i] = key[i].rstrip('\n')
		response[i] = response[i].rstrip('\n')
		if key[i] == "":
			if response[i] == "":
				continue
			else:
				print("sentence break expected at line " + str(i))
				sys.exit(0)
		keyFields = key[i].split('\t')
		if len(keyFields) != 2:
			print("format error in key at line " + str(i) + ":" + key[i])
			exit()
		keyToken = keyFields[0]
		keyTag = keyFields[1]
		responseFields = response[i].split('\t')
		if len(responseFields) != 2:
			print("format error at line " + str(i))
			exit()
		responseToken = responseFields[0]
		responseTag = responseFields[1]
		if responseToken != keyToken:
			print("token mismatch at line " + str(i))
			exit()
		if responseTag == keyTag:
			correct = correct + 1
		else:
			incorrect = incorrect + 1
                # the previous token ends a group if
                #   we are in a group AND
                #   the current tag is O OR the current tag is a B tag
                #   the current tag is an I tag with a different type from the current group
		responseEnd =  responseStart!=0 and (responseTag=='O' or responseTag[0:1]=='B' or (responseTag[0:1]=='I' and responseTag[2:]!=responseGroupType))
                # the current token begins a group if
                #   the previous token was not in a group or ended a group AND
                #   the current tag is an I or B tag
		responseBegin = (responseStart==0 or responseEnd) and (responseTag[0:1]=='B' or responseTag[0:1]=='I')
		keyEnd =  keyStart!=0 and (keyTag=='O' or keyTag[0:1]=='B' or (keyTag[0:1]=='I' and keyTag[2:]!=keyGroupType))
		keyBegin = (keyStart==0 or keyEnd) and (keyTag[0:1]=='B' or keyTag[0:1]=='I')
		if responseEnd: 
		    responseGroupCount = responseGroupCount + 1
		if keyEnd:
		    keyGroupCount = keyGroupCount + 1
		if responseEnd and keyEnd and responseStart == keyStart and responseGroupType == keyGroupType:
		    correctGroupCount = correctGroupCount + 1
		if responseBegin:
		    responseStart = i
		    responseGroupType = responseTag[2:]
		elif responseEnd:
		    responseStart = 0
		if keyBegin:
		    keyStart = i
		    keyGroupType = keyTag[2:]
		elif keyEnd:
		    keyStart = 0
	print(str(correct) + " out of " + str(correct + incorrect) + " tags correct")
	accuracy = 100.0 * correct / (correct + incorrect)
	print("  accuracy: "+ str(accuracy))
	#print ("groups in key",keyGroupCount)
	#print responseGroupCount, "groups in response"
	#print correctGroupCount, "correct groups"
	precision = 100.0 * correctGroupCount / responseGroupCount
	recall = 100.0 * correctGroupCount / keyGroupCount
	F = 2 * precision  * recall / (precision + recall)
	print("  precision: " + str(precision))
	print("  recall: " + str(recall))
	print("  F1: " + str(F))

if __name__=='__main__':
	score('data/CONLL_dev.name','data/response.name')
