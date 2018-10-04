<h3> About</h3>

This is a named entity recogniser created in Python using the Maximum Entrophy Classifier in NLTK and trained on the CONLL dataset. The 2003 CoNLL (Conference on Natural Language Learning) corpus uses texts from the Reuters news service. This corpus identifies three types of names: person, organization, and location, and a fourth category, MISC (miscellaneous) for other types of names.[1]

Words chosen to featurize:   
• words with part of speech in [NNP,NNPS]  
• words with part of speech in [NN,NNS,JJ] with first letter capitalised  
• hyphenated words when either part of word before hyphen or after hyphen has first letter capitalised  
• if the current word is in [‘of’,’and’,’for’] and previous tag is I-ORG and next word part of speech is in [NNP,NNPS]  

<h3>Features extracted</h3>
• current word  
• Binarized embeddings (pre-trained Glove 6B 50 dimensions [2]) of current token implemented as per Guo et al 2014 [3]  
• prefixes of the current word upto 6 letters  
• suffixes of the current word upto six letters lower-case  
• current word in title case  
• current word in upper case  
• previous word  
• next word  
• conjunction of previous word and current word  
• conjunction of current word and next word  
• conjunction of previous word, current word and next word  
• current word part of speech  
• previous word part of speech  
• next word part of speech  
• conjunction of previous word and current word part of speech  
• conjunction of current word and next word part of speech  
• conjunction of previous word, current word and next word part of speech current word  
• current word chunk  
• previous word chunk  
• next word chunk  
• conjunction of previous word and current word chunk  
• conjunction of current word and next word chunk  
• conjunction of previous word, current word and next word chunk  
• whether current work is hyphenated or not  
• whether current word is an ampersand or not  
• whether current word is upper case or not  
• whether current word or previous word or next word is a common organization word like ‘Company’,’Association’,’Pharmaceuticals’,’Medical’ etc.  
• whether current word is the first word of a sentence  
• whether current word is the last word of a sentence  
• current word shape  
• previous word shape  
• next word shape  
• conjunction of previous word and current word shape  
• conjunction of current word and next word shape  
• conjunction of previous word, current word and next word shape  
• previous word label  
• current word wordnet result  
• conjunction of current word and next word wordnet result  
• conjunction of previous word, current word and next word wordnet result  
• current word google knowledge graph result  
• current word, next word google knowledge graph result  
• current word, next word, next word google knowledge graph result  

<h3>Post-processing done on the results based on the following things</h3>
• previous word label  
• previous word part of speech  
• next word part of speech  
• previous 25 words’ labels  
• As a last ditch effort, I looked up google knowledge graph for proper nouns which have not been caught by the classifier and were given the ‘O’ label. The results were written to a text file and are in the 'google' folder.

<h3>Gazzeteers used</h3>
Wordnet from nltk [4]  
Google knowledge graph [5]  
The Google API was used to get knowledge graph results pertaining to entity type for current word, current word + next word and current word + next word + next word. Results are stored as text files in a folder called 'google' to run program offline and avoid hitting api limits. The code to extract the knowledge graph details is also in the 'google' folder. 

Another model was built using the centroids of the Kmeans clustering of word embeddings as a feature. 

Models are saved as pickle files in folder 'models'.

<h3>References</h3>
[1] https://www.clips.uantwerpen.be/conll2003/ner/  
[2] https://nlp.stanford.edu/projects/glove/  
[3] http://aclweb.org/anthology/D/D14/D14-1012.pdf  
[4] http://www.nltk.org/howto/wordnet.html   
[5] https://developers.google.com/knowledge-graph/

