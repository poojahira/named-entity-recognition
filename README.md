<h3> About</h3>

This is a named entity recogniser created in Python using the Maximum Entropy Classifier in NLTK and trained on the CONLL dataset. The 2003 CoNLL (Conference on Natural Language Learning) corpus uses texts from the Reuters news service. This corpus identifies three types of names: person, organization, and location, and a fourth category, MISC (miscellaneous) for other types of names.[1]

Words chosen to featurize:   
• words with part of speech in [NNP,NNPS]<br />
• words with part of speech in [NN,NNS,JJ] with first letter capitalised<br />
• hyphenated words when either part of word before hyphen or after hyphen has first letter capitalised<br /> 
• if the current word is in [‘of’,’and’,’for’] and previous tag is I-ORG and next word part of speech is in [NNP,NNPS]

<h3>Features extracted</h3>
• current word<br />
• Binarized embeddings (pre-trained Glove 6B 50 dimensions [2]) of current token implemented as per Guo et al 2014 [3]<br />
• prefixes of the current word upto 6 letters<br />
• suffixes of the current word upto six letters lower-case<br /> 
• current word in title case<br /> 
• current word in upper case<br />
• previous word<br />
• next word<br />
• conjunction of previous word and current word<br />
• conjunction of current word and next word<br /> 
• conjunction of previous word, current word and next word<br />
• current word part of speech<br />
• previous word part of speech<br />
• next word part of speech<br />
• conjunction of previous word and current word part of speech<br />
• conjunction of current word and next word part of speech<br />
• conjunction of previous word, current word and next word part of speech current word<br />
• current word chunk<br />
• previous word chunk<br />
• next word chunk<br />
• conjunction of previous word and current word chunk<br />
• conjunction of current word and next word chunk<br />
• conjunction of previous word, current word and next word chunk<br />
• whether current work is hyphenated or not<br />
• whether current word is an ampersand or not<br />
• whether current word is upper case or not<br /> 
• whether current word or previous word or next word is a common organization word like 'Company','Association','Pharmaceuticals','Medical' etc.<br />
• whether current word is the first word of a sentence<br />
• whether current word is the last word of a sentence  
• current word shape<br />
• previous word shape<br />
• next word shape<br />
• conjunction of previous word and current word shape<br />
• conjunction of current word and next word shape<br />
• conjunction of previous word, current word and next word shape<br />
• previous word label<br />
• current word wordnet result<br />
• conjunction of current word and next word wordnet result<br /> 
• conjunction of previous word, current word and next word wordnet result<br />
• current word google knowledge graph result<br />
• current word, next word google knowledge graph result<br />
• current word, next word, next word google knowledge graph result<br />

<h3>Post-processing done on the results based on the following things</h3>
• previous word label<br />
• previous word part of speech<br />
• next word part of speech<br />
• previous 25 words’ labels<br />
• As a last ditch effort, I looked up google knowledge graph for proper nouns which have not been caught by the classifier and were given the ‘O’ label. The results were written to a text file and are in the 'google' folder.

<h3>Gazzeteers used</h3>
• Wordnet from nltk [4]<br />
• Google knowledge graph [5]<br />
The Google API was used to get knowledge graph results pertaining to entity type for current word, current word + next word and current word + next word + next word. Results are stored as text files in a folder called 'google' to run program offline and avoid hitting api limits. The code to extract the knowledge graph details is also in the 'google' folder.<br />

<h3>Models</h3>
Including the model described above, another model was built using the centroids of the Kmeans clustering of word embeddings as a feature. Models are saved as pickle files in folder 'models'.

<h3>References</h3>
[1] https://www.clips.uantwerpen.be/conll2003/ner/<br />
[2] https://nlp.stanford.edu/projects/glove/<br />
[3] http://aclweb.org/anthology/D/D14/D14-1012.pdf<br />
[4] http://www.nltk.org/howto/wordnet.html<br />
[5] https://developers.google.com/knowledge-graph/
