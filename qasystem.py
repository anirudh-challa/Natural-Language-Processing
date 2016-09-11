#Kongposh author

import zipfile, argparse, os, nltk, operator, re, sys
from nltk.corpus import wordnet as wn
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.parse import DependencyGraph
from nltk.stem.lancaster import LancasterStemmer
from lib2to3.pytree import Node

###############################################################################
## Utility Functions ##########################################################
###############################################################################
def read_file(filename):
	fh = open(filename, 'r')
	text = fh.read()
	fh.close()
	return text
# This method takes as input the file extension of the set of files you want to open
# and processes the data accordingly
# Assumption: this python program is in the same directory as the training files
def getData(file_extension):
	dataset_dict = {}
	# iterate through all the files in the current directory
	for filename in os.listdir("."):
		if filename.endswith(file_extension):
			# get stories and cumulatively add them to the dataset_dict
			if file_extension == ".story" or file_extension == ".sch":
				dataset_dict[filename[0:len(filename) - len(file_extension)]] = open(filename, 'rU').read()
			# question and answer files and cumulatively add them to the dataset_dict
			elif file_extension == ".answers" or file_extension == ".questions":
				getQA(open(filename, 'rU', encoding="latin1"), dataset_dict)
	return dataset_dict
# returns a dictionary where the question numbers are the key
# and its items are another dict of difficulty, question, type, and answer
# e.g. story_dict = {'fables-01-1': {'Difficulty': x, 'Question': y, 'Type':}, 'fables-01-2': {...}, ...}
def getQA(content, dataset_dict):
	qid = ""
	for line in content:
		if "QuestionID: " in line:
			qid = line[len("QuestionID: "):len(line) - 1]
			# dataset_dict[qid] = defaultdict()
			dataset_dict[qid] = {}
		elif "Question: " in line:
			dataset_dict[qid]['Question'] = line[len("Question: "):len(line) - 1]
		elif "Answer: " in line:
			dataset_dict[qid]['Answer'] = line[len("Answer:") + 1:len(line) - 1]
		elif "Difficulty: " in line:
			dataset_dict[qid]['Difficulty'] = line[len("Difficult: ") + 1:len(line) - 1]
		elif "Type: " in line:
			dataset_dict[qid]['Type'] = line[len("Type:") + 1:len(line) - 1]
	return dataset_dict
###############################################################################
## Question Answering Functions ###############################################
###############################################################################
def lemming(word, tag):
	if tag in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'):
		pos = 'v'
	elif tag in ('RB', 'RBR', 'RBS'):
		pos = 'r'
	elif tag in ('JJ', 'JJR', 'JJS'):
		pos = 'a'
	else:
		pos = 'n'
	return WordNetLemmatizer().lemmatize(word, pos)
def filefind(currentq, question):
	borfile = currentq[0] + "-" + currentq[1]
	schors = question[1]["Type"]
	if schors in "Sch":
		borfile = borfile + ".sch"
	else:
		borfile = borfile + ".story"
	return borfile
def get_lines(text, stopwords):
	lines = nltk.sent_tokenize(text)
	lines = [nltk.word_tokenize(sent) for sent in lines]
	res = []
	for line in lines:
		data = []
		for token in line:
			token_lower = token.lower()
			if token_lower not in stopwords and is_word(token_lower):
				data.append(token_lower)
		res.append(data)
	return res
def get_semantic_line(text):
	lines = nltk.sent_tokenize(text)
	lines = [nltk.word_tokenize(sent) for sent in lines]
	res = []
	for line in lines:
		data = get_semantic_words(line)
		res.append(data)
	res = [nltk.pos_tag(sent) for sent in res]
	return res
def get_original_lines(text):
	line = nltk.sent_tokenize(text)
	line = [nltk.word_tokenize(sent) for sent in line]
	return line
def get_line(text, stopwords):
	line = nltk.sent_tokenize(text)
	line = [nltk.word_tokenize(sent) for sent in line]
	line = [nltk.pos_tag(sent) for sent in line]
	res = get_ans(line, stopwords)
	return res
def is_word(word):
	return re.match(r'\b\w+\b', word)
def get_ans(line, stopwords):
	ans = []
	for tagged_tokens in line:
		res = []
		for token in tagged_tokens:
			token_lower = token[0].lower()
			if token_lower not in stopwords and is_word(token_lower):
				tag = token[1]
				token_lemma = lemming(token_lower, tag)
				token_stem = nltk.stem.SnowballStemmer('english').stem(token_lemma)
				res.append(token_stem)
		ans.append(set(res))
	return ans
def get_phrase(tgtoks, qans):
	for i in range(len(tgtoks) - 1, 0, -1):
		word = (tgtoks[i])[0]
		if word in qans:
			return tgtoks[i + 1:]
# Here we use the other functions to find the best possible asnwer for highest recall
def bestanswer(qans, line, stopwords):
	answers = []
	i = 0
	for sent in line:
		sans = get_ans(sent, stopwords)
		overlap = len(qans & sans)
		answers.append((overlap, sent, i))
		i = i + 1
	answers = sorted(answers, key=operator.itemgetter(0), reverse=True)
	best_answer = (answers[0])[1]
	index = (answers[0])[2]
	return (best_answer, index)
def write_file(kvpair, file, filename):#filename is the input file
	lines = read_file(filename).split('\n')
	for line in lines:
		for dic in kvpair:
			tup = [entry.split('-') for entry in dic]
			for forb in set([arr[0] for arr in tup]):
				for story_id in sorted(set([arr[1] for arr in tup if arr[0] == forb])):
					questions = sorted([arr for arr in tup if arr[0] == forb and arr[1] == story_id],
					                   key=lambda arr: int(arr[2]))
					questions = ['-'.join(arr) for arr in questions]
					ans = [(question, dic[question]) for question in questions]
					if ans[0][0].startswith(line):
						file.write(
							'\n\n'.join(
								['\n'.join(['QuestionId:{0}'.format(a[0]), 'Answer:{0}'.format(a[1])]) for a in
								 ans]))
						file.write('\n\n')

def create_filename(parseCurrQID, question):
	currFileName = parseCurrQID[0] + "-" + parseCurrQID[1]
	currType = question[1]["Type"]
	if currType == "Story":
		currFileName = currFileName + ".story"
	else:
		currFileName = currFileName + ".sch"
	return currFileName
def read_dep_questions(fh):
	dep_lines = []
	qid = None
	for line in fh:
		line = line.strip()
		if len(line) == 0:
			res = "\n".join(dep_lines)
			return (qid, res)
		elif re.match(r"^QuestionId:\s+(.*)$", line):
			m = re.match(r"^QuestionId:\s+(.*)$", line)
			qid = m.group(1)
			continue
		dep_lines.append(line)
	res = "\n".join(dep_lines) if len(dep_lines) > 0 else None
	return (qid, res)
def read_dep_parses_questions(depfile):
	fh = open(depfile, 'r')
	# list to store the results
	graphs = {}
	# Read the lines containing the first parse.
	res = read_dep_questions(fh)
	dep = res[1]
	qid = res[0]
	# While there are more lines:
	# 1) create the DependencyGraph
	# 2) add it to our list
	# 3) try again until we're done
	while dep is not None:
		graph = DependencyGraph(dep)
		graphs[qid] = graph
		res = read_dep_questions(fh)
		qid = res[0]
		dep = res[1]
	fh.close()
	return graphs
def read_dep_parses(depfile):
	fh = open(depfile, 'r')
	# list to store the results
	graphs = []
	# Read the lines containing the first parse.
	dep = read_dep(fh)
	# While there are more lines:
	# 1) create the DependencyGraph
	# 2) add it to our list
	# 3) try again until we're done
	while dep is not None:
		graph = DependencyGraph(dep)
		graphs.append(graph)
		dep = read_dep(fh)
	fh.close()
	return graphs
def read_dep(fh):
	dep_lines = []
	for line in fh:
		line = line.strip()
		if len(line) == 0:
			return "\n".join(dep_lines)
		elif re.match(r"^QuestionId:\s+(.*)$", line):
			continue
		dep_lines.append(line)
	return "\n".join(dep_lines) if len(dep_lines) > 0 else None
def find_main(graph):
	for node in graph.nodes.values():
		if node['rel'] == 'ROOT':
			return node
	return None
def find_node(word, graph):
	for node in graph.nodes.values():
		if node["word"] == word:
			return node
	return None
def get_dependents(node, graph, level):
	results = []
	count = 0;
	for item in node["deps"]:
		if (count <= level):
			address = node["deps"][item][0]
			dep = graph.nodes[address]
			results.append(dep)
			results = results + get_dependents(dep, graph, level - 1)
	return results
def get_comparison_tag(qgraph):
	comparison_tag = ""
	dict = {"Where": ["nmod", "prep"], "Who": ["nsubj"], "When": ["prep"]}  # ,"What":"NN"}
	for node in qgraph.nodes.values():
		if (node["word"] in dict):
			comparison_tag = dict[node["word"]]
			return comparison_tag
	return ["ROOT"]
def find_root_word(graph):
	for node in graph.nodes.values():
		if node['rel'] == 'ROOT':
			return node["word"]
	return None
def is_same_word(qword, sword):
	qwords = get_level2_words(qword)
	swords = get_level2_words(sword)
	for word in qwords:
		if word in swords:
			return True
	return False
def get_most_probable_sentence(question_data, data):
	max_value = 0
	count = 0
	result = None
	index = 0
	qwords = question_data[0]
	for data_sent in data:
		count = 0
		for sword in data_sent:
			for qword in qwords:
				if qword == sword:
					count = count + 1
				elif sword[:1] == qword[:1] and is_syntatically_similar(sword, qword):
					count = count + 1
		if (count > max_value):
			max_value = count
			result = (data_sent, index)
		index = index + 1
	return result
def stem(word):
	token_stem = nltk.stem.SnowballStemmer('english').stem(word)
	return token_stem
def find_main_ref_word(qgraph, sgraph):
	qmain = find_main(qgraph)
	smain = find_main(sgraph)
	orig_qword = qmain["word"]
	qword = qmain["word"]
	sword = smain["word"]
	qword = stem(lemming(qword, "VB"))
	is_same = is_same_word(qword, sword)
	if is_same:
		return smain
	for node in sgraph.nodes.values():
		tag = node["tag"]
		if tag != None and "VB" in tag:
			orig_sword = node["word"]
			word = stem(lemming(node["word"], "VB"))
			if qword == word:
				return node
			elif is_same_word(orig_qword, orig_sword):
				return node
	return smain
def find_answer(qgraph, sgraph, qtext):
	comparison_tag = get_comparison_tag(qgraph)
	qmain = find_main(qgraph)
	smain = find_main(sgraph)
	qword = qmain["word"]
	sword = smain["word"]
	ref_main = find_main_ref_word(qgraph, sgraph)
	ref_word = ref_main['word']
	# write code to check similarity
	is_same = is_same_word(qword, sword)
	# find_node(qword, sgraph)
	# print("SNODE AFTER IS " + str(snode))
	res = sgraph.nodes.values()
	resList = list(res)
	answer = ""
	deps = []
	flag = False
	shortlisted = []
	for i in range(len(resList)):
		node = resList[i]
		tag = node['tag']
		deps = []
		if str(comparison_tag) in str(tag):
			deps.append(node)
			deps = deps + get_dependents(node, sgraph, 2)
			deps = sorted(deps, key=operator.itemgetter("address"))
			answer = " ".join(dep["word"] for dep in deps)
			shortlisted.append((node, answer))
	qdata = []
	fdata = []
	for line in shortlisted:
		data = get_lower_case_data(line[1])
		fdata.append((line[0], data))
	qdata = get_lower_case_data(qtext)
	res = []
	for data in fdata:
		(node, parents) = get_parents(data[0], sgraph)
		parents = sorted(parents, key=operator.itemgetter("address"))
		str_parents = " ".join(parent["word"] for parent in parents)
		res.append((node, str_parents))
	# list of all possible answers in res
	# if is_same is true then calculate the closest score to root verb and thats the answer else find most common words with question else return the entire asner
	min_score = 1001;
	res_node = None
	answer = None
	if True:
		for tuple in res:
			verb = tuple[0]
			score = get_heads_score(verb, ref_word, sgraph)
			if score == min_score and verb['rel'] in "ccomp":
				min_score = score
				res_node = verb
			elif score < min_score:
				min_score = score
				res_node = verb
		answer = res_node["word"]
	'''else:
		 res_node = get_most_probable_sentence(qdata, fdata)
		 i = res_node[1]
		 answer =  str(shortlisted[i][1])'''
	return str(answer)
def get_parents(node, sgraph):
	results = []
	results.append(node)
	heads = get_heads(node, sgraph)
	head_node = heads[0]
	deps = heads[1]
	results = results + deps
	data = (head_node, results)
	return data
def get_heads(node, graph):
	res = []
	tag = node["tag"]
	if "VB" in tag:
		return (node, res)
	head_index = node["head"]
	next_node = None
	for n in graph.nodes.values():
		i = n["address"]
		if (i == head_index):
			next_node = n
			res.append(n)
			break
	heads = get_heads(next_node, graph)
	head_node = heads[0]
	res = res + heads[1]
	data = (head_node, res)
	return data
def get_heads_score(node, sword, graph):
	count = 0
	if node == None:
		return 1000
	word = node["word"]
	if word == sword:
		return count
	head_index = node["head"]
	next_node = None
	for n in graph.nodes.values():
		i = n["address"]
		if (i == head_index):
			next_node = n
			break
	count = get_heads_score(next_node, sword, graph) + 1
	return count
# res = res + get_heads(next_node, graph)
def find_answer_old(qgraph, sgraph):
	'''comparison_tag = get_comparison_tag(qgraph)
	qmain = find_main(qgraph)
	smain = find_main(sgraph)
	qword = qmain["word"]
	sword = smain["word"]
	snode = smain #find_node(qword, sgraph)'''
	# print("SNODE AFTER IS " + str(snode))
	comparison_tag = get_comparison_tag(qgraph)
	qmain = find_main(qgraph)
	smain = find_main(sgraph)
	qword = qmain["word"]
	sword = smain["word"]
	ref_main = find_main_ref_word(qgraph, sgraph)
	ref_word = ref_main['word']
	snode = smain
	res = sgraph.nodes.values()
	resList = list(res)
	answer = ""
	deps = []
	flag = False
	for i in range(len(resList)):
		node = resList[i]
		if node.get('head', None) == ref_main["address"]:
			if "ROOT" in comparison_tag:
				if flag is False:
					deps.append(ref_main)
					flag = True
				deps.append(node)
				deps = deps + get_dependents(node, sgraph, 2)
				deps = sorted(deps, key=operator.itemgetter("address"))
			elif node['rel'] in comparison_tag:
				deps.append(node)
				deps = deps + get_dependents(node, sgraph, 2)
				deps = sorted(deps, key=operator.itemgetter("address"))
			# answer = answer + " ".join(dep["word"] for dep in deps)
	if len(deps) == 0 and ref_main["address"] != 0:
		for i in range(len(resList)):
			node = resList[i]
			if node.get('head', None) == snode["address"]:
				if "ROOT" in comparison_tag:
					if flag is False:
						deps.append(snode)
						flag = True
						deps.append(node)
						deps = deps + get_dependents(node, sgraph, 2)
						deps = sorted(deps, key=operator.itemgetter("address"))
				elif node['rel'] in comparison_tag:
					deps.append(node)
					deps = deps + get_dependents(node, sgraph, 2)
					deps = sorted(deps, key=operator.itemgetter("address"))
				# answer = answer + " ".join(dep["word"] for dep in deps)
	answer = " ".join(dep["word"] for dep in deps)
	return answer
def get_lower_case_data(text):
	ls = LancasterStemmer()
	sentences = get_sentences(text)
	tokens = get_tokens(sentences)
	result = []
	for token_sent in tokens:
		result = result + ([ls.stem(word.lower()) for word in token_sent])
	return result
def get_normalized_data(text):
	sentences = get_sentences(text)
	tokens = get_tokens(sentences)
	posdata = get_postagged_data(tokens)
	return posdata
def get_sentences(text):
	sentences = nltk.sent_tokenize(text)
	return sentences
def get_tokens(sentences):
	tokens = []
	for sentence in sentences:
		tokens_sent = nltk.word_tokenize(sentence)
		tokens.append(tokens_sent)
	return tokens
def get_postagged_data(token_sentences):
	postagged = []
	for token_sent in token_sentences:
		pos = nltk.pos_tag(token_sent)
		postagged.append(pos)
	return postagged
def get_level2_words(word):
	bad_synsets = wn.synsets(word)
	bad_words_set = set()
	bad_words = []
	for synset in bad_synsets:
		bad_words = bad_words + synset.lemma_names()
	bad_words_set = set(bad_words)
	return (bad_words_set)
def get_semantic_words(words):
	res = []
	for word in words:
		res = res + list(get_level2_words(word))
	return res
def is_syntatically_similar(word1, word2):
	word1_syn = wn.synsets(word1)
	word2_syn = wn.synsets(word2)
	for syn1 in word1_syn:
		for syn2 in word2_syn:
			val = syn1.path_similarity(syn2)
			if val == 1:
				return True;
	return False

def main(filename):
	# optional functions for opening and organizing some of the data
	# if you do not understand how the data is being returned,
	# you can write your own methods; these are to help you get started
	# str(get_level2_words("felt"))
	sch = getData(".sch")  # returns a list of scheherazade realizations
	questions = getData(".questions")  # returns a dict of questionIds
	answers = getData(".answers")  # returns a dict of questionIds
	stopwords = set(nltk.corpus.stopwords.words("english"))
	fabop = {}
	blogop = {}
	for question in questions.items():
		qid = question[0]
		currentq = question[0].split("-")

		borfile = filefind(currentq, question)
		currFileName = borfile
		ques = question[1]["Question"]
		text = read_file(borfile)
		question_lines = get_line(ques, stopwords)
		text_lines = get_line(text, stopwords)
		res = get_most_probable_sentence(question_lines, text_lines)
		if res == None:
			index = 0
		else:
			index = res[1]
		original_sentence_words = get_original_lines(text)[index]
		sentence = " ".join(t for t in original_sentence_words)
		answer = sentence
		# question_words = get_semantic_line(ques)
		# qans = get_ans(question_words[0], stopwords)
		# print("qans is: " +str(qans))
		# parseCurrQID = question[0].split("-")
		# currFileName = create_filename(parseCurrQID, question)
		# answer = bestanswer(qans, line, stopwords)
		# index = answer[1]
		# fans = " ".join(t[0] for t in answer[0] if t not in stopwords)

		name = currFileName.split(".")[0]
		qgraphs = read_dep_parses_questions(name + ".questions.dep")
		sgraphs = None
		if "sch" in currFileName:
			sgraphs = read_dep_parses(name + ".sch.dep")
		else:
			sgraphs = read_dep_parses(name + ".story.dep")
		qdata = qgraphs[qid]
		sdata = sgraphs[index]
		if 'Who' in ques and 'about' in ques:
			answer = about_answer(read_dep_parses(name + ".sch.dep"))
		else:
			answer = find_answer_old(qdata, sdata)
		if currentq[0] == "blogs":
			blogop.update({question[0]: answer})
		else:
			fabop.update({question[0]: answer})

	file = open("Challa_Maheshwari_Sapru_answers.txt", 'w', encoding="utf-8")
	write_file([fabop, blogop], file, filename)
	file.close()

def about_answer(sgraphs):
	subject_words = {}
	subject_word_tags = ['nmod', 'dobj', 'nsubj', 'acl:relcl']
	for sgraph in sgraphs:
		for node in sgraph.nodes.values():
			if node['rel'] in subject_word_tags:
				# root_word =stem(lemming(node['word'], 'n'))
				root_word = node['word']
				subject_words[root_word] = subject_words.get(root_word, 0) + 1
	sorted_items = sorted(subject_words.items(), key=operator.itemgetter(1), reverse=True)
	if len(sorted_items) == 0:
		return None
	elif len(sorted_items) == 1:
		return sorted_items[0][0]
	else:
		return sorted_items[0][0] + " and " + sorted_items[1][0]


###############################################################################
## Program Entry Point ########################################################
###############################################################################
if __name__ == '__main__':
	filename = sys.argv[1]
	main(filename)