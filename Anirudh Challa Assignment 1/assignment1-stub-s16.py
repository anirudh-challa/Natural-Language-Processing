import nltk, zipfile, argparse,pickle,sys
import pandas as pd

def tokenize(documents):
    sentences = [nltk.sent_tokenize(document) for document in documents]

    words = [nltk.word_tokenize(sentence.lower()) for doc in sentences for sentence in doc]

    original_words = [nltk.word_tokenize(sentence) for doc in sentences for sentence in doc]

    corpus_size = 0
    vocab_size = 0
    wrdlist = []
    for wrds in words:
        for wrd in wrds:
            wrdlist.append(wrd.lower())
    
    corpus_size = len(wrdlist)
    vocab_size = len(set(wrdlist))
    return corpus_size,vocab_size,words,sentences,original_words

def pos_tagging(documents,f):
    corpus_tags=[]
    for doc in documents:
        for sentence in doc:
            tokens = nltk.word_tokenize(sentence)
            tags = nltk.pos_tag(tokens)
            sent=""
            for word,tag in tags:
                sent += word + '/'+tag + ' '
            f.write(sent+'\n')
            
            corpus_tags.append(tags)
        f.write('\n')
    return corpus_tags

def frequency(corpus_tags,documents,words,f):
    df = pd.DataFrame()
    
    tagfreq = nltk.FreqDist(tag for tags in corpus_tags for word,tag in tags)
    print ("The most frequent POS tag is: %s" %(tagfreq.most_common(1)))
    
    wordfreq = nltk.FreqDist(word.lower() for doc in words for word in doc)
    
    for fr in wordfreq.most_common():
        f.write("{0}\n".format(fr))
    
    condfreq = nltk.ConditionalFreqDist((tag,word.lower()) for tags in corpus_tags for word,tag in tags)
    
    wrdlist = []
    for wrds in words:
        for wrd in wrds:
            wrdlist.append(wrd.lower())
    vocabulary = set(wrdlist)
    
    for pos in condfreq.conditions():
        word_freq_dist_for_pos = condfreq[pos]
        for word in vocabulary:
            num_times_word_was_pos = word_freq_dist_for_pos.get(word, 0)
            df.loc[pos,word] = num_times_word_was_pos
    print('\n')
    return condfreq,df

    
    
def similarwords(cf,documents):
    tgs = ['NN','VBD','JJ','RB']
    txt = nltk.Text(w.lower() for d in documents for w in d)
    for tg in tgs:
        (wrd,cnt) = cf[tg].most_common(1)[0];
        print (cf[tg].most_common(2));
        print("The most frequent word for POS %s is \"%s\" and its similar words are: " %(tg,wrd))
        txt.similar(wrd)
        print('\n')
        
    print ("The Collocations are:")
    (txt.collocations())
    print ('\n')



###############################################################################
## Utility Functions ##########################################################
###############################################################################
# This method takes the path to a zip archive.
# It first creates a ZipFile object.
# Using a list comprehension it creates a list where each element contains
# the raw text of the fable file.
# We iterate over each named file in the archive:
#     for fn in zip_archive.namelist()
# For each file that ends with '.txt' we open the file in read only
# mode:
#     zip_archive.open(fn, 'rU')
# Finally, we read the raw contents of the file:
#     zip_archive.open(fn, 'rU').read()
def unzip_corpus(input_file):
    zip_archive = zipfile.ZipFile(input_file)
    contents = [zip_archive.open(fn, 'rU').read().decode('utf-8') for fn in zip_archive.namelist() if fn.endswith(".txt")]
    return contents

###############################################################################
## Stub Functions #############################################################
###############################################################################
def process_corpus(corpus_name):
    input_file = corpus_name + ".zip"
    corpus_contents = unzip_corpus(input_file)
    
    #part 1
    corpus_size,vocab_size,words,sentences,initial_words = tokenize(corpus_contents)
    print ("The total number of words in the corpus are %d" %(corpus_size))
    print ("The vocab size of the corpus is %d" %(vocab_size))
    
    #part 2
    f = open(corpus_name+'-pos.txt', 'w')
    tags = pos_tagging(sentences,f)
    f.close()
    
    #part 3
    f = open(corpus_name+'-word-freq.txt', 'w')
    #cf = frequency(tags,sentences,words,f)
    cf,df = frequency(tags,sentences,words,f)
    f.close()
    
    df.to_csv(corpus_name+'-pos-word-freq.txt',header=True)
    df.to_csv(corpus_name+'-pos-word-freq.csv',header=True)
    

    
    #part 4
    similarwords(cf,initial_words)
    
    # Your code goes here
    pass

###############################################################################
## Program Entry Point ########################################################
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assignment 1')
    parser.add_argument('--corpus', required=True, dest="corpus", metavar='NAME',  help='Which corpus to process {fables, blogs}')

    args = parser.parse_args()
    
    corpus_name = args.corpus
    
    if corpus_name == "fables" or "blogs":
        print("The name of the corpus is " + corpus_name + '\n')
        process_corpus(corpus_name)
    else:
        print("Unknown corpus name: {0}".format(corpus_name))