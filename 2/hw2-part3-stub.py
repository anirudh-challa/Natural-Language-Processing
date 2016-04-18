import re, nltk

def get_score(review):
    return int(re.search(r'Overall = ([1-5])', review).group(1))

def get_text(review):
    return re.search(r'Text = "(.*)"', review).group(1)

def collocations(merged_pos_sentences,merged_neg_sentences):
    pos_text = nltk.Text(word for word in merged_pos_sentences)
    neg_text = nltk.Text(word for word in merged_neg_sentences)
    
    print("\nWe find Positive Collocations\n")
    pos_text.collocations()
    print("\nWe find Negative Collocations\n")
    neg_text.collocations()

def process_reviews(file_name):
    file = open(file_name, "rb")
    raw_data = file.read().decode("latin1")
    file.close()

    pos_texts = []
    neg_texts = []
    first_sent = None
    for review in re.split(r'\.\n', raw_data):
        overall_score = get_score(review)
        review_text = get_text(review)
        if overall_score > 3:
            pos_texts.append(review_text)
        elif overall_score < 3:
            neg_texts.append(review_text)
        if first_sent == None:
            sent = nltk.sent_tokenize(review_text)
            if (len(sent) > 0):
                first_sent = sent[0]
                
    
    print("Doing: Lowering case of text")
    pos_texts = [text.lower() for text in pos_texts]
    neg_texts = [text.lower() for text in neg_texts]
          
    print("Doing: Removing stop words from text")
    pos_tokens = [nltk.sent_tokenize(text) for text in pos_texts]
    neg_tokens = [nltk.sent_tokenize(text) for text in neg_texts]
    pos_words = [nltk.word_tokenize(sentence) for text in pos_tokens for sentence in text]
    neg_words = [nltk.word_tokenize(sentence) for text in neg_tokens for sentence in text]

    
    stopwords = nltk.corpus.stopwords.words("english")
    pos_without_stop = []
    neg_without_stop = []
    
  
    for sentence in pos_words:
        sent = []
        for word in sentence:
            c = re.search("\w",word)
            if(word not in stopwords and c!=None):
                sent.append(word)
        pos_without_stop.append(sent)
    

        
    for sentence in neg_words:
        sent = []
        for word in sentence:
            c = re.search("\w",word)
            if(word not in stopwords and c!=None):
                sent.append(word)
        neg_without_stop.append(sent)
    

    pos_freq = nltk.FreqDist(word for sentence in pos_without_stop for word in sentence)
    neg_freq = nltk.FreqDist(word for sentence in neg_without_stop for word in sentence)
    

    merged_pos_sentences = []
    for sentence in pos_without_stop:
        merged_pos_sentences = merged_pos_sentences + sentence
        
    merged_neg_sentences = []
    for sentence in neg_without_stop:
        merged_neg_sentences = merged_neg_sentences + sentence    
    pos_bigram_freq = nltk.FreqDist(condword+' '+ word for (condword,word)in nltk.bigrams(merged_pos_sentences))
    neg_bigram_freq = nltk.FreqDist(condword+' '+ word for (condword,word)in nltk.bigrams(merged_neg_sentences))
    
    pos_cond_freq = nltk.ConditionalFreqDist((condword,word) for (condword,word)in nltk.bigrams(merged_pos_sentences))
    neg_cond_freq = nltk.ConditionalFreqDist((condword,word) for (condword,word)in nltk.bigrams(merged_neg_sentences))
    
    
    
    category = {'positive': pos_freq, 'negative' : neg_freq}
    for key in category.keys():
        s=""
        for word,freq in category[key].most_common():
            s += word + ' ' + str(freq)+'\n'
        
        print("Writing " + key+'-unigram-freq.txt')
        write_file(key+'-unigram-freq.txt',s)
    
    category = {'positive': pos_bigram_freq, 'negative' : neg_bigram_freq}
    for key in category.keys():
        s=""
        for word,freq in category[key].most_common():
            s += word + ' ' + str(freq)+'\n'
        
        print("Writing " + key+'-bigram-freq.txt')
        write_file(key+'-bigram-freq.txt',s)
    
    
    category = {'positive': pos_freq, 'negative' : neg_freq}
    for key in category.keys():
        
        print("\nThe most frequent unigrams for {0} reviews:-".format(key))
        for word,freq in category[key].most_common(10):
            print (word + ' ' + str(freq))
            
    category = {'positive': pos_bigram_freq, 'negative' : neg_bigram_freq}
    for key in category.keys():
        print("\nThe most frequent bigrams for {0} reviews:".format(key))
        for word,freq in category[key].most_common(10):
            print (word + ' ' + str(freq))
        
    
    collocations(merged_pos_sentences,merged_neg_sentences)
    
    
    print("\nDoing: Probability of the first sentence without the stop words\n")
    print('The first line is '+'\"'+' '.join(pos_without_stop[0])+'\"')
    print("P(first line) = P(excellent)P(restaurant|excellent)")
    print("P(excellent) = " + str(pos_freq.freq('excellent')))
    print("P(restaurant|excellent) = " + str(pos_cond_freq['excellent'].freq('restaurant')))
    print("P(first line) =",pos_freq.freq('excellent')*pos_cond_freq['excellent'].freq('restaurant'))
    
    

    print("\nDoing: Probability of the first sentence with the stop words\n")
    print('The first line is '+'\"'+' '.join(pos_words[0])+'\"')
    print("P(first line) = P(an)P(excellent)P(restaurant|an excellent)P(.|excellent restaurant)\n")
    print("This uses trigram model(Second order)  of Markov Assumption")
    

    print("\nDoing: P(mashed U potatoes)\n")
    print("P(mashed U potatoes) = P(mashed) + P(potatoes) - P(mashed^potatoes)")
    print("P(mashed) = ", pos_freq.freq('mashed'))
    print("P(potatoes) = ", pos_freq.freq('potatoes'))
    print("\nAssuming Probability of a word being 'mashed' or 'potatoes' is independent of each other")
    p1 = pos_freq.freq('mashed')*pos_freq.freq('potatoes')
    print("P(mashed^potatoes) = P(mashed)*P(potatoes) = ",p1 )
    print("\nAssuming Probability of a word being 'mashed' or 'potatoes' is not independent of each other")
    p2 = pos_cond_freq['mashed'].freq('potatoes')*pos_freq.freq('mashed')
    print("P(mashed^potatoes) = P(potatoes|mashed)*P(mashed) = ",p2)
    print("\nP(mashed U potatoes) = %f or %f" %(pos_freq.freq('mashed') + pos_freq.freq('potatoes') - p1,pos_freq.freq('mashed') + pos_freq.freq('potatoes') - p2))
    
    
    print("If we find a word that is not in our frequency tables, then the probobality of the sentence becomes 0." )
    print ("To overcome this we use Additive smoothing in which we add 1 to the count of every word in the vocabulary, so that no word has 0 probability.")
    return merged_pos_sentences,merged_neg_sentences
    
# Write to File, this function is just for reference, because the encoding matters.
def write_file(file_name, data):
    file = open(file_name, 'w', encoding="utf-8")    # or you can say encoding="latin1"
    file.write(data)
    file.close()

if __name__ == '__main__':
    #filename = sys.argv[1]
    file_name = "restaurant-training.data"
    import sys
    sys.stdout = open("language-model-answers.txt",'w')
    merged_pos_sentences,merged_neg_sentences = process_reviews(file_name)
    sys.stdout = sys.__stdout__
    collocations(merged_pos_sentences,merged_neg_sentences)
