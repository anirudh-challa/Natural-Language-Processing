import argparse, re, nltk

# https://docs.python.org/3/howto/regex.html
# https://docs.python.org/3/library/re.html
# https://www.debuggex.com/

def get_words(pos_sent):

    # Your code goes here
    sen = re.sub(r'/[^\s]+','',pos_sent)
    return sen
    pass


def get_noun_phrase(pos_sent):
    np_words_with_tags = re.findall('(?:(?:\S+\/DT )?(?:\S+\/JJ )*(?:\S+\/NN.? )*(?:\S+\/NN.?))',pos_sent)
    noun_phrases = [get_words(phrase) for phrase in np_words_with_tags]
    return noun_phrases
    



def most_freq_noun_phrase(pos_sent_fname):
    f = open(pos_sent_fname,'r')
    doc =""
    docs=[]
    for line in f:
        if(line=='\n'):
            docs.append(doc)
            doc=""
        else:
            doc = doc + (line.strip('\n'))
    f.close()
    
    dn = 1
    for doc in docs:
        noun_phrases = get_noun_phrase(doc)
        frq = nltk.FreqDist(phrase.lower() for phrase in noun_phrases)
        print("\nThe most freq NP in document[{0}]: {1}".format(dn,frq.most_common(3)))
        dn+=1
    return


    pass

if __name__ == '__main__':

    # python hw2-part1-stub.py -f fables-pos.txt
    # python hw2-part1-stub.py -f blogs-pos.txt
    parser = argparse.ArgumentParser(description='Assignment 2')
    parser.add_argument('-f', dest="pos_sent_fname", default="fables-pos.txt",  help='File name that contant the POS.')

    args = parser.parse_args()
    pos_sent_fname = args.pos_sent_fname

    text = 'Less/RBR than/IN 1/2/CD of/IN all/DT US/NNP businesses/NNS are/VBP sole/JJ proprietorships/NNS ?/.'
    print(text)
    print(str(get_words(text)))
    print(str(get_noun_phrase(text)))

    most_freq_noun_phrase(pos_sent_fname)
    
