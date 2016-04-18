from nltk.corpus import wordnet as wn
import sys,argparse

# http://stevenloria.com/tutorial-wordnet-textblob/
# http://www.nltk.org/howto/wordnet.html
# http://wordnetweb.princeton.edu/perl/webwn

def definition(word):
    print("Synsets and their definitions")
    syn_sets = wn.synsets(word)
    for synset in syn_sets:
        print("Synset: " + str(synset))
        print("Definition: " + synset.definition())
        print("\n")
    return syn_sets


def hypernyms(syn_sets):
    print("The Hypernyms are")
    print("Synset: " + str(syn_sets[0]))
    print("Hypernyms: " + str(syn_sets[0].hypernyms()))
    print("\n")
    return

def paths(syn_sets):          

    paths = syn_sets[0].hypernym_paths()
    print("Hypernym paths of {} ".format(str(syn_sets[0])))
    print("")
    for i in range(len(paths)):
        print("Path[{}]:".format(i))
        [print(syn.name()) for syn in paths[i]]
        print("")
    return


if __name__ == '__main__':
    

    word = "pen"

    import sys
    sys.stdout = open("wordnet.txt",'w')
    syn_sets = definition(word)
    hypernyms(syn_sets)
    paths(syn_sets)
    sys.stdout = sys.__stdout__
    
    
