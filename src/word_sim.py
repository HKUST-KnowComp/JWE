# -*- coding: UTF-8 -*-
'''
Test the embeddings on word similarity task
Select 240.txt 297.txt 
'''
import numpy as np
import pdb
import sys,getopt
from scipy.stats import spearmanr
def build_dictionary(word_list):
        dictionary = dict()
        cnt = 0
        for w in word_list:
                dictionary[w] = cnt
                cnt += 1
        return dictionary
def read_wordpair(sim_file):
        f1 = open(sim_file, 'r')
        pairs = []
        for line in f1:
                pair = line.split()
                pair[2] = float(pair[2])
                pairs.append(pair)
        f1.close()
        return pairs
def read_vectors(vec_file):
        # input:  the file of word2vectors
        # output: word dictionay, embedding matrix -- np ndarray
        f = open(vec_file,'r')
        cnt = 0
        word_list = []
        embeddings = []
        word_size = 0
        embed_dim = 0
        for line in f:
                data = line.split()
                if cnt == 0:
                        word_size = data[0]
                        embed_dim = data[1]
                else:
                        word_list.append(data[0])
                        tmpVec = [float(x) for x in data[1:]]
                        embeddings.append(tmpVec)
                cnt = cnt + 1
        f.close()
        embeddings = np.array(embeddings)
        for i in range(int(word_size)):
                embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])
        dict_word = build_dictionary(word_list)
        return word_size, embed_dim, dict_word, embeddings
if  __name__ == '__main__':
        fname1 = 'evaluation/297.txt'
        vec_file = 'bin/zh_wiki_bin.txt'
        try:
                opts, args = getopt.getopt(sys.argv[1:],"hs:e:",["similarity_file=","embed_file="])
        except getopt.GetoptError:
                print ('word_sim.py -s <similarity_file> -e <embed_file>')
                sys.exit(2)
        for opt, arg in opts:
                if opt == '-h':
                        print ('word_sim.py -a <similarity_file> -e <embed_file>')
                        sys.exit()
                elif opt in ("-s", "--similarity_file"):
                        fname1 = arg
                elif opt in ("-e", "--embed_file"):
                        vec_file = arg
        pairs = read_wordpair(fname1)
        word_size, embed_dim, dict_word, embeddings = read_vectors(vec_file)
        human_sim = []
        vec_sim = []
        cnt = 0
        total = len(pairs)
        for pair in pairs:
                w1 = pair[0]
                w2 = pair[1]
                #w1 = w1.decode('utf-8')
                #w2 = w2.decode('utf-8')
                if w1 in dict_word and w2 in dict_word:
                        cnt += 1
                        id1 = dict_word[w1]
                        id2 = dict_word[w2]
                        vsim = embeddings[id1].dot(embeddings[id2].T) / (np.linalg.norm(embeddings[id1]) * np.linalg.norm(embeddings[id2]))
                        human_sim.append(pair[2])
                        vec_sim.append(vsim)
        print(cnt, ' word pairs appered in the training dictionary , total word pairs ', total)
        score = spearmanr(human_sim, vec_sim)
        print(score)