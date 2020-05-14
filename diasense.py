from words_to_bert import word_embeddings, get_sentences
from bert_serving.client import BertClient
from scipy.spatial.distance import cosine
import numpy as np
import os
from docopt import docopt
import re

def main():
    args = docopt('''Compute BERT embeddings for target words and calculate change metrics.

    Usage:
        diasense.py <language> <corpus1> <corpus2> <targets> <result_path> <sent_limit>

        <language> = english, german, swedish or latin
        <corpus1> = path to corpus1 (txt-file)
        <corpus2> = path to corpus2 (txt-file)
        <targets> = path to target words (txt-file)
        <result_path> = path to directory for results
        <sent_limit> = number of sentences considered in calculation of BERT embeddings in each corpus
        
    ''')

    language = args['<language>']    
    corpus1 = args['<corpus1>']
    corpus2 = args['<corpus2>']
    targets = args['<targets>']
    result_path = args['<result_path>']
    sent_limit = int(args['<sent_limit>'])
                    
    with open(targets, 'r') as target_in:
        target_list = [line.rstrip() for line in target_in]
    
    metrics_path = result_path + '/metrics'
        
    if not os.path.exists(metrics_path):
        os.mkdir(metrics_path)       

    metrics = metrics_path + '/metrics.txt'
    delta_later_out = metrics_path + '/delta_later.txt'
    delta_compare_out = metrics_path + '/delta_compare.txt'
        
    metrics = open(metrics, 'w')
    delta_later_out = open(delta_later_out, 'w')
    delta_compare_out = open(delta_compare_out, 'w')
    
    metrics.write('TARGET\tEARLIER\tEARLIER_STD\tLATER\tLATER_STD\tCOMPARE\tCOMPARE_MIXED\tDELTA_LATER\tDELTA_COMPARE\n')

    with open(corpus1, 'r') as corpus:
        corpus1 = [line.rstrip() for line in corpus]
        
    with open(corpus2, 'r') as corpus:
        corpus2 = [line.rstrip() for line in corpus]

    
    #get BERT sentence encoder; bert-as-service should be started beforehand (separately in your terminal)
    #recommendation: bert-serving-start -pooling_strategy NONE -show_tokens_to_client -model_dir multi_cased_L-12_H-768_A-12 -max_seq_len=128
    bc = BertClient(check_length=False)    
        
    for target in target_list:
        
        #get sentences in which target occurs
        sentences_c1 = get_sentences(corpus1, target)
        sentences_c2 = get_sentences(corpus2, target)
        if len(sentences_c1) > sent_limit:
            sentences_c1 = sentences_c1[0:sent_limit]     
        if len(sentences_c2) > sent_limit:
            sentences_c2 = sentences_c2[0:sent_limit]    
        
        
        #corpus 1
        
        #get sentence embeddings (embeddings and tokenization) for sentences which contain target word in corpus1
        embed_target_c1, tokens_target_c1 = bc.encode(sentences_c1, show_tokens=True)
        #get word embeddings for target words
        target_embeddings_c1 = word_embeddings(embed_target_c1, tokens_target_c1, target)
            
        #earlier 
        earlier_dist = []
        
        #get all distances between target word embeddings in corpus1
        for i, embed in enumerate(target_embeddings_c1):
            j = 1 
            while i+j in range(len(target_embeddings_c1)):
                dist = cosine(target_embeddings_c1[i], target_embeddings_c1[i+j])
                earlier_dist.append(dist)
                j += 1
                
        #mean of all distances in corpus1  (=earlier)
        earlier = np.mean(np.array(earlier_dist))
        #standard deviation in earlier
        earlier_std = np.std(np.array(earlier_dist), axis = 0)
        
        #corpus2
        
        #get sentence embeddings (embeddings and tokenization) for sentences which contain target word in corpus2
        embed_target_c2, tokens_target_c2 = bc.encode(sentences_c2, show_tokens=True)
        #get word embeddings for target words
        target_embeddings_c2 = word_embeddings(embed_target_c2, tokens_target_c2, target)
        
        #later    
        later_dist = []
        
        #get all distances between target word embeddings in corpus2
        for i, embed in enumerate(target_embeddings_c2):
            j = 1 
            while i+j in range(len(target_embeddings_c2)):
                dist = cosine(target_embeddings_c2[i], target_embeddings_c2[i+j])
                later_dist.append(dist)
                j += 1
                
        #mean of all distances in corpus2  (=later)
        later = np.mean(np.array(later_dist))
        #standard deviation in later
        later_std = np.std(np.array(later_dist), axis=0)
        
        #delta_later  
        delta_later = later - earlier
   
        #compare
        compare_dist = []
        
        #get all distances between pairs of target word embeddings, where one embedding is from corpus1 and the other from corpus2
        for embed in target_embeddings_c1:
            for embed2 in target_embeddings_c2:
                dist = cosine(embed, embed2)
                compare_dist.append(dist)
        
        #mean of distances between pairs (=compare)       
        compare = np.mean(np.array(compare_dist))

        #compare_mixed
        all_embeddings = np.concatenate((target_embeddings_c1, target_embeddings_c2), axis = 0)
        
        mixed_dist = []
        
        #get all distances between all target word embeddings in corpus1 and corpus2
        for i, embed in enumerate(all_embeddings):
            j = 1 
            while i+j in range(len(all_embeddings)):
                dist = cosine(all_embeddings[i], all_embeddings[i+j])
                mixed_dist.append(dist)
                j += 1        
        
        #mean of distances between all target word embeddings
        compare_mixed = np.mean(np.array(mixed_dist))

        #delta_compare (here redefined as compare - compare_mixed)
        delta_compare = abs(compare - compare_mixed)
        
        metrics.write(target + '\t' + str(earlier) + '\t' + str(earlier_std) + '\t' + str(later) + '\t' + str(later_std) + '\t' + str(compare) + '\t' + str(compare_mixed) + '\t' + str(delta_later) + '\t' + str(delta_compare) + '\n')
        delta_later_out.write(target + '\t' + str(delta_later) + '\n')
        delta_compare_out.write(target + '\t' + str(delta_compare) + '\n')



        
        
if __name__ == '__main__':
    main()    
