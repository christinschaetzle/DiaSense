import numpy as np
import re


#get sentences in which 'word' occurs from 'corpus', cleaning for improved results
def get_sentences(corpus, word):
    sentences_corp = [line for line in corpus if word in line.split(' ')]      
    sentences_corp = [line for line in sentences_corp if not re.search(r'\d\d?\.\d\d', line)]
    for i, sent in enumerate(sentences_corp):
        if re.search(r'\d+', sentences_corp[i]):
            sentences_corp[i] = re.sub(r'\d+',  '', sentences_corp[i])
        sentences_corp[i] = sentences_corp[i].replace('  ', ' ').replace('-', '').replace('*', '').replace('.', '')
    return sentences_corp

#map bert tokenization to words
def tokens_to_words(tokenized_sentence):
    bert_to_words = []
    for j, token in enumerate(tokenized_sentence):
        if j+1 < len(tokenized_sentence):
            if not '##' in tokenized_sentence[j] and '##' in tokenized_sentence[j+1]:
                begin = j
                tokens_to_word = tokenized_sentence[j] + tokenized_sentence[j+1]
                tokens_to_word = tokens_to_word.replace('##', '')
                n = 2
                while j+n < len(tokenized_sentence) and '##' in tokenized_sentence[j+n]:
                    tokens_to_word = tokens_to_word + tokenized_sentence[j+n]
                    tokens_to_word = tokens_to_word.replace('##', '')
                    n+=1
                end = j+n-1 
                bert_to_words.append((tokens_to_word, begin, end))
            elif not '##' in tokenized_sentence[j] and not '##' in tokenized_sentence[j+1]:
                bert_to_words.append((token, j, j))
    bert_to_words.pop(0)  #remove CLS
    
    return bert_to_words

#get word embeddings for target words from sentence embeddings; average over subword-embeddings if target words tokenized into more than one token   
def word_embeddings(sentence_embeddings, bert_tokenization, word):
    word_embeddings = []
    for i, sent_tokenized in enumerate(bert_tokenization):
        bert_to_words = tokens_to_words(sent_tokenized)
        word = word.lower().replace("ä", "a").replace("Ä", "A").replace("ö", "o").replace("Ö", "Ö").replace("ü", "u").replace("Ü", "U")        
        word = re.sub(r'\_\w+\b', '', word)
        for bert_word in bert_to_words: 
            if bert_word[0] == word:
                if bert_word[1] != bert_word[2]:    #word not tokenized as one token, average embeddings of corresponding tokens
                    word_embed = np.average(sentence_embeddings[i][bert_word[1]:bert_word[2]+1], axis = 0)
                    word_embeddings.append(word_embed)
                else:
                    #word has been tokenized as one token
                    word_embed = sentence_embeddings[i][bert_word[1]]
                    word_embeddings.append(word_embed)

    word_embeddings = np.array(word_embeddings)
    
    return word_embeddings


    
    
