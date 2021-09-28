#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from os.path import join, exists
from os import makedirs
import joblib
from sklearn.datasets import dump_svmlight_file
from flashtext import KeywordProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from constants import *
from utility import *


def build_tfidfvectorizer(words, tokenizer=None, min_df=1, ngram_range=(1,1), shownames=False):
    vectorizer = TfidfVectorizer(min_df=min_df, tokenizer=tokenizer, ngram_range=ngram_range)
    X = vectorizer.fit_transform(words)
    if shownames:
        names = vectorizer.get_feature_names()
        return vectorize, X, names
    else:
        return vectorizer, X

#===================================================================
def dialogical(documents, keywords_file, target_spk, min_df=1):
    """
    Generic function for backchannels, OCR, connectors, causal
    """
    train_wds = get_dialogical_wds(documents=documents, keywords_file=keywords_file, target_spk=target_spk)
    vectorizer, X_train = build_tfidfvectorizer(train_wds, tokenizer=lambda s: s.split(), min_df=min_df)
    print("\nX_train:", X_train.shape)
    return X_train, vectorizer

def lexical(documents, feat, target_spk, min_df, ngram):
    """
    Generic function for:
    - bow (ngram=1, 23)
    - bow without deictics (bow_nodeic), bow without 'je' and 'tu' (bow_nojetu)
    - deictique, jetu (in ratio)
    """
    train_wds = get_lexical_wds(documents=documents, feat=feat, target_spk=target_spk)
    if ngram == 1:
        ngram_range = (1,1)
    elif ngram == 23:
        ngram_range = (2,3)
    vectorizer, X_train = build_tfidfvectorizer(train_wds, tokenizer=lambda s: s.split(), min_df=min_df, ngram_range=ngram_range)
    print("\nX_train:", X_train.shape)
    return X_train, vectorizer

def syntactic(documents, feat, target_spk, min_df, ngram):
    """
    Generic function for POS and treelet
    """
    train_wds = get_syntactic_wds(documents=documents, feat=feat, target_spk=target_spk, ngram=ngram)
    vectorizer, X_train = build_tfidfvectorizer(train_wds, tokenizer=lambda s: s.split(), min_df=min_df)
    print("\nX_train:", X_train.shape)
    return X_train, vectorizer

#===================================================================
def get_dialogical_wds(documents, keywords_file, target_spk):
    """
    parameters:
        - target_spk: string, 'st' or 'pp', determine the classfication objects
        - feat: string, choose from {'backchannel', 'ocr', 'conn', 'causal'}
        - keywords_file: a txt file with one line a keyword to match
    return: a list of strings, length of list = #docs to classify
    """
    # set up keyword processor
    kw_processor = KeywordProcessor()
    kw_processor.add_keyword_from_file(keywords_file)
    wds = []
    for doc in documents:
        doc_wds = []
        for sent in doc.filter(lambda s: s.header.speaker in target_spk):
            tdp = sent.header.text
            kw_found = kw_processor.extract_keywords(tdp.lower())
            kw2 = list(map(lambda a: '-'.join(a.split()) if len(a.split()) > 1 else a, kw_found))
            doc_wds.extend(kw2)
        wds.append(' '.join(doc_wds))
    return wds

def get_lexical_wds(documents, feat, target_spk):
    """
    parameters:
        - target_spk: string, 'st' or 'pp', determine the classfication objects
        - feat: string, choose from:
            'bow' = no excluding any deictics, keep the text as it is
            'bow_nodeic' = exclude all words from list1
            'bow_nojetu' = exlude all words from list2
            'deictique' = only keep words in list1, i.e.: only keep deictic words
            'jetu' = only keep words in list2, i.e.: only keep 'je' et 'tu'
    return: a list of strings/ratios (for feat='jetu'), length of list = #docs to classify
    """
    wds = []
    for doc in documents:
        if feat == 'jetu':
            je, tu = 1e-5, 1e-5 # in case of 0 occurance
        else:
            doc_wds = []
        #parse a sub-doc from target_spk 
        for sent in doc.filter(lambda s: s.header.speaker in target_spk):
            tdp = sent.header.text_melt
            if feat != 'bow':
                tdp = tdp.replace('_', '-') #in melt all composed words are linked with '_'
                if feat == 'bow_nodeic':
                    new_tdp = ' '.join([i for i in tdp.split() if not i in SLAM_LIST1_DEIC])
                    # filter(lambda parameter_list: expression, tdp)  
                elif feat == 'bow_nojetu':
                    new_tdp = ' '.join([i for i in tdp.split() if not i in SLAM_LIST2_JETU])
                elif feat == 'deictique':
                    new_tdp = ' '.join([i for i in tdp.split() if i in SLAM_LIST1_DEIC])
                elif feat == 'jetu':
                    je += len([i for i in tdp.split() if i in ['je', "j'"]])
                    tu += len([i for i in tdp.split() if i in ['tu', "t'"]])
            else:
                new_tdp = tdp
            doc_wds.append(new_tdp) 
        if feat == 'jetu':
            wds.append(math.log(je / tu)) #a list of ratios
        else:
            wds.append(' '.join(doc_wds))
    return wds

def get_syntactic_wds(documents, feat, target_spk, ngram=1):
    """
    parameters:
        - target_spk: string, 'st' or 'pp', determine the classfication objects
        - feat: string, choose from:
            'pos' = part-of-speech, ngram with n in {1,2,3,123}, 123pos is the combination
            'treelet' = syntactic relation, ngram with n in {2,3,123}, 123 is the combination of {1pos, 2treelet, 3treelet}
    return: a list of strings, length of list = #docs to classify
    """
    wds = []
    for doc in documents:
        doc_wds = ''
        doc_1pos = []
        for sent in doc.filter(lambda s: s.header.speaker in target_spk):
            tdp = sent.words
            dg, pos = build_graph(tdp)
            doc_1pos.extend(pos)
            # pos, 2pos, 3pos, 123pos
            if 'pos' in feat:
                if ngram == 1 or ngram == 123:
                    doc_wds += ' '.join(doc_1pos) + ' '
                if ngram == 2 or ngram == 123:
                    bi = [x + '-' +doc_1pos[j+1] for j, x in enumerate(doc_1pos) if j < len(doc_1pos) - 1]
                    doc_wds += ' '.join(bi) + ' '
                if ngram == 3 or ngram == 123:
                    tri = [x+'-'+doc_1pos[j+1]+'-'+doc_1pos[j+2] for j, x in enumerate(doc_1pos) if j < len(doc_1pos) - 2]
                    doc_wds += ' '.join(tri) + ' '
            # 2treelet, 3treelet, 123treelet (=1pos+2treelet+3treelet)
            elif 'treelet' in feat:
                if ngram == 123:
                    doc_wds += ' '.join(doc_1pos) + ' '
                if ngram == 2 or ngram == 123:
                    bi = []
                    for (h, r, d) in dg.triples():  # (head_word,head_tag), relation, (dep_word, dep_tag)
                        bi.append(h[1]+'->'+r+'->'+d[1])
                    doc_wds += ' '.join(bi) + ' '
                if ngram == 3 or ngram == 123:
                    tri = get_three_tok_treelets(dg)
                    doc_wds += ' '.join(tri) + ' '
        wds.append(doc_wds)
    return wds


def compute(document_type, feat, documents, Y, target_spk, ngram=1, min_df=1):
    #get X
    if feat in ['backchannel', 'ocr', 'conn', 'causal']:
        X, vectorizer = dialogical(documents, SLAM_KEYWORDS_FILE[feat], target_spk=target_spk, min_df=min_df)
    elif feat in ['bow', 'bow_nodeic', 'bow_nojetu', 'deictique', 'jetu']:
        X, vectorizer = lexical(documents, feat, target_spk=target_spk, min_df=min_df, ngram=ngram)
    elif feat in ['pos', 'treelet']:
        # python3 compute_feats_indiv.py --feat 123pos
        # python3 compute_feats_indiv.py --feat pos --ngram 123 #DONE
        X, vectorizer = syntactic(documents, feat, target_spk=target_spk, min_df=min_df, ngram=ngram)
    else:
        raise ValueError("Feature not recognized. Refer to --help for more information.") 

    # write file svmlight
    rep_feat = join(SLAM_DIR_FEATURES, f'{document_type}/{feat}_{ngram}')
    if not exists(rep_feat):
        makedirs(rep_feat)
    dump_svmlight_file(X, Y, join(rep_feat, f'train-{target_spk}.svmlight'))
    
    # save vocabulary
    vocab = vectorizer.vocabulary_ if vectorizer else {k: 'feat'+str(k) for k in range(X.shape[1])}
    joblib.dump(vocab, join(rep_feat, 'vocab-'+feat+f'-{target_spk}'), compress=3)
