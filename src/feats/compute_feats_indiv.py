#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import data
import features
from constants import *
from ufo.document import Document

# old = 10319
# new : 10966
# new without empty line : 10742 (diff = 224)
# new empty line count a la main = 306

# find . -name "*.ufo" | xargs grep -E "speaker: s|speaker: t" | grep "text_melt: empty" | wc -l

def _generate_individual_documents(tgt_spk):
    documents = []
    labels = []

    for (document, label) in zip(data.documents, SLAM_LABELS):
        # Keep non-empty sentences in selected sentences
        sentences = list(document.filter(lambda s: s.header.speaker in tgt_spk and s.header.text_melt.strip() != 'empty'))

        # Create a new document with selected sentences
        subdocument = Document()
        subdocument.build_from_sentences(sentences)

        # Add subdocument in the documents list
        documents.extend(subdocument.split())

        # Update labels list
        # labels.extend(len(sentences)*[label])
        for sent in sentences:
            if sent.header.speaker == 'p' and label == 1:
                labels.append(2) #psy with scz
            elif sent.header.speaker == 'p' and label == -1:
                labels.append(-2) #psy with temoin
            else: #speaker = 's' or 't', directly append the original label 1 or -1
                labels.append(label)
        
    assert len(documents) == len(labels)

    return documents, labels


if __name__ == "__main__":
    # ===== RECUPERATION DES OPTIONS
    parser = argparse.ArgumentParser(description='Read ufo data, generate features files.')
    parser.add_argument('--feat', dest='feat', action='store', default="bow", help="""Type of features:\n
    - lexical: bow(ngram=1, 23), deictique, jetu
    - syntactic: pos (ngram=1,2,3,123), treelet(ngram=2,3,123)
    - dialogical: backchannel, ocr, conn, causal
    """)
    parser.add_argument('--ngram', dest='ngram', type=int, default=1, help='for bow and syntactic features, n=1, 2, 3, 123')
    parser.add_argument('--spk', dest='spk', type=str, default='st', help='if count psy as speaker then its stp, otherwise just st.')
    args = parser.parse_args()
    feat = args.feat
    ngram = args.ngram
    target_spk = args.spk
    
    documents, Y = _generate_individual_documents(target_spk)
    features.compute('indiv', feat, documents, Y, target_spk, ngram=ngram)
