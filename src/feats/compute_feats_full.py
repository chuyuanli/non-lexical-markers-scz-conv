#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import data
import features
from constants import *

if __name__ == "__main__":
    # ===== RECUPERATION DES OPTIONS
    parser = argparse.ArgumentParser(description='Read ufo data, generate features files.')
    parser.add_argument('--feat', dest='feat', action='store', default="bow", help="""Type of features:\n
    - lexical: bow(ngram=1, 23), deictique, jetu
    - syntactic: pos (ngram=1,2,3,123), treelet(ngram=2,3,123)
    - dialogical: backchannel, ocr, conn, causal
    """)
    parser.add_argument('--ngram', dest='ngram', type=int, default=1, help='for bow and syntactic features, n=1, 2, 3, 123')
    parser.add_argument('--spk', dest='spk', type=str, default='st', help="if count psy as speaker then its stp, otherwise just st.")
    args = parser.parse_args()
    feat = args.feat
    ngram = args.ngram
    target_spk = args.spk
    
    features.compute('full', feat, data.documents, SLAM_LABELS, target_spk, ngram=ngram)
