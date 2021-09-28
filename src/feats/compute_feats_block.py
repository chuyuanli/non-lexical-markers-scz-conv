#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import data
import features
from constants import *
from ufo.document import Document

def _generate_block_documents(tgt_spk, block_size):
    documents = []
    labels = []

    for (document, label) in zip(data.documents, SLAM_LABELS):
        sub_blocks = document.get_speaker_blocks(tgt_spk, block_size)

        # Update labels list
        labels.extend(len(sub_blocks)*[label])

        for sb in sub_blocks:
            new_document = Document()
            new_document.build_from_sentences(sb)
            documents.append(new_document)

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
    parser.add_argument('--blocksize', dest='blocksize', type=int, default=128, help='Size of blocks (128, 256, 512, 1024)')
    parser.add_argument('--spk', dest='spk', type=str, default='st', help="""if count psy as speaker then its stp, otherwise just st. 
                                                                            ATTENTION: for comparability: st b-128 is comparable with stp b-256.""")
    args = parser.parse_args()
    feat = args.feat
    ngram = args.ngram
    block_size = args.blocksize
    target_spk = args.spk
    
    documents, Y = _generate_block_documents(target_spk, block_size)

    # Computes feature on each document
    features.compute(f'block_{block_size}', feat, documents, Y, target_spk, ngram=ngram)
