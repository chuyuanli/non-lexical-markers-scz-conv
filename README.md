# Investigating non lexical markers of the language of schizophrenia inspontaneous conversations

This is the source code for the paper "Investigating non lexical markers of the language of schizophrenia inspontaneous conversations" (CODI 2021, EMNLP workshop).

### Dataset SLAM
Due to the confidentiality of dataset, we don't release the data publically. 
Please contact Maxime Amblard (firstname dot name at loria dot fr) should you have further request.

### Code structure
- You will need first extract features from the `src/feats/` folder. Depending on the window size, choose from {`compute_feats_block.py`, `compute_feats_full.py`, `compute_feats_indiv.py`}. Every successfully extracted feature will have two files: `train-st.svmlight` and `vocab-featureName-st`.
- With svmlight and vocab files, opt for classif.py script for classification task. Read carefully the instructions for arguments. 

### UFO scheme
ufo is short for Unified Feature Object. It is a CONLLU-based format with extention to more discourse/dialogical information. In this paper we use this format to pre-process our dataset. For [more information](https://slodim.gitlabpages.inria.fr/slodim-ufo/) about the format, please contact Pierre Lefebvre or Chuyuan Li (firstname dot name at loria dot fr).