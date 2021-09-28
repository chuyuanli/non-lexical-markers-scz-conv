# Data files
SLAM_DIR_DATA = 'data/'
SLAM_DIR_FEATURES = 'ufo_features/'
SLAM_DIR_UFO = SLAM_DIR_DATA + 'ufo/'
SLAM_DIR_ORIGINAL = SLAM_DIR_DATA + 'original/'

# Connectoris lists
SLAM_LEXICAL_LIST_DIR = 'lexical_lst/'
SLAM_FILE_CONN = SLAM_LEXICAL_LIST_DIR + 'conn.txt'
SLAM_FILE_OCR = SLAM_LEXICAL_LIST_DIR + 'ocr.txt'
SLAM_FILE_BC = SLAM_LEXICAL_LIST_DIR + 'backchannel.txt'
SLAM_FILE_CAUSE = SLAM_LEXICAL_LIST_DIR + 'inq_causal_fr.txt'
SLAM_FILE_DEIC = SLAM_LEXICAL_LIST_DIR + 'deictiques_v2.txt'

SLAM_KEYWORDS_FILE = {
    'ocr':  SLAM_DIR_DATA + SLAM_FILE_OCR,
    'conn': SLAM_DIR_DATA + SLAM_FILE_CONN,
    'backchannel': SLAM_DIR_DATA + SLAM_FILE_BC,
    'causal': SLAM_DIR_DATA + SLAM_FILE_CAUSE,
    'deictique': SLAM_DIR_DATA + SLAM_FILE_DEIC
}

# P/T: 1, P/S: -1
SLAM_LABELS = [
    1,  # 001PEPout
    1,  # 002LOAout
    1,  # 003DOCout
    1,  # 004KATout
    1,  # 005AUBout
    1,  # 006MABout
    1,  # 007THCout
    1,  # 008BALout
    1,  # 009MAEout
    1,  # 010BLDout
    1,  # 011HAMout
    1,  # 012PEJout
    -1, # 013DURout
    -1, # 014BRLout
    -1, # 015HOSout
    -1, # 016FARout
    -1, # 018NICout
    -1, # 019TIOout
    -1, # 020MAIout
    -1, # 021MOLout
    -1, # 022CEMout
    -1, # 023PEHout
    -1, # 024MAAout
    -1, # 025FIGout
    -1, # 026FRAout
    -1, # 027TEAout
    -1, # 028MAHout
    -1, # 029STJout
    -1, # 030BIVout
    -1, # 031ROJout
    -1, # 032MASout
    -1, # 033VEQout
    -1, # 034BADout
    1,  # 035RIPout
    1,  # 036DEFout
    1,  # 037VOAout
    1,  # 038BANout
    1,  # 039BOBout
    -1, # 040NICout
    -1, # 041SAGout
    1   # 042BRBout
]

SLAM_LIST1_DEIC = ['je', 'me', 'moi', 'nous', 'mon', 'ma', 'mes', 'notre', 'nos',
    'tu', 'te', 'toi', 'vous', 'ton', 'ta', 'tes', 'votre', 'vos', 'on',
    'aujourd’hui', 'maintenant', 'demain', 'hier', 'avant', 'à-ce-moment-là',
    'la semaine prochaine', 'ici', 'à côté', 'là', 'ceci', 'celui-ci', 'celui-là',
    'ce', 'cet', 'cette', 'ces', 'voici', "tout-à-l'-heure", "j'", "t'", "m'"]

SLAM_LIST2_JETU = ["j'", "t'", "je", "tu"]
