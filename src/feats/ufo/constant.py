#!/usr/bin/env python3

UFO_DEFAULT_VALUE = '_'
"""Default UFO attribute **string** value"""

# WARNING: Regex have to be modified if is become a negative number
UFO_DEFAULT_ID = 0
"""Default UFO ID value"""

# Header sections
K_HDR_ID = 'id'
K_HDR_TEXT = 'text'
K_HDR_TEXT_MELT = 'text_melt'
K_HDR_SPEAKER = 'speaker'
K_HDR_CONN = 'connective'
K_HDR_DA = 'dialogue_act'
K_HDR_EDU = 'linked_edu'
K_HDR_DREL = 'discourse_rel'
K_HDR_NON_VERBAL = 'non_verbal'
K_HDR_EXTRA = 'extra'

K_HDR_ID_REGEX = r'#id:\s*([0-9]+)'
K_HDR_TEXT_REGEX = r'#text:\s*(.*)'
K_HDR_TEXT_MELT_REGEX = r'#text_melt:\s*(.+)'
K_HDR_SPEAKER_REGEX = r'#speaker:\s*(.+)'
K_HDR_CONN_REGEX = r'^#connective:\s*(.*)$'
K_HDR_DA_REGEX = r'^#dialogue_act:\s*(.*)$'  # TBD, should be a close list
K_HDR_EDU_REGEX = r'^#linked_edu:\s*([0-9]+)'
K_HDR_DREL_REGEX = r'^#discourse_rel:\s*(.*)$'  # TBD, should be a close list
K_HDR_NON_VERBAL_REGEX = r'^#non_verbal:\s*(.+)'
K_HDR_EXTRA_REGEX = r'^#extra/(.+):\s*(.+)'

K_MULTIWORD_REGEX = r'^\d+-\d+.+'
K_WORD_REGEX = r'(.+\t){10}'
