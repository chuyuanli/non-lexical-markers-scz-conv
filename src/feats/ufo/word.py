#!/usr/bin/env python3

import distutils.util
from dataclasses import dataclass
from ufo.constant import *

# tense, aspect, modality values, ref: http://www.timeml.org/publications/timeMLdocs/timeml_1.2.1.html


@dataclass
class Multiword:
    """Range of a multiword expression

    .. caution::When this code is updated, do not forget to keep the same order between attributes declaration in this class and attributes printing in `str()` method
    """
    id_first: int = UFO_DEFAULT_ID
    """First token ID in the multiword/contracted expression"""

    id_last: int = UFO_DEFAULT_ID
    """Last token ID in the multiword/contracted expression"""

    form: str = UFO_DEFAULT_VALUE
    """Multiword/contracted expression"""

    misc: str = UFO_DEFAULT_VALUE  # other annotation
    """Misc"""

    def str(self):
        """Return a string formatted Multiword
        """
        return f"{self.id_first}-{self.id_last}\t{self.form}\t{self.misc}"


@dataclass
class Word:
    """Sentence token and its attributes

    First 10 attributes are identical to CONNLU format.

    .. caution::When this code is updated, do not forget to keep the same order between attributes declaration in this class and attributes printing in `str()` method
    """
    # 10 attributes in CoNLL-U format
    id: int = UFO_DEFAULT_ID
    """Token ID"""

    form: str = UFO_DEFAULT_VALUE
    """Token form"""

    lemma: str = UFO_DEFAULT_VALUE
    """Token lemma"""

    upos: str = UFO_DEFAULT_VALUE
    """Universal Part-of-Speech"""

    xpos: str = UFO_DEFAULT_VALUE
    """Language specific Part-of-Speech"""

    feats: str = UFO_DEFAULT_VALUE
    """List of morphological features"""

    head: int = UFO_DEFAULT_VALUE
    """Head of current word, either id or 0"""

    deprel: str = UFO_DEFAULT_VALUE
    """Universal dependecy relation to the head"""

    deps: str = UFO_DEFAULT_VALUE
    """Enhanced dependency graph"""

    misc: str = UFO_DEFAULT_VALUE
    """Other annotation"""

    # 14 new attributes
    backchannel_grp: int = UFO_DEFAULT_ID
    """Backchannel group number"""

    connective_grp: int = UFO_DEFAULT_ID
    """Connective group number"""

    ocr_grp: int = UFO_DEFAULT_ID
    """OCR group number"""

    disfluence_type: str = UFO_DEFAULT_VALUE
    """
    .. todo::e.g `repetition` | `euh` | `?`
    """

    disfluence_grp: int = UFO_DEFAULT_ID
    """Disfluence group number"""

    coref_sent: int = UFO_DEFAULT_ID
    """Coreference sentence ID"""

    coref_token: int = UFO_DEFAULT_ID
    """ID of coreference token in the previous sentence"""

    tense: str = UFO_DEFAULT_VALUE
    """e.g `FUTURE` | `INFINITIVE` | `PAST` | `PASTPART` | `PRESENT` | `PRESPART` | `NONE`"""

    aspect: str = UFO_DEFAULT_VALUE
    """e.g `PROGRESSIVE` | `PERFECTIVE` | `PERFECTIVE_PROGRESSIVE` | `NONE`"""

    modality: str = UFO_DEFAULT_VALUE
    """
    .. todo::
    """

    topic: str = UFO_DEFAULT_VALUE
    """
    .. todo::
    """

    question_type: str = UFO_DEFAULT_VALUE
    """
    .. todo::e.g `wg` | `yes-no`
    """

    misc2: str = UFO_DEFAULT_VALUE
    """Misc 2"""

    misc3: str = UFO_DEFAULT_VALUE
    """Misc 3"""

    def __post_init__(self):
        """Convert some fields to their actual types
        """
        # Integers
        self.id = int(self.id)

        if self.head != UFO_DEFAULT_VALUE:
            self.head = int(self.head)

        self.backchannel_grp = int(self.backchannel_grp)
        self.connective_grp = int(self.connective_grp)
        self.ocr_grp = int(self.ocr_grp)
        self.disfluence_grp = int(self.disfluence_grp)
        self.coref_sent = int(self.coref_sent)
        self.coref_token = int(self.coref_token)

    def str(self):
        """Return a string formatted Word
        """
        ret = f"{self.id}\t{self.form}\t{self.lemma}\t{self.upos}\t{self.xpos}\t{self.feats}\t"
        ret += f"{self.head}\t{self.deprel}\t{self.deps}\t{self.misc}\t{self.backchannel_grp}\t"
        ret += f"{self.connective_grp}\t{self.ocr_grp}\t"
        ret += f"{self.disfluence_type}\t{self.disfluence_grp}\t{self.coref_sent}\t{self.coref_token}\t"
        ret += f"{self.tense}\t{self.aspect}\t{self.modality}\t{self.topic}\t{self.question_type}\t"
        ret += f"{self.misc2}\t{self.misc3}"

        return ret
