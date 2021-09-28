#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Dict
from ufo.constant import *


@dataclass
class Header:
    """Sentence header
    """
    id: int = UFO_DEFAULT_ID
    """Sentence ID"""

    text: str = UFO_DEFAULT_VALUE
    """Original transcription text"""

    text_melt: str = UFO_DEFAULT_VALUE
    """MELT forms concatenated (with underscored `multi_words`)"""

    speaker: str = UFO_DEFAULT_VALUE
    """Speaker name"""

    connective: str = UFO_DEFAULT_VALUE
    """Discourse connective"""

    dialogue_act: str = UFO_DEFAULT_VALUE
    """
    .. TODO:: e.g `offer` | `counter-offer` | `acceptance` | `refusal` | `other`
    """

    linked_edu: int = UFO_DEFAULT_ID
    """ID to which previous sentence/EDU this speech turn is adressed to"""

    discourse_rel: str = UFO_DEFAULT_VALUE
    """
    .. TODO:: Relation with the linked sentence
    """

    non_verbal: str = UFO_DEFAULT_VALUE
    """
    .. TODO:: Non verbal element
    """

    _extra: Dict = field(default_factory=dict)
    """Customized attributes"""

    def add_attribute(self, key, value=UFO_DEFAULT_VALUE):
        self._extra[key] = value

    def get_attribute(self, key):
        try:
            return self._extra[key]
        except KeyError:
            return UFO_DEFAULT_VALUE

    def str(self):
        """Returns string-formatted Header
        """
        ret = f"#{K_HDR_ID}: {self.id}\n"
        ret += f"#{K_HDR_TEXT}: {self.text}\n"
        ret += f"#{K_HDR_TEXT_MELT}: {self.text_melt}\n"
        ret += f"#{K_HDR_SPEAKER}: {self.speaker}\n"
        ret += f"#{K_HDR_CONN}: {self.connective}\n"
        ret += f"#{K_HDR_DA}: {self.dialogue_act}\n"
        ret += f"#{K_HDR_EDU}: {self.linked_edu}\n"
        ret += f"#{K_HDR_DREL}: {self.discourse_rel}\n"
        ret += f"#{K_HDR_NON_VERBAL}: {self.non_verbal}"

        for (key, value) in self._extra.items():
            ret += f"\n#{K_HDR_EXTRA}/{key}: {value}"

        return ret
