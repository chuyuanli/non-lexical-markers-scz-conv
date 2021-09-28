#!/usr/bin/env python3

import itertools
from dataclasses import dataclass, field
from typing import List

from ufo.constant import *
from ufo.header import Header
from ufo.word import Word, Multiword


@dataclass
class Sentence:
    header: Header = field(default_factory=Header)
    words: List[Word] = field(default_factory=list)

    def str(self):
        """Return a string formatted Sentence
        """
        # Header in first position
        sentence = self.header.str() + '\n'
        # And then add each word
        for word in self.words:
            sentence += word.str() + '\n'

        return f'{sentence}'

    def add_attribute(self, key, value=UFO_DEFAULT_VALUE):
        """Add a customized attribute to the sentence header

        - `key` is the attribute identifier
        - `value` is the attribute value
        """
        self.header.add_attribute(key, value)

    def get_attribute(self, key):
        """Return extra attribute value
        """
        return self.header.get_attribute(key)

    def without_words_in(self, exclude_list, attribute_name='text'):
        """Return the given sentence attribute excluding words in `exclude_list`

        - `exclude_list` is the list of words to exclude
        - `attribute_name` is the attribute on which filter apply (default is `text`)

        .. note:: this does not return a list of `Word` objects
        """
        attribute = getattr(self.header, attribute_name).split()
        return ' '.join(itertools.filterfalse(lambda x: x in exclude_list, attribute))

    def keep_only_words_in(self, include_list, attribute_name='text'):
        """Return the given sentence attribute keeping only words in `inlcude_list`

        - `include_list` is the list of words to keep
        - `attribute_name` is the attribute on which filter apply (default is `text`)

        .. note:: this does not return a list of `Word` objects
        """
        attribute = getattr(self.header, attribute_name).split()
        return ' '.join(filter(lambda x: x in include_list, attribute))

    def parse_and_add_word(self, str_word, is_multiword=False):
        """Take a string reprensenting a CONLLU token or multiword,
        creates a Word object and add it to current the sentence

        - `str_word` is a string containing tab-separated token information
        - `is_multiword` is `True` if `str_word` has to be parsed a range
         (see [CONNLU specification](https://universaldependencies.org/format.html))
        """
        splitted = str_word.strip().split('\t')
        word = None

        if is_multiword:
            id_first = int(splitted[0].split('-')[0])
            id_last = int(splitted[0].split('-')[1])
            form = splitted[1]
            misc = splitted[2]
            word = Multiword(id_first, id_last, form, misc)
        else:
            word = Word(*splitted)

        self.add_word(word)

    def add_word(self, word):
        """Add a Word object to the sentence

        - `word` is a `Word` object
        """
        self.words.append(word)

    def count_words(self):
        """Return the number of tokens in sentence

        Multiword objects and empty lines are not counted
        """
        return len([word for word in self.words
                    if not isinstance(word, Multiword)
                    and word.form != '[empty line]'])
