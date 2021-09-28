#!/usr/bin/env python3

import re
import functools
from dataclasses import dataclass, field
from typing import List

from ufo.constant import *
from ufo.sentence import Sentence
from ufo.word import Word, Multiword


@dataclass
class Document:
    """This is the class representing a UFO document. Following example shows a basic
    use of this class.

    ```
    # Create a Document object
    document = Document()

    # Loads its content from an UFO file
    document.load("example.ufo")

    # Print each sentence's speaker
    for sentences in document.get_next_sentence():
        print(f'{sentence.header.speaker}')
    ```
    """
    sentences: List[Sentence] = field(default_factory=list)

    @classmethod
    def from_file(cls, filename):
        """Create a new document loaded from a file
        """
        document = Document()
        document.load(filename)

        return document

    def add_sentence(self, sentence):
        """Add a sentence to current document
        """
        self.sentences.append(sentence)

    def get_sentence_by_id(self, sentence_id):
        """Select sentence by ID

        - `sentence_id` is the sentence ID to retrieve

        It will return `None` is no sentence has been found
        """
        for sentence in self.sentences:
            if sentence.header.id == sentence_id:
                return sentence

        return None

    def get_sentence_by_speaker(self, speaker):
        """Select all the sentences from the given speaker

        - `speaker` is a string or list containing speakers's names
        """
        return filter(lambda sentence: sentence.header.speaker in speaker, self.sentences)

    def filter(self, predicate):
        """Returns a sequence of sentences for which `predicate` returns `True`

        - `predicate` is a function

        **Example**: to get the sentences starting with 'I am...'
        ```
        document = Document()
        document.load('example.ufo')

        document.filter(lambda s: s.header.text.startswith('I am'))
        ```
        """
        return filter(predicate, self.sentences)

    def get_full_text(self):
        """Build and return concatenated speech turns
        """
        turns = [sentence.header.text for sentence in self.sentences]
        return ' '.join(turns)

    def get_speaker_blocks(self, speaker, block_size):
        """Build `block_size`-words block composed of the given speaker tokens

        The returned value is a list of Sentences.

        - `speaker` is the speaker name
        - `block_size` is a size of 1 block (number of token)

        """
        # List of blocks
        blocks = []

        # List of sentences
        block = []

        # Size of current block
        size = 0

        # For each sentence of the given speaker
        for sentence in self.get_sentence_by_speaker(speaker):
            # If block_size if reached, add the current block
            # to the list and reset the size
            if size >= block_size:
                blocks.append(block)
                size = 0
                block = []

            # Else, fill the current block and update size
            block.append(sentence)
            size += sentence.words[-1].id
            # print(size)

        # TODO : possible ?
        if block:
            blocks.append(block)

        return blocks

    def build_from_sentences(self, sentences):
        """Fill a document with the given sentences

        All the sentences are removed from document before re-filling

        - `sentences` is a `Sentence` list
        """
        self.sentences = []
        self.sentences.extend(sentences)

    def split(self):
        """Split the document by sentences

        Returns a list of `Document`, each containing a speech turn.
        """
        documents = []
        for sentence in self.sentences:
            document = Document()
            document.add_sentence(sentence)
            documents.append(document)

        return documents

    def save(self, filename):
        """Save document into a file

        - `filename` is name of the output file
        """
        with open(filename, 'w', encoding='utf-8') as file:
            for sentence in self.sentences:
                file.write(sentence.str() + '\n')

    def load(self, filename):
        """Import from a file

        - `filename` is name of the input file
        """
        # Pre-compile each regex
        header_id_regex = re.compile(K_HDR_ID_REGEX, re.IGNORECASE)
        header_text_regex = re.compile(K_HDR_TEXT_REGEX, re.IGNORECASE)
        header_text_melt_regex = re.compile(K_HDR_TEXT_MELT_REGEX, re.IGNORECASE)
        header_speaker_regex = re.compile(K_HDR_SPEAKER_REGEX, re.IGNORECASE)
        header_conn_regex = re.compile(K_HDR_CONN_REGEX, re.IGNORECASE)
        header_da_regex = re.compile(K_HDR_DA_REGEX, re.IGNORECASE)
        header_edu_regex = re.compile(K_HDR_EDU_REGEX, re.IGNORECASE)
        header_drel_regex = re.compile(K_HDR_DREL_REGEX, re.IGNORECASE)
        header_nv_regex = re.compile(K_HDR_NON_VERBAL_REGEX, re.IGNORECASE)
        header_extra_regex = re.compile(K_HDR_EXTRA_REGEX, re.IGNORECASE)

        word_regex = re.compile(K_WORD_REGEX, re.IGNORECASE)
        multiword_regex = re.compile(K_MULTIWORD_REGEX, re.IGNORECASE)

        # Create a new Document
        document = Document()

        # Open file to import
        with open(filename, 'r', encoding='utf-8') as file:
            sentence = Sentence()

            # Scan all lines
            for line in file:
                stripped = line.strip()

                # Parse the current with all regex
                id_line = header_id_regex.match(stripped)
                text_line = header_text_regex.match(stripped)
                text_melt_line = header_text_melt_regex.match(stripped)
                speaker_line = header_speaker_regex.match(stripped)
                connective_line = header_conn_regex.match(stripped)
                da_line = header_da_regex.match(stripped)
                edu_line = header_edu_regex.match(stripped)
                drel_line = header_drel_regex.match(stripped)
                nv_line = header_nv_regex.match(stripped)
                extra_line = header_extra_regex.match(stripped)

                word_line = word_regex.match(stripped)
                multiword_line = multiword_regex.match(stripped)

                # If it matches the ID in header...
                if id_line is not None:
                    sentence.header.id = int(id_line.group(1))

                # else if it matches the text in header...
                elif text_line is not None:
                    sentence.header.text = text_line.group(1)

                # else if it matches the text in header...
                elif text_melt_line is not None:
                    sentence.header.text_melt = text_melt_line.group(1)

                # else if it matches the speaker in header...
                elif speaker_line is not None:
                    sentence.header.speaker = speaker_line.group(1)

                # else if it matches the connective in header...
                elif connective_line is not None:
                    sentence.header.connective = connective_line.group(1)

                # else if it matches the dialogue act in header...
                elif da_line is not None:
                    sentence.header.dialogue_act = da_line.group(1)

                # else if it matches the linked edu in header...
                elif edu_line is not None:
                    sentence.header.linked_edu = int(edu_line.group(1))

                # else if it matches the discourse relation in header...
                elif drel_line is not None:
                    sentence.header.discourse_rel = drel_line.group(1)

                # else if it matches the non-verbal line in header...
                elif nv_line is not None:
                    sentence.header.non_verbal = nv_line.group(1)

                # else if it matches an extra attribute definition...
                elif extra_line is not None:
                    key = extra_line.group(1)
                    value = extra_line.group(2)
                    sentence.add_attribute(key, value)

                # else if it is a word...
                elif multiword_line is not None:
                    sentence.parse_and_add_word(multiword_line.group(0), True)

                # else if it is a word...
                elif word_line is not None:
                    sentence.parse_and_add_word(word_line.group(0))

                # else, it's a blank line: save current sentence into document
                # and create the next one
                else:
                    self.sentences.append(sentence)
                    sentence = Sentence()
