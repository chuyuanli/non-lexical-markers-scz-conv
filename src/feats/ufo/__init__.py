"""## Unified Feature Object (UFO) format

### General description

The format is based on the [CONNLU format](https://universaldependencies.org/format.html).
Each file is represented by the `Document` type, which contains a set of
`Sentence`s. Each sentence is composed of a `Header`,
and a set of `Word`, containing differents attributes.

### Example of an UFO document

```
#id: 1
#text: Ah oui bon
#original_text: Ah oui bon
#speaker: s
#connective: _
#dialogue_act: _
#linked_edu: 3
#discourse_rel: _
#non_verbal: smile_
#extra/my_field: my_value
1	Ah	ah	ADV	_	_	3	advmod	_	_	False	1	True	54	True	120	False	_	0	False	0	0	_	_	_	_	False	_	_	_
2	oui	oui	ADV	_	_	1	root	_	_	True	1	True	4	False	43	False	_	0	False	0	0	_	_	_	_	False	_	_	_
3	bon	bon	ADV	_	_	5	adv	_	_	True	4	True	0	False	30	False	_	0	False	0	0	_	_	_	_	False	_	_	_
```

## Getting started

### Installation

Copy the `ufo` directory inside your project.

### Hello world

To use the UFO inside a projet, you need first to import the package:
```
import ufo.document as ufo
```

Then you can create a UFO document, fill it from a file and print its content:

```
document = ufo.Document()
document.load('my_file.ufo')

for sentence in document.get_next_sentence():
    print(sentence)
```

## How to...

### Load a document from a file

A document can be loaded from a file by using the `from_file()` method:

```
document = Document.from_file('my_file.ufo')
```

### Count the number of speech turn in a document

There is no direct method to return the number of sentence in a document.
However, as the field are accessible, the sentences can be counted using:

```
nb_sentences = len(document.sentences)
```

### Use header dynamic fields

It is possible to add several custom fields in header, using the following sentence method:

```
sentence = document.get_sentence_by_id(2)
sentence.add_attribute('attribute_name', 'value')
```

This new attribute will appear in the `#extra/<attribute_name>: <value>` section in header.

It can be retrieved this way:

```
attribute_value = sentence.get_attribute('attribute_name')
```


"""
