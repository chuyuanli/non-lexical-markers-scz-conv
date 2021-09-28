#!/usr/bin/env python3

import os
from constants import SLAM_DIR_UFO
from ufo.document import Document

documents = []

for file in sorted(os.listdir(SLAM_DIR_UFO)):
    document = Document()
    document.load(os.path.join(SLAM_DIR_UFO, file))
    documents.append(document)