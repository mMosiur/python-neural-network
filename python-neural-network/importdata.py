"""
Module designed to import idx data into question-answer pairs.
"""

from idx2numpy import convert_from_file
from qapairs import QAPairs

def import_data(image_source_file, label_source_file):
    """
    Imports data, from two files - one with images idx file,
    the other with labels idx file, into QAPairs data structure
    """
    try:
        with open(image_source_file, "rb") as imgs, open(label_source_file, "rb") as labs:
            imgs = convert_from_file(imgs)
            labs = convert_from_file(labs)
    except:
        print("Error during processing of the images file")
        raise
    return QAPairs(imgs, labs)
