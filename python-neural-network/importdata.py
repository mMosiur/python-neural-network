"""
Module designed to import idx data into question-answer pairs.
"""

from idx2numpy import convert_from_file

class QAPair:
    """ Data structure of question-answer pairs """
    def __init__(self, img, lbl):
        self.image = img / 255 # Converting numbers between 0-255(int) to activations 0-1(float)
        self.label = lbl

def import_data(image_source_file, label_source_file):
    """
    Imports data, from two files - one with images idx file,
    the other with labels idx file, into QAPairs data structure
    """
    try:
        with open(image_source_file, "rb") as imgs, open(label_source_file, "rb") as lbls:
            imgs = convert_from_file(imgs)
            lbls = convert_from_file(lbls)
    except:
        print("Error during processing of the images file")
        raise
    data = []
    for pair in list(zip(imgs, lbls)):
        data.append(QAPair(pair[0], pair[1]))
    return data
