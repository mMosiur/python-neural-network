#!/usr/bin/env python
import numpy as np
import idx2numpy
import misc

class qaPairs:
    def __init__(self, imgs, lbls):
        self.images = imgs
        self.labels = lbls
        return
    def get_label(self, n):
        return self.labels[n]
    def get_image(self, n):
        return self.images[n]
    def print_pair(self, n):
        print("Image:")
        img = self.get_image(n)
        for i in range(28):
            row = ""
            for j in range(28):
                row += misc.num_to_str(img[i][j])
                row += " "
            print(row)
        print("Label: "+str(self.get_label(n)))
        return

def import_data(image_source_file, label_source_file):
    misc.log("Data import")
    #------------------------------------------------#
    try:
        misc.log("Images and labels import", 1)
        with open(image_source_file, "rb") as imgs, open(label_source_file, "rb") as labs:
            ndarr_imgs = idx2numpy.convert_from_file(imgs)
            ndarr_labs = idx2numpy.convert_from_file(labs)
    except:
        misc.log("Error during processing of the images file", 1)
        raise
    #------------------------------------------------#
    misc.log("Data import successful")
    return qaPairs(ndarr_imgs, ndarr_labs)

images_source = "resources/train-images.idx3-ubyte"
labels_source = "resources/train-labels.idx1-ubyte"
data = import_data(images_source,labels_source)

for i in range(2):
    data.print_pair(i)
    print("\n")

misc.log("End of program")