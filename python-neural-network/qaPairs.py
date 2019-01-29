"""
Module designed to create a question-answer pairs data structure.
"""

def num_to_str(c):
    """ Converts a non-negative number (up to 3 digits) into a 3 digit string """
    if(c<10):
        return str(c)+"  "
    elif(c<100):
        return str(c)+" "
    else:
        return str(c)

class QAPairs:
    """ Data structure of question-answer pairs """
    def __init__(self, imgs, lbls):
        self.images = imgs/255 # Converting numbers between 0-255(int) to activations 0-1(float)
        self.labels = lbls
        return
    def get_label(self, n):
        try:
            return self.labels[n]
        except IndexError:
            print("There is no label with index {0}.".format(n))
            return 0
    def get_image(self, n):
        try:
            return self.images[n]
        except IndexError:
            print("There is no image with index {0}.".format(n))
            return 0
    def print_pair(self, n):
        print("Image:")
        img = (self.get_image(n)*255).astype(int) # Converting from activations 0-1(float) to numbers 0-255(int)
        if(img.any()):
            for i in range(28):
                row = ""
                for j in range(28):
                    row += num_to_str(img[i][j])
                    row += " "
                print(row)
        print("Label: "+str(self.get_label(n)))
        return
