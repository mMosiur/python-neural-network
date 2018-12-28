"""
Module designed to create a question-answer pairs data structure.
"""

def num_to_str(c):
    """Converts a non-negative number (up to 3 digits) into a 3 digit string"""
    if(c<10):
        return str(c)+"  "
    elif(c<100):
        return str(c)+" "
    else:
        return str(c)

class QAPairs:
    """Data structure of question-answer pairs"""
    def __init__(self, imgs, lbls):
        self.images = imgs
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
        img = self.get_image(n)
        if(img.any()):
            for i in range(28):
                row = ""
                for j in range(28):
                    row += num_to_str(img[i][j])
                    row += " "
                print(row)
        print("Label: "+str(self.get_label(n)))
        return