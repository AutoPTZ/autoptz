import os


# reads all the files in the /negative folder and generates neg.txt from them.
def generate_negative_description_file():
    # open the output file for writing. will overwrite all existing data in there
    with open('neg.txt', 'w') as f:
        # loop over all the filenames
        for filename in os.listdir('negatives'):
            f.write('negatives/' + filename + '\n')


def generate_positive_description_file():
    # open the output file for writing. will overwrite all existing data in there
    with open('pos.txt', 'w') as f:
        # loop over all the filenames
        for filename in os.listdir('positives'):
            f.write('positives/' + filename + " 1" + '\n')


if __name__ == '__main__':
    generate_negative_description_file()
    generate_positive_description_file()

#create annotation
##logic\old_cascade_creator\utils\opencv_annotation.exe --annotations=logic\old_cascade_creator\pos.txt --images=logic\old_cascade_creator\positives\

