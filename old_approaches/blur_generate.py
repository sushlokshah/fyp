import numpy
import os
import torch
from PIL import Image

# get argument dataset path
if __name__ == '__main__':

    # set number of merge frames
    merge_frames = 8
    step_size = 5
    show_one = True

    data_root = 'C:\\Users\\Machine Learning GPU\Desktop\\GOPRO_Large_all(2)\\test'
    saving = 'C:\\Users\\Machine Learning GPU\\Desktop\\created'
    # list all sequence folders in train dir
    for sequences in os.listdir(data_root):
        seq_root = os.path.join(data_root, sequences)
        img_list = os.listdir(seq_root)
        img_list = [os.path.join(seq_root, img) for img in img_list]

        print(len(img_list))
        print(sequences)

        for i in range(0, len(img_list)-merge_frames, step_size):
            print(i)
            # read n images and add them
            for j in range(merge_frames):
                img = numpy.asarray(Image.open(img_list[i+j]))/merge_frames
                if j == 0:
                    img_merge = img
                else:
                    img_merge = img_merge + img
            img_merge = Image.fromarray(numpy.uint8(img_merge))
            if not os.path.exists(os.path.join(saving, sequences)):
                os.mkdir(os.path.join(saving, sequences))
            img_merge.save(os.path.join(saving, sequences,
                           'merge_{}_{}.png'.format(i, i+merge_frames)))

            # if show_one:
            #     print(img_merge.min(), img_merge.max(), img_merge.mean())
            #     img_merge = Image.fromarray(numpy.uint8(img_merge))
            #     img_merge.show()

            #     show_one = False
