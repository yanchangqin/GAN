import os
import tensorflow as tf
from PIL import Image
import numpy as np

test_dir = "D://gan_face/Cartoon_faces/faces"
class MyDataset:
    def read_file(self):
        self.filenames = []
        for filename in os.listdir(test_dir):
            pic = os.path.join('{0}/{1}'.format(test_dir, filename))
            self.filenames.append(pic)

    def _parse_function(self,filename):
        self.image_string = tf.read_file(filename)
        image_decode = tf.image.decode_image(self.image_string)

        return image_decode

    def get_batch(self,set):
        data_filenames = tf.constant(self.filenames)
        data_set1 = tf.data.Dataset.from_tensor_slices(data_filenames)
        data_set2 = data_set1.map(self._parse_function)
        data_set3 = data_set2.repeat()
        data_set4 = data_set3.shuffle(set)
        batch_data_set = data_set4.batch(set)
        iterator = batch_data_set.make_one_shot_iterator()
        batch = iterator.get_next()
        with tf.Session() as sess:
            image = sess.run(batch)
            img = (image / 255 - 0.5) * 2
            print(img)
            return img

mydata = MyDataset()
mydata.read_file()
mydata.get_batch(1)



