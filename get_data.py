import numpy as np
import os
import PIL.Image as image

pic_path = 'D://gan_face/Cartoon_faces/faces'

class Get_data:
    def __init__(self):
        self.arr = []
        for filename in os.listdir(pic_path):
            pic = os.path.join('{0}/{1}'.format(pic_path,filename))
            self.arr.append(pic)

    def get_batch(self,batch):
        self.pic = []
        random_num = np.random.randint(len(self.arr))
        for i in range(batch):
            picture_path= self.arr[random_num]
            picture =image.open(picture_path)
            array =(np.array(picture)/255-0.5)/0.5
            self.pic.append(array)
        return self.pic
# data = Get_data()
# data.get_batch(2)