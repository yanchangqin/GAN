import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from get_data import Get_data

class D_Net:
    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal(shape=[3,3,3,64],dtype=tf.float32,stddev=0.02))
        self.b1 = tf.Variable(tf.zeros(shape=[64],dtype=tf.float32))
        self.w2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], dtype=tf.float32, stddev=0.02))
        self.b2 = tf.Variable(tf.zeros(shape=[128], dtype=tf.float32))
        self.w3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], dtype=tf.float32, stddev=0.02))
        self.b3 = tf.Variable(tf.zeros(shape=[256], dtype=tf.float32))
        self.w4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], dtype=tf.float32, stddev=0.02))
        self.b4 = tf.Variable(tf.zeros(shape=[512], dtype=tf.float32))
        self.w5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 1024], dtype=tf.float32, stddev=0.02))
        self.b5 = tf.Variable(tf.zeros(shape=[1024], dtype=tf.float32))
        self.w6 = tf.Variable(tf.truncated_normal(shape=[3,3,1024, 1], dtype=tf.float32, stddev=0.02))
        self.b6 = tf.Variable(tf.zeros(shape=[1], dtype=tf.float32))
    def forward(self,x):
        y1 = tf.nn.leaky_relu(tf.nn.conv2d(x,self.w1,strides=[1,2,2,1],padding='SAME')+self.b1)#48*48*64
        # print(y1.shape)
        y2 = tf.nn.leaky_relu(
            tf.layers.batch_normalization(tf.nn.conv2d(y1, self.w2, strides=[1, 2, 2, 1], padding='SAME') + self.b2))#24*24*128
        y3 = tf.nn.leaky_relu(
            tf.layers.batch_normalization(tf.nn.conv2d(y2, self.w3, strides=[1,2,2,1], padding='SAME') + self.b3))#12*12*256
        y4 = tf.nn.leaky_relu(
            tf.layers.batch_normalization(tf.nn.conv2d(y3, self.w4, strides=[1, 2, 2, 1], padding='SAME') + self.b4))#6*6*512
        y5 = tf.nn.leaky_relu(
            tf.layers.batch_normalization(tf.nn.conv2d(y4, self.w5, strides=[1, 2, 2, 1], padding='SAME') + self.b5))#3*3*1024
        y6 = tf.nn.leaky_relu(
            tf.layers.batch_normalization(
                tf.nn.conv2d(y5, self.w6, strides=[1, 1, 1, 1], padding='VALID') + self.b6))  # 3*3*1
        y6 = tf.reshape(y6,[-1,1])
        # print(y6.shape)
        return y6
    def params(self):
        return [self.w1,self.b1,self.w2,self.b2,self.w3,self.b3,self.w4,self.b4,self.w5,self.b5,self.w6,self.b6]

class G_Net:
    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal(shape=[1024,3*3*1024],dtype=tf.float32,stddev=0.02))
        self.b1 = tf.Variable(tf.zeros(shape=[3*3*1024], dtype=tf.float32))
        self.w2 = tf.Variable(tf.truncated_normal(shape=[3,3,512,1024], dtype=tf.float32, stddev=0.02))
        self.b2 = tf.Variable(tf.zeros(shape=[512], dtype=tf.float32))
        self.w3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], dtype=tf.float32, stddev=0.02))
        self.b3 = tf.Variable(tf.zeros(shape=[256], dtype=tf.float32))
        self.w4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], dtype=tf.float32, stddev=0.02))
        self.b4 = tf.Variable(tf.zeros(shape=[128], dtype=tf.float32))
        self.w5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], dtype=tf.float32, stddev=0.02))
        self.b5 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32))
        self.w6 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], dtype=tf.float32, stddev=0.02))
        self.b6 = tf.Variable(tf.zeros(shape=[3], dtype=tf.float32))
    def forward(self,x):
        y1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(x,self.w1)+self.b1))
        y1 = tf.reshape(y1,[-1,3,3,1024])
        y2 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d_transpose(y1,self.w2,[100,6,6,512],strides=[1,2,2,1],padding="SAME")+self.b2))
        y3 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d_transpose(y2, self.w3, [100, 12, 12, 256], strides=[1, 2, 2, 1], padding="SAME") + self.b3))
        y4 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d_transpose(y3, self.w4, [100, 24, 24, 128], strides=[1, 2, 2, 1], padding="SAME") + self.b4))
        y5 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d_transpose(y4, self.w5, [100, 48, 48, 64], strides=[1, 2, 2, 1], padding="SAME") + self.b5))
        y6 = tf.tanh(tf.nn.conv2d_transpose(y5, self.w6, [100, 96, 96, 3], strides=[1, 2, 2, 1], padding="SAME") + self.b6)
        return y6
    def params(self):
        return [self.w1,self.b1,self.w2,self.b2,self.w3,self.b3,self.w4,self.b4,self.w5,self.b5,self.w6,self.b6]

class Net:
    def __init__(self):
        self.real_x = tf.placeholder(dtype=tf.float32,shape=[None,96,96,3])
        # print(self.real_x)
        self.fake_x = tf.placeholder(dtype=tf.float32,shape=[None,1024])

        self.real_label = tf.placeholder(dtype=tf.float32,shape=[None,1])
        self.fake_label = tf.placeholder(dtype=tf.float32, shape=[None,1])

        self.g_net = G_Net()
        self.d_net = D_Net()
    def forward(self):
        self.g_out = self.g_net.forward(self.fake_x)
        self.D_fake_out = self.d_net.forward(self.g_out)
        self.D_real_out = self.d_net.forward(self.real_x)
    def loss(self):
        self.fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.fake_label,logits=self.D_fake_out))
        self.real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.real_label,logits=self.D_real_out))
        self.D_loss = self.fake_loss+self.real_loss

        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.fake_label,logits=self.D_fake_out))
    def backward(self):
        self.D_optimizer = tf.train.AdamOptimizer(0.0002,0.5).minimize(self.D_loss,var_list=self.d_net.params())
        self.G_optimizer = tf.train.AdamOptimizer(0.0002,0.5).minimize(self.G_loss,var_list=self.g_net.params())

if __name__ == '__main__':
    net = Net()
    net.forward()
    net.loss()
    net.backward()
    data = Get_data()
    init  = tf.global_variables_initializer()
    save = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        j=0
        for i in range(20000):
            real_x = data.get_batch(100)
            real_label = np.ones(shape=[100,1])

            fake_x = np.random.uniform(-1,1,(100,1024))
            fake_label = np.zeros(shape=[100,1])
            d_loss,_ = sess.run([net.D_loss,net.D_optimizer],feed_dict={net.fake_x:fake_x,net.fake_label:fake_label,net.real_x:real_x,net.real_label:real_label})
            fake_xs = np.random.uniform(-1,1,(100,1024))
            fake_labels = np.ones(shape=[100,1])
            g_loss,_ = sess.run([net.G_loss, net.G_optimizer], feed_dict={net.fake_x:fake_xs,net.fake_label:fake_labels})
            if i%10 ==0:
                fake_xss = np.random.uniform(-1,1,(100,1024))
                array = sess.run(net.g_out,feed_dict={net.fake_x:fake_xss})
                img = (np.reshape(array[0],[96,96,3])*0.5+0.5)*255
                plt.imshow(img)
                plt.pause(0.1)
                print('训练次数：',j)
                print('d_loss:',d_loss)
                print('g_loss:',g_loss)
                j+=1
                save.save(sess,'my_net/save_net.ckpt')