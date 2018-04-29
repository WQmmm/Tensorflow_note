#coding:utf-8
#opt3_1:前向传播（浅层神经网络）
 
#1.导入模块
import tensorflow as tf
 
#2.定义参数
x = tf.constant([[1.0,2.0]])
w = tf.constant([[3.0],[4.0]])
 
#3.前向传播过程
y = tf.matmul(x,w)
print y
 
#4.生成会话
with tf.Session() as sess:
	print sess.run(y)
	
