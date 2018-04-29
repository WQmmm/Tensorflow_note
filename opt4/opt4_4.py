#coding:utf-8
#opt4_3:交叉熵
#设损失函数 loss=(w+1)^2,令w的初值是常数5，反向传播就是求最优w，即求最小loss对应的w值
import tensorflow as tf
#定义待优化参数w初值是5
w = tf.Variable(tf.constant(5,dtype=tf.float32))
#定义损失函数loss
loss = tf.square(w+1)
#定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#生成会话，训练四十轮
with tf.Session() as sess:
	init_op = tf.initialize_all_variables()
	sess.run(init_op)
	for i in range(40):
		sess.run(train_step)
		w_val = sess.run(w)
		loss_val = sess.run(loss)
		print "Atfer %s steps:w is %f,    loss is %f." %(i,w_val,loss_val)
