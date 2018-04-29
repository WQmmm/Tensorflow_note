#coding:utf-8
#opt4_5:学习率
#设损失函数 loss=(w+1)^2,令w的初值是常数5，反向传播就是求最优w，及最小loss对应的w值
import tensorflow as tf 
#定义待优化参数w初值赋5
w = tf.Variable(tf.constant(5,dtype=tf.float32))
#定义损失函数loss
loss = tf.square(w+1)
#定义反向传播算法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#生成会话，训练40轮
with tf.Session() as sess:
	init_op = tf.initialize_all_variables()
	sess.run(init_op)
	for i in range(40):
		sess.run(train_step)
		w_val = sess.run(w)
		loss_val = sess.run(loss)
		print "After %s stop: w is %f, loss is %f." %(i,w_val,loss_val) 
