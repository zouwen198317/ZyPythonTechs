# RNN LSTM (回归例子可视化) #

## matplotlib 可视化  ##
使用 Matplotlib 模块来进行可视化过程, 在建立好 model 以后, 设置 plt.ion() 使 plt.show()可以连续显示.

	model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
	sess = tf.Session()
	# sess.run(tf.initialize_all_variables()) # tf 马上就要废弃这种写法
	# 替换成下面的写法:
	sess.run(tf.global_variables_initializer())
	plt.ion()   # 设置连续 plot
	plt.show()
然后在 sess.run() 后面加上plt.draw()的步骤.

	_, cost, state, pred = sess.run(
	            [model.train_op, model.cost, model.cell_final_state, model.pred],
	            feed_dict=feed_dict)
	# plotting
	plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
	plt.ylim((-1.2, 1.2))
	plt.draw()
	plt.pause(0.3)  # 每 0.3 s 刷新一次
最后的结果显示为:

![](https://morvanzhou.github.io/static/results/tensorflow/5_10_1.png)