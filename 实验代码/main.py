#coding=utf-8
import numpy
import csv
import tensorflow as tf

train_pixel = [] # 训练集
train_label = [] # 训练标签
test_pixel = [] # 测试集
train_count = -2 # 训练集大小
validation_count = -2 # 验证集大小
test_count = -2 # 测试集大小
now = 0 # 全局计数器

# 加载训练集
def load_train_data():
    global train_pixel, train_label, train_count, validation_count
    # 从train.csv文件读取数据，train.csv每行为785个整数，第1个为标签，后面784个为对应像素灰度（0-255）
    csvfile = file('train.csv', 'rb')
    reader = csv.reader(csvfile)
    for line in reader:
        train_count = train_count + 1
        if (train_count == -1):
            continue
        # 将标签转换为对应的10维向量
        train_label.append(numpy.zeros(10))
        train_label[train_count][int(line[0])] = 1.0
        train_pixel.append([])
        for i in xrange(1, len(line)):
            train_pixel[train_count].append(float(line[i]) / 255) # 灰度归一化
    csvfile.close()
    # 从训练集中抽取部分作为验证集
    validation_count = int(0.0125 * train_count);
    train_count -= validation_count;

# 加载测试集
def load_test_data():
    global test_pixel, test_count
    # 从test.csv文件读取数据，test.csv每行为784个整数，为对应像素灰度（0-255）
    csvfile = file('test.csv', 'rb')
    reader = csv.reader(csvfile)
    for line in reader:
        test_count = test_count + 1
        if (test_count == -1):
            continue
        test_pixel.append([])
        for i in xrange(0, len(line)):
            test_pixel[test_count].append(float(line[i]) / 255) # 灰度归一化
    csvfile.close()

# 保存结果
def save_result(result):
    # 保存至result.csv文件
    csvfile = file('result.csv', 'wb')
    writer = csv.writer(csvfile)
    # csv文件表头
    writer.writerow(['ImageId', 'Label'])
    # 每行保存序号（从1开始）和对应结果
    for i in xrange(0, len(result)):
        writer.writerow([i + 1, result[i]])
    csvfile.close()

# 获取下一批训练集，大小为num
def next_batch(num):
    global now
    label = numpy.array(train_label[now: now + num])
    pixel = numpy.array(train_pixel[now: now + num])
    now = now + num
    # 如果训练集全部用完，重新开始用
    if (now > train_count): now = now - train_count
    return (pixel, label)

# 获取验证集
def validation_set():
    label = numpy.array(train_label[train_count: train_count + validation_count])
    pixel = numpy.array(train_pixel[train_count: train_count + validation_count])
    return (pixel, label)

# 随机初始化大小为shape的神经元
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

# 生成大小为shape的偏置变量
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# 卷积核
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

# 2x2最大池化
def max_pool(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

# 训练神经网络
def train_nn():
    sess = tf.InteractiveSession() # 建立session

    x = tf.placeholder(tf.float32, shape = [None, 784]) # 神经网络输入
    y_ = tf.placeholder(tf.float32, shape = [None, 10]) # 神经网络输出
    x_image = tf.reshape(x, [-1, 28, 28, 1]) # 将784像素重置为28x28像素
    
    W_conv1 = weight_variable([5, 5, 1, 32]) # 第一层卷积层
    b_conv1 = bias_variable([32]) # 第一层偏置
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # 第一层卷积，relu激活
    h_pool1 = max_pool(h_conv1) # 第一层2x2池化
    
    W_conv2 = weight_variable([5, 5, 32, 64]) # 第二层卷积层
    b_conv2 = bias_variable([64]) # 第二层偏置
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # 第二层卷积，relu激活
    h_pool2 = max_pool(h_conv2) # 第二层2x2池化
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64]) # 将第二层重置shape

    W_fc1 = weight_variable([7 * 7 * 64, 1024]) # 第三层全连接层
    b_fc1 = bias_variable([1024]) # 第三层偏置
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # 第三层计算，relu激活
    
    keep_prob = tf.placeholder(tf.float32) # 保持概率
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # 适度丢弃

    W_fc2 = weight_variable([1024, 10]) # 第四层全连接层
    b_fc2 = bias_variable([10]) # 第四层偏置
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 # 第四层计算，保留原始计算结果
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv)) # 交叉熵计算
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # 最小化交叉熵
    prediction = tf.argmax(y_conv, 1) # 预测结果
    correct_prediction = tf.equal(prediction, tf.argmax(y_, 1)) # 结果是否正确
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 准确率
    
    sess.run(tf.global_variables_initializer()) # 初始化所有变量
    saver = tf.train.Saver() # 参数保存/加载器
        
    with tf.device('/cpu:0'): # 指定在某设备上运行（如CPU、显卡）
        best_accuracy = 0.0 # 在验证集最高准确率
        best_num = 0 # 对应最高准确率的训练次数
        for i in range(1000): # 训练50000次
            batch = next_batch(50) # 每次提供50个训练样本
            # 每训练500次用验证集测试准确率，若高于最高准确率，记录并保存结果
            if i % 500 == 0: 
                [validation_pixel, validation_label] = validation_set()
                train_accuracy = accuracy.eval(feed_dict = {x: validation_pixel, y_: validation_label, keep_prob: 1.0})
                if (train_accuracy >= best_accuracy):
                    best_accuracy = train_accuracy
                    best_num = i
                    saver.save(sess, './model')
                print("step %d, training accuracy %g, best accuracy %g, best num %d" % (i, train_accuracy, best_accuracy, best_num))
            train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5}) # 训练
        
        load_path = saver.restore(sess, './model') # 加载训练最好结果
        result = [] # 结果向量
        # 每次记录500个结果，避免消耗过多资源（尤其是使用显卡时）
        i = 0
        while (i <= test_count):
            result[i: i + 500] = prediction.eval(feed_dict = {x: test_pixel[i: i + 500], keep_prob: 1.0})
            i += 500
        save_result(result) # 保存结果文件

'''
    # LaNet-5模型，如果换用模型直接把上述代码替换为下段代码即可

    # 第一卷积层
    W_conv1 = weight_variable([5, 5, 1, 6])
    b_conv1 = bias_variable([6])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二卷积层
    W_conv2 = weight_variable([5, 5, 6, 16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 第一全连接层
    W_fc1 = weight_variable([4 * 4 * 16, 120])
    b_fc1 = bias_variable([120])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 * 16])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 第二全连接层
    W_fc2 = weight_variable([120, 84])
    b_fc2 = bias_variable([84])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    
    # 第三全连接层
    W_fc3 = weight_variable([84, 10])
    b_fc3 = bias_variable([10])
    y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
'''

# 主程序
def main():
    load_train_data() # 加载训练集
    load_test_data() # 加载测试集
    train_nn() # 训练神经网络

if __name__ == '__main__':
    main()

