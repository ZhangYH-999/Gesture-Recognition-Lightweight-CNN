from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pickle
from read_data import read_gesture


# 超参数
batch_size = 100
learning_rate = 0.01
n_inputs = 10000
n_outputs = 10
n_classes = 10
n_epochs = 100

# 图片信息参数
img_height = 100
img_width = 100
img_depth = 1
num_img = 2062

if __name__ == '__main__':
    # 读取手势数据
    X, y = read_gesture()   # X(2062, 100, 100)  y(2062, )
    y = y.reshape([-1, 1])
    X = X.reshape([-1, 100, 100, 1])
    # X(2062, 100, 100, 1), y(2062, 1)

    # 独热编码
    y = to_categorical(y, num_classes=10)
    print("X shape:", X.shape, "y shape", y.shape)

    # 归一化
    X /= 255

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # 定义每层滤波器数量
    n_filters = [8, 16]

    # 构建序列化模型 2062 100 100
    model = Sequential()

    model.add(Conv2D(filters=n_filters[0], kernel_size=4, padding='SAME', activation='relu'))
    model.add(MaxPooling2D(pool_size=(8, 8), strides=(4, 4)))
    model.add(Conv2D(filters=n_filters[1], kernel_size=2, padding='SAME', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(units=n_outputs, activation='softmax'))

    # 编译模型
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer='adam'
                  )

    # 训练模型
    model_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, validation_data=(x_test, y_test))

    model.summary()
    # 评估模型比分输出准确分数
    scores = model.evaluate(x_test, y_test)
    print('\n loss: ', scores[0])
    print('\n accuracy: ', scores[1])

    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.legend(['train_acc', 'test_acc'])
    plt.title('train_acc and test_acc')
    plt.grid()
    plt.show()

    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.legend(['train_loss', 'test_loss'])
    plt.title('train_loss and test_loss')
    plt.show()

    select = str(input("Whether save model? (y/n)"))
    if select == 'y':
        model.save("gesture_model.h5")
        print("Successfully")
