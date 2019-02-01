# 顺序模型 Sequential

##  构建模型

- 向`Sequential`模型中传递一个layer的list

    ```python
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    
    model = Sequential([
        Dense(32, units=784),
        Activation('relu'),
        Dense(10),
        Activation('softmax'),
    ])
    ```

- 通过`.add()`方法添加layer

  ```python
  model = Sequential()
  model.add(Dense(32, input_shape=(784,)))
  model.add(Activation('relu'))
  ```

### 指定输入数据shape

第一层需要指定shape参数，后面的各层可以自动推导出中间数据的shape。指定方法：

- 传递`input_shape`参数，该参数为tuple类型，可以填`None`，代表可能是任何正整数。数据的batch大小不应包含在其中。
- 2D层如`Dense`可通过`input_dim`来隐含指定输入shape，参数类型为int。3D层可以通过`input_dim`和`imput_length`来指定shape。
- 指定batch_size时可以传递`batch_size` 参数，例如batch大小是32，shape是(6, 8)，则传递`batch_size=32 ` 和`input_shape=(6, 8)` 。

## 编译

模型训练前通过`compile`进行配置，`compile`接受三个参数：

- `optimizer`优化器：可指定为预定义的优化器名，如`rmsprop`，`adagrad`，或一个`Optimizer`类的对象。
- `loss`损失函数：模型试图最小化的目标函数，可指定为预定义的损失函数名，如`categorical_crossentropy`、`mse`，也可以为一个损失函数。
- `metrics`指标列表：分类问题一般设置为`metrics=['accuracy']`。指标可以为预定义的指标名，也可以是用户定义的函数。指标函数返回值应为单个张量，或完成`metric_name -> metric_value`映射的字典。

```python
# 多类别分类
model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
# 二分类
model.compile(optimizer='rmsprop', 
             loss='binary_crossentropy',
             metrics=['accuracy'])
# 均方误差回归
model.compile(optimizer='rmsprop',
             loss='mse')
# 自定义指标
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy', mean_pred])
```

## 训练

Keras训练以Numpy数组作为输入数据和标签数据类型，训练模型一般使用`fit`函数。

```python
# 二分类
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy'])
# 生成数据
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# 训练模型
model.fit(data, labels, epochs=10, batch_size=32)
```

```python
# 多分类
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
# 生成数据
import numpy as np
data = np.random.random((1000, 100))
labels = np.random/randint(10, size=(1000,1))
# 将标签转化为独热码
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
# 训练
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
```

## 例子

Keras代码包的exampls文件夹

