# 常用层

## Dense层

```python
keras.layers.core.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

全连接层，实现`output=activation(dot(input, kernel)+bias)`。`activation`是逐元素计算的激活函数，`kernel`是本层的权值矩阵，`bias`是偏置向量，仅`use_bias=True`时才会添加。

输入维度大于2时会先被压成与`kernel`匹配的大小。

```python
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
# input (*, 16) output (*, 32)
model.add(Dense(32))
# 第一层之后的各层不需要指定输入大小
```

### 参数

- units：输出维度，正整数。
- activation：激活函数，可以是预定义的激活函数名或逐元素的Theano函数。不指定该参数时不使用激活函数。
- use_bias：布尔，是否使用偏置。
- kernel_initializer：权值初始化方法，可以是预定义的初始化方法名（字符串），或初始化器。
- bias_initializer：偏置向量初始化方法，同上。
- kernel_regularizer：权重正则项，为Regularizer对象。
- bias_regularizer：偏置正则项，同上。
- activity_regularizer：输出正则项，同上。
- kernel_constraints：权重约束项，为Constraints对象。
- bias_constraints：偏置约束项，同上。

### 输入

形如(batch_size, ..., input_dim)的nD张量，最常见情况为(batch_size, input_dim)的2D张量。

### 输出

形如(batch_size, ..., units)的nD张量，最常见情况为(batch_size, units)的2D张量。

## Activation层

```python
keras.layers.core.Activation(activation)
```

激活层对一个层的输出使用激活函数。

### 参数

- activation：使用的激活函数，可以是预定义的激活函数名或Tensorflow/Theano函数。

### 输入

shape任意，使用激活层作为第一层时需要指定`input_shape`。

### 输出

shape与输入相同。

## Dropout层

```python
keras.layers.core.Dropout(rate, noise_shape=None, seed=None)
```

为输入数据使用Dropout，在训练过程中每次更新参数时按照一定概率随机断开输入神经元，防止过拟合。

### 参数

- rate：0至1的浮点数，控制断开神经元的比例。
- noise_shape：整数张量，
- seed：整数，使用的随机数种子。

## Flatten层

```python
keras.layers.core.Flatten()
```

用于将输入层压扁，把多维的输入一维化，常用在卷积层到全连接层的过渡。该层不影响batch大小。

```python
model = Sequential()
model.add(Convolution2D(64, 3, 3,
                       border_mode='same',
                       input_shape=(3, 32, 32)))
# model.output_shape == (None, 64, 32, 32)
model.add(Flatten())
# model.output_shape == (None, 65536)
```





















