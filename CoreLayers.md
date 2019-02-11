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

## Reshape层

```python
keras.layers.core.Reshape(target_shape)
```

可将输入的shape转换为特定的shape。

### 参数

- target_shape：目标shape，类型为整数元组，不包含样本数目维度（batch）。

### 输入

shape任意但必须固定，该层为模型第一层时需要指定`input_shape`。

### 输出

shape为`(batch_size, )+target_shape`。

### 例子

```python
# 作为顺序模型的第一层
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
# model.output_shape == (None, 3, 4) None is the batch dimension
# 作为顺序模型的中间层
model.add(Reshape(6, 2))
# model.output_shape == (None, 6, 2)
# 支持使用-1
model.add(Reshape((-1, 2, 2)))
# model.output_shape == (None, 3, 2, 2)
```

## Permute层

```python
keras.layers.core.Permute(dims)
```

Permute可将输入维度按照给定模式进行重排，如当RNN和CNN网络连接时，可能用到此层。

### 参数

- dims：整数元组，指定重排的模式，不含样本数维度。重排模式下标从1开始。如（2，1）代表将输入的第二个维度重排到输出的第一个维度，将输入的第一个维度重排到第二个维度。

### 例子

```python
model = Sequential()
model.add(Permute((2, 1), input_shape(10, 64)))
# model.output_shape == (None, 64, 10) None 为batch
```

### 输入

任意，该层为模型第一层时需要指定`input_shape`。

### 输出

shape与输入相同，但是维度按照指定模式重新排列。

## RepeatVector层

```python
keras.layers.core.RepeatVector(n)
```

RepeatVector层将输入重复n次

### 参数

- n：整数，重复次数

### 输入

shape为形如（nb_samples, features）的2D张量。

### 输出

shape为形如（nb_samples, n, features)的3D张量。

### 例子

```python
model = Sequential()
model.add(Dense(32, input_dim=32))
# now: model.output_shape == (None, 32)
# note: `None` is the batch dimension

model.add(RepeatVector(3))
# now: model.output_shape == (None, 3, 32)
```

## Lambda层

```python
keras.layers.core.Lambda(function, output_shape=None, mask=None, arguments=None)
```

本函数用以对上一层的输出施以任何Theano/TensorFlow表达式

### 参数

- function：要实现的函数，该函数仅接受一个变量，即上一层的输出
- output_shape：函数应该返回的值的shape，可以是一个tuple，也可以是一个根据输入shape计算输出shape的函数
- mask: 掩膜
- arguments：可选，字典，用来记录向函数中传递的其他关键字参数

### 例子

```python
# add a x -> x^2 layer
model.add(Lambda(lambda x: x ** 2))
# add a layer that returns the concatenation
# of the positive part of the input and
# the opposite of the negative part

def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] *= 2
    return tuple(shape)

model.add(Lambda(antirectifier,
         output_shape=antirectifier_output_shape))
```

### 输入shape

任意，当使用该层作为第一层时，要指定`input_shape`

### 输出shape

由`output_shape`参数指定的输出shape，当使用tensorflow时可自动推断

## ActivityRegularizer层

```python
keras.layers.core.ActivityRegularization(l1=0.0, l2=0.0)
```

经过本层的数据不会有任何变化，但会基于其激活值更新损失函数值

### 参数

- l1：1范数正则因子（正浮点数）
- l2：2范数正则因子（正浮点数）

### 输入shape

任意，当使用该层作为第一层时，要指定`input_shape`

### 输出shape

与输入shape相同

## Masking层

```python
keras.layers.core.Masking(mask_value=0.0)
```

使用给定的值对输入的序列信号进行“屏蔽”，用以定位需要跳过的时间步

对于输入张量的时间步，即输入张量的第1维度（维度从0开始算，见例子），如果输入张量在该时间步上都等于`mask_value`，则该时间步将在模型接下来的所有层（只要支持masking）被跳过（屏蔽）。

如果模型接下来的一些层不支持masking，却接受到masking过的数据，则抛出异常。

### 例子

考虑输入数据`x`是一个形如(samples,timesteps,features)的张量，现将其送入LSTM层。因为你缺少时间步为3和5的信号，所以你希望将其掩盖。这时候应该：

- 赋值`x[:,3,:] = 0.`，`x[:,5,:] = 0.`
- 在LSTM层之前插入`mask_value=0.`的`Masking`层

```python
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(32))
```

## SpatialDropout

