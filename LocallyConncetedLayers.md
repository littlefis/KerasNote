# 局部连接层LocallyConnceted

## LocallyConnected1D层

```python
keras.layers.LocallyConnected1D(filters, kernel_size, strides=1, padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

`LocallyConnected1D`层与`Conv1D`工作方式类似，唯一的区别是不进行权值共享。即施加在不同输入位置的滤波器是不一样的。

### 参数

- filters：卷积核的数目（即输出的维度）
- kernel_size：整数或由单个整数构成的list/tuple，卷积核的空域或时域窗长度
- strides：整数或由单个整数构成的list/tuple，为卷积的步长。任何不为1的strides均与任何不为1的dilation_rata均不兼容
- padding：补0策略，目前仅支持`valid`（大小写敏感），`same`可能会在将来支持。
- data_format：字符串
- activation：激活函数，为预定义的激活函数名（参考[激活函数](https://keras-cn.readthedocs.io/en/latest/other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）

### 输入shape

3D tensor with shape: `(batch_size, steps, input_dim)`

### 输出shape

形如(batch_size, new_steps, filters)的3D张量，因为有向量填充的原因，`steps`的值会改变

------

## LocallyConnected2D层

```python
keras.layers.LocallyConnected2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

`LocallyConnected2D`层与`Convolution2D`工作方式类似，唯一的区别是不进行权值共享。即施加在不同输入patch的滤波器是不一样的，当使用该层作为模型首层时，需要提供参数`input_dim`或`input_shape`参数。参数含义参考`Convolution2D`。

### 参数

- filters：卷积核的数目（即输出的维度）
- kernel_size：单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
- strides：单个整数或由两个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。
- padding：补0策略，目前仅支持`valid`（大小写敏感），`same`可能会在将来支持。

### 输入shape

- 4D tensor with shape: `(samples, channels, rows, cols)` if data_format='channels_first' 
- 4D tensor with shape: `(samples, rows, cols, channels)` if data_format='channels_last'.

### 输出shape

- 4D tensor with shape: `(samples, filters, new_rows, new_cols)` if data_format='channels_first' 

- 4D tensor with shape: `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
-  `rows` and `cols` values might have changed due to padding.

### 例子

```python
# 1D
# apply a unshared weight convolution 1d of length 3 to a sequence with
# 10 timesteps, with 64 output filters
model = Sequential()
model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
# now model.output_shape == (None, 8, 64)
# add a new conv1d on top
model.add(LocallyConnected1D(32, 3))
# now model.output_shape == (None, 6, 32)

# 2D
# apply a 3x3 unshared weights convolution with 64 output filters
# on a 32x32 image with `data_format="channels_last"`:
model = Sequential()
model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
# now model.output_shape == (None, 30, 30, 64)
# notice that this layer will consume (30*30)*(3*3*3*64)
# + (30*30)*64 parameters

# add a 3x3 unshared weights convolution on top, with 32 output filters:
model.add(LocallyConnected2D(32, (3, 3)))
# now model.output_shape == (None, 28, 28, 32)
```