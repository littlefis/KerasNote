# 池化层

## MaxPooling1D层

```python
keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')
```

对时域1D信号进行最大值池化

### 参数

- pool_size：整数，池化窗口大小
- strides：整数或None，下采样因子，例如设2将会使得输出shape为输入的一半，若为None则默认值为pool_size。
- padding：‘valid’或者‘same’
- data_format：字符串，`channels_last`（默认）或者`channels_first`。前者为`(batch, steps, features)`，后者为`batch, features, steps)`。

### 输入shape

- If `data_format='channels_last'`: 3D tensor with shape: `(batch_size, steps, features)`
- If `data_format='channels_first'`: 3D tensor with shape: `(batch_size, features, steps)`

### 输出shape

- If `data_format='channels_last'`: 3D tensor with shape: `(batch_size, downsampled_steps, features)`
- If `data_format='channels_first'`: 3D tensor with shape: `(batch_size, features, downsampled_steps)`

------

## MaxPooling2D层

```python
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

为空域信号施加最大值池化

### 参数

- pool_size：整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半。为整数意为各个维度值相同且为该数字。
- strides：整数或长为2的整数tuple，或者None，步长值。
- border_mode：‘valid’或者‘same’
- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。`channels_last` 对应输入 `(batch, height, width, channels)` ， `channels_first` 对应输入`(batch, channels, height, width)`。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。

### 输入shape

- If `data_format='channels_last'`: 4D tensor with shape: `(batch_size, rows, cols, channels)`
- If `data_format='channels_first'`: 4D tensor with shape: `(batch_size, channels, rows, cols)`

### 输出shape

- If `data_format='channels_last'`: 4D tensor with shape: `(batch_size, pooled_rows, pooled_cols, channels)`
- If `data_format='channels_first'`: 4D tensor with shape: `(batch_size, channels, pooled_rows, pooled_cols)`

------

## MaxPooling3D层

```python
keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

为3D信号（空域或时空域）施加最大值池化

### 参数

- pool_size：整数或长为3的整数tuple，代表在三个维度上的下采样因子，如取（2，2，2）将使信号在每个维度都变为原来的一半长。
- strides：整数或长为3的整数tuple，或者None，步长值。
- padding：‘valid’或者‘same’
- data_format：字符串，“channels_first”或“channels_last”之一，代表数据的通道维的位置。`channels_last` corresponds to inputs with shape `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while `channels_first` corresponds to inputs with shape`(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。

### 输入shape

- If `data_format='channels_last'`: 5D tensor with shape: `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
- If `data_format='channels_first'`: 5D tensor with shape: `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

### 输出shape

- If `data_format='channels_last'`: 5D tensor with shape: `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
- If `data_format='channels_first'`: 5D tensor with shape: `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`

------

## AveragePooling1D层

```python
keras.layers.AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')
```

对时域1D信号进行平均值池化

### 参数

- pool_size：整数，池化窗口大小
- strides：整数或None，下采样因子，例如设2将会使得输出shape为输入的一半，若为None则默认值为pool_size。
- padding：‘valid’或者‘same’
- data_format：

### 输入shape

- If `data_format='channels_last'`: 3D tensor with shape: `(batch_size, steps, features)`
- If `data_format='channels_first'`: 3D tensor with shape: `(batch_size, features, steps)`

### 输出shape

- If `data_format='channels_last'`: 3D tensor with shape: `(batch_size, downsampled_steps, features)`
- If `data_format='channels_first'`: 3D tensor with shape: `(batch_size, features, downsampled_steps)`

------

## AveragePooling2D层

```python
keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

为空域信号施加平均值池化

### 参数

- pool_size：整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半。为整数意为各个维度值相同且为该数字。
- strides：整数或长为2的整数tuple，或者None，步长值。
- border_mode：‘valid’或者‘same’
- data_format：

### 输入shape

- If `data_format='channels_last'`: 4D tensor with shape: `(batch_size, rows, cols, channels)`
- If `data_format='channels_first'`: 4D tensor with shape: `(batch_size, channels, rows, cols)`

### 输出shape

- If `data_format='channels_last'`: 4D tensor with shape: `(batch_size, pooled_rows, pooled_cols, channels)`
- If `data_format='channels_first'`: 4D tensor with shape: `(batch_size, channels, pooled_rows, pooled_cols)`

------

## AveragePooling3D层

```python
keras.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

为3D信号（空域或时空域）施加平均值池化

### 参数

- pool_size：整数或长为3的整数tuple，代表在三个维度上的下采样因子，如取（2，2，2）将使信号在每个维度都变为原来的一半长。
- strides：整数或长为3的整数tuple，或者None，步长值。
- padding：‘valid’或者‘same’
- data_format：

### 输入shape

- If `data_format='channels_last'`: 5D tensor with shape: `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
- If `data_format='channels_first'`: 5D tensor with shape: `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

### 输出shape

- If `data_format='channels_last'`: 5D tensor with shape: `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
- If `data_format='channels_first'`: 5D tensor with shape: `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`

------

## GlobalMaxPooling1D层

```python
keras.layers.GlobalMaxPooling1D(data_format='channels_last')
```

对于时间信号的全局最大池化

### 输入shape

- If `data_format='channels_last'`: 3D tensor with shape: `(batch_size, steps, features)`
- If `data_format='channels_first'`: 3D tensor with shape: `(batch_size, features, steps)`

### 输出shape

- 2D tensor with shape: `(batch_size, features)`

------

## GlobalAveragePooling1D层

```python
keras.layers.GlobalAveragePooling1D(data_format='channels_last')
```

为时域信号施加全局平均值池化

### 输入shape

- If `data_format='channels_last'`: 3D tensor with shape: `(batch_size, steps, features)`
- If `data_format='channels_first'`: 3D tensor with shape: `(batch_size, features, steps)`

### 输出shape

- 2D tensor with shape: `(batch_size, features)`

------

## GlobalMaxPooling2D层

```python
keras.layers.GlobalMaxPooling2D(data_format=None)
```

为空域信号施加全局最大值池化

### 输入shape

- If `data_format='channels_last'`: 4D tensor with shape: `(batch_size, rows, cols, channels)`
- If `data_format='channels_first'`: 4D tensor with shape: `(batch_size, channels, rows, cols)`

### 输出shape

2D tensor with shape: `(batch_size, channels)`

------

## GlobalAveragePooling2D层

```python
keras.layers.GlobalAveragePooling2D(data_format=None)
```

为空域信号施加全局平均值池化

### 参数

- data_format：字符串

### 输入shape

- If `data_format='channels_last'`: 4D tensor with shape: `(batch_size, rows, cols, channels)`
- If `data_format='channels_first'`: 4D tensor with shape: `(batch_size, channels, rows, cols)`

### 输出shape

2D tensor with shape: `(batch_size, channels)`

------

## GlobalMaxPooling3D层

```python
keras.layers.GlobalMaxPooling3D(data_format=None)
```

### 输入shape

- If `data_format='channels_last'`: 5D tensor with shape: `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
- If `data_format='channels_first'`: 5D tensor with shape: `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

### 输出shape

2D tensor with shape: `(batch_size, channels)`

------

```python
keras.layers.GlobalAveragePooling3D(data_format=None)
```

同上