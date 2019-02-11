# 卷积层

## Conv1D层

```python
keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

一维卷积层（时域卷积），用以在一维输入信号上进行邻域滤波。当使用该层作为首层时，需要提供关键字参数`input_shape`。例如`(10,128)`代表一个长为10的序列，序列中每个信号为128向量。而`(None, 128)`代表变长的128维向量序列。

该层生成将输入信号与卷积核按照单一的空域（或时域）方向进行卷积。如果`use_bias=True`，则还会加上一个偏置项，若`activation`不为None，则输出为经过激活函数的输出。

### 参数

- filters：卷积核的数目（即输出的维度）
- kernel_size：整数或由单个整数构成的list/tuple，卷积核的空域或时域窗（1D卷积窗口）长度
- strides：整数或由单个整数构成的list/tuple，为卷积的步长。任何不为1的strides与任何不为1的dilation_rate均不兼容
- padding：补0策略，为“valid”, “same” 或“causal”，“causal”将产生因果（膨胀的）卷积，即output[t]不依赖于input[t+1：]。当对不能违反时间顺序的时序信号建模时有用。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同
- data_format：字符串，`channels_last`（默认）或`channels_first`。输入数据的维度顺序，last对应的输入为（batch，steps，channels）（Keras中的时间数据默认格式），first对应（batch，channels，steps）
- activation：激活函数，为预定义的激活函数名，或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
- dilation_rate：整数或由单个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容
- use_bias:布尔值，是否使用偏置项
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。
- kernel_regularizer：施加在权重上的正则项，为Regularizer对象
- bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
- activity_regularizer：施加在输出上的正则项，为Regularizer对象
- kernel_constraints：施加在权重上的约束项，为Constraints对象
- bias_constraints：施加在偏置上的约束项，为Constraints对象

### 输入shape

形如（samples，steps，input_dim）的3D张量

### 输出shape

形如（samples，new_steps，nb_filter）的3D张量，因为有向量填充的原因，`steps`的值会改变

 

## Conv2D层

```python
keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

二维卷积层，即对图像的空域卷积。该层对二维输入进行滑动窗卷积，当使用该层作为第一层时，应提供`input_shape`参数。例如`input_shape = (128,128,3)`代表128*128的彩色RGB图像（`data_format='channels_last'`）

### 参数

- filters：卷积核的数目（即输出的维度）
- kernel_size：单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
- strides：单个整数或由两个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
- padding：补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
- activation：激活函数，为预定义的激活函数名，或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
- dilation_rate：单个整数或由两个个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。
- use_bias:布尔值，是否使用偏置项
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。
- kernel_regularizer：施加在权重上的正则项，为Regularizer对象
- bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
- activity_regularizer：施加在输出上的正则项，为Regularizer对象
- kernel_constraints：施加在权重上的约束项，为Constraints对象
- bias_constraints：施加在偏置上的约束项，为Constraints对象

### 输入shape

`channels_first`模式下，输入形如（batch, channels, rows, cols）的4D张量

`channels_last`模式下，输入形如（batch, rows, cols, channels）的4D张量

### 输出shape

`channels_first`模式下，为形如（batch, filters, new_rows, new_cols）的4D张量

`channels_last`模式下，为形如（batch, new_rows, new_cols, filters）的4D张量

输出的行列数可能会因为填充方法而改变

 

## SeparableConv1D层

```python
keras.layers.SeparableConv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
```

该层是在深度方向上的可分离卷积。

可分离卷积首先按深度方向进行卷积（对每个输入通道分别卷积），然后逐点进行卷积，将上一步的卷积结果混合到输出通道中。参数`depth_multiplier`控制了在depthwise卷积（第一步）的过程中，每个输入通道信号产生多少个输出通道。

直观来说，可分离卷积可以看做讲一个卷积核分解为两个小的卷积核，或看作Inception模块的一种极端情况。

当使用该层作为第一层时，应提供`input_shape`参数。例如`input_shape = (3,128,128)`代表128*128的彩色RGB图像

### 参数

- filters：卷积核的数目（即输出的维度）
- kernel_size：单个整数或由两个个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
- strides：单个整数或由两个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
- padding：补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
- activation：激活函数，为预定义的激活函数名，或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
- dilation_rate：单个整数或由两个个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。
- use_bias:布尔值，是否使用偏置项
- depth_multiplier：在按深度卷积的步骤中，每个输入通道使用多少个输出通道
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。
- depthwise_regularizer：施加在按深度卷积的权重上的正则项，为Regularizer对象
- pointwise_regularizer：施加在按点卷积的权重上的正则项，为Regularizer对象
- kernel_regularizer：施加在权重上的正则项，为Regularizer对象
- bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
- activity_regularizer：施加在输出上的正则项，为Regularizer对象
- kernel_constraints：施加在权重上的约束项，为Constraints对象
- bias_constraints：施加在偏置上的约束项，为Constraints对象
- depthwise_constraint：施加在按深度卷积权重上的约束项，为Constraints对象
- pointwise_constraint施加在按点卷积权重的约束项，为Constraints对象

### 输入shape

3D tensor with shape: `(batch, channels, steps)` if `data_format` is `"channels_first"` or 3D tensor with shape: `(batch, steps, channels)` if `data_format` is `"channels_last"`.

### 输出shape

3D tensor with shape: `(batch, filters, new_steps)` if `data_format` is `"channels_first"` or 3D tensor with shape: `(batch, new_steps, filters)` if `data_format` is `"channels_last"`. `new_steps` values might have changed due to padding or strides.

## SeparableConv2D

与1D类似

 ## DepthwiseConv2D



## Conv2DTranspose层

```python
keras.layers.Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

该层是转置的卷积操作（反卷积）。需要反卷积的情况通常发生在用户想要对一个普通卷积的结果做反方向的变换。例如，将具有该卷积层输出shape的tensor转换为具有该卷积层输入shape的tensor。同时保留与卷积层兼容的连接模式。

当使用该层作为第一层时，应提供`input_shape`参数。例如`input_shape=(128, 128, 3)`对于128x128 RGB图片`data_format="channels_last"`。

### 参数

- filters：卷积核的数目（即输出的维度）
- kernel_size：单个整数或由两个个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
- strides：单个整数或由两个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
- padding：补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
- activation：激活函数，为预定义的激活函数名，或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
- dilation_rate：单个整数或由两个个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。
- use_bias:布尔值，是否使用偏置项
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。
- kernel_regularizer：施加在权重上的正则项，为Regularizer对象
- bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
- activity_regularizer：施加在输出上的正则项，为Regularizer对象
- kernel_constraints：施加在权重上的约束项，为Constraints对象
- bias_constraints：施加在偏置上的约束项，为Constraints对象

### 输入shape

4D tensor with shape: `(batch, channels, rows, cols)` if `data_format` is `"channels_first"` or 4D tensor with shape: `(batch, rows, cols, channels)` if `data_format` is `"channels_last"`.

### 输出shape

4D tensor with shape: `(batch, filters, new_rows, new_cols)` if `data_format` is `"channels_first"` or 4D tensor with shape: `(batch, new_rows, new_cols, filters)` if `data_format` is `"channels_last"`. `rows`and `cols`4D tensor with shape: `(batch, filters, new_rows, new_cols)` if `data_format` is `"channels_first"` or 4D tensor with shape: `(batch, new_rows, new_cols, filters)` if `data_format` is `"channels_last"`. `rows`and `cols` values might have changed due to padding. If `output_padding` is specified:

```python
new_rows = ((rows - 1) * strides[0] + kernel_size[0]
            - 2 * padding[0] + output_padding[0])
new_cols = ((cols - 1) * strides[1] + kernel_size[1]
            - 2 * padding[1] + output_padding[1])
```



## Conv3D层

```python
keras.layers.Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

三维卷积对三维的输入进行滑动窗卷积，当使用该层作为第一层时，应提供`input_shape`参数。例如`input_shape = (3,10,128,128)`代表对10帧128*128的彩色RGB图像进行卷积。数据的通道位置仍然有`data_format`参数指定。

### 参数

- filters：卷积核的数目（即输出的维度）
- kernel_size：单个整数或由3个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
- strides：单个整数或由3个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
- padding：补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
- activation：激活函数，为预定义的激活函数名（参考[激活函数](https://keras-cn.readthedocs.io/en/latest/other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
- dilation_rate：单个整数或由3个个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
- data_format：字符串，“channels_first”或“channels_last”之一，代表数据的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128x128的数据为例，“channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。
- use_bias:布尔值，是否使用偏置项

### 输入shape

5D tensor with shape: `(batch, channels, conv_dim1, conv_dim2, conv_dim3)` if `data_format` is `"channels_first"` or 5D tensor with shape: `(batch, conv_dim1, conv_dim2, conv_dim3, channels)` if `data_format` is `"channels_last"`.

### 输出shape

5D tensor with shape: `(batch, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if `data_format`is `"channels_first"` or 5D tensor with shape: `(batch, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)` if `data_format` is `"channels_last"`. `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have changed due to padding.

 

## Cropping1D层

```python
keras.layers.Cropping1D(cropping=(1, 1))
```

在时间轴（axis1）上对1D输入（即时间序列）进行裁剪

### 参数

- cropping：长为2的tuple，指定在序列的首尾要裁剪掉多少个元素

### 输入shape

- 形如(batch, axis_to_crop, features)的3D张量

### 输出shape

- 形如(batch, cropped_axis, features)的3D张量

 

## Cropping2D层

```python
keras.layers.Cropping2D(cropping=((0, 0), (0, 0)), data_format=None)
```

对2D输入（图像）进行裁剪，将在空域维度，即宽和高的方向上裁剪

### 参数

- cropping： int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
  - int：长宽进行相等对称裁剪
  - (1, 1)：长宽不同的对称裁剪，`(symmetric_height_crop, symmetric_width_crop)`
  - ((1, 1), (1, 1))：((top_crop, bottom_crop), (left_crop, right_crop))
- data_format

### 输入shape

4D tensor with shape: 

- If `data_format` is `"channels_last"`: `(batch, rows, cols, channels)`
- If `data_format` is `"channels_first"`: `(batch, channels, rows, cols)`

### 输出shape

4D tensor with shape: 

- If `data_format` is `"channels_last"`: `(batch, cropped_rows, cropped_cols, channels)`
-  If `data_format` is `"channels_first"`: `(batch, channels, cropped_rows, cropped_cols)`

### 例子

```python
# Crop the input 2D images or feature maps
model = Sequential()
model.add(Cropping2D(cropping=((2, 2), (4, 4)),
                     input_shape=(28, 28, 3)))
# now model.output_shape == (None, 24, 20, 3)
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Cropping2D(cropping=((2, 2), (2, 2))))
# now model.output_shape == (None, 20, 16, 64)
```

## Cropping3D层

```python
keras.layers.Cropping3D(cropping=((1, 1), (1, 1), (1, 1)), data_format=None)
```

3D数据

### 参数

- cropping：int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
  - If int: 长宽高相等对称裁剪
  - If tuple of 3 ints:  `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`.
  - If tuple of 3 tuples of 2 ints:` ((left_dim1_crop, right_dim1_crop), (left_dim2_crop, right_dim2_crop), (left_dim3_crop, right_dim3_crop))`
- data_format：

### 输入shape

5D tensor with shape: 

- If `data_format` is `"channels_last"`: `(batch, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop,
  depth)`
- If `data_format` is `"channels_first"`: `(batch, depth,
  first_axis_to_crop, second_axis_to_crop, third_axis_to_crop)`

### 输出shape

5D tensor with shape: 

- If `data_format` is `"channels_last"`: `(batch, first_cropped_axis, second_cropped_axis, third_cropped_axis,
  depth)`
- If `data_format` is `"channels_first"`: `(batch, depth,
  first_cropped_axis, second_cropped_axis, third_cropped_axis)`

 

## UpSampling1D层

```python
keras.layers.UpSampling1D(size=2)
```

在时间轴上，将每个时间步重复`length`次

### 参数

- size：上采样因子

### 输入shape

- 形如（batch，steps，features）的3D张量

### 输出shape

- 形如（batch，upsampled_steps，features）的3D张量

 

## UpSampling2D层

```
keras.layers.convolutional.UpSampling2D(size=(2, 2), data_format=None)
```

将数据的行和列分别重复size[0]和size[1]次

### 参数

- size：整数tuple，分别为行和列上采样因子
- data_format：

### 输入shape

`channels_first`模式下，为形如（batch, channels, height, width）的4D张量

`channels_last`模式下，为形如（batch, height, width, channels）的4D张量

### 输出shape

`channels_first`模式下，为形如（batch, channels, upsampled_rows, upsampled_cols）的4D张量

`channels_last`模式下，为形如（batch, upsampled_rows, upsampled_cols, channels）的4D张量

 

## UpSampling3D层

```
keras.layers.convolutional.UpSampling3D(size=(2, 2, 2), data_format=None)
```

将数据的三个维度上分别重复size[0]、size[1]和size[2]次

### 参数

- size：int或者长为3的整数tuple，代表在三个维度上的上采样因子
- data_format：

### 输入shape

`channels_first`模式下，为形如（batch, channels, dim1, dim2, dim3）的5D张量

`channels_last`模式下，为形如（batch, dim1, dim2, dim3, channels ）的5D张量

### 输出shape

`channels_first`模式，为形如（batch, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3）的5D张量

`channels_last`模式，为形如（batch, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels）的5D张量

 

## ZeroPadding1D

```python
keras.layers.ZeroPadding1D(padding=1)
```

对1D输入的首尾端（如时域序列）填充0，以控制卷积以后向量的长度

### Arguments

- padding: int, or tuple of int (length 2), or dictionary.

  - If int:

  How many zeros to add at the beginning and end of the padding dimension (axis 1).

  - If tuple of int (length 2):

  How many zeros to add at the beginning and at the end of the padding dimension (`(left_pad, right_pad)`).

### Input shape

3D tensor with shape `(batch, axis_to_pad, features)`

### Output shape

3D tensor with shape `(batch, padded_axis, features)`



## ZeroPadding2D

```python
keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None)
```

Zero-padding layer for 2D input (e.g. picture).

This layer can add rows and columns of zeros at the top, bottom, left and right side of an image tensor.

### Arguments

- padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.

  - If int: the same symmetric padding is applied to height and width.
  - If tuple of 2 ints: interpreted as two different symmetric padding values for height and width: `(symmetric_height_pad, symmetric_width_pad)`.
  - If tuple of 2 tuples of 2 ints: interpreted as `((top_pad, bottom_pad), (left_pad, right_pad))`

- data_format: 

### Input shape

4D tensor with shape: 

- If `data_format` is `"channels_last"`: `(batch, rows, cols, channels)` 
- If `data_format` is `"channels_first"`: `(batch, channels, rows, cols)`

### Output shape

4D tensor with shape: 

- If `data_format` is `"channels_last"`: `(batch, padded_rows, padded_cols, channels)`

- If `data_format` is `"channels_first"`: `(batch, channels, padded_rows, padded_cols)`


### ZeroPadding3D

```python
keras.layers.ZeroPadding3D(padding=(1, 1, 1), data_format=None)
```

Zero-padding layer for 3D data (spatial or spatio-temporal).

### Arguments

- padding: int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.

  - If int: the same symmetric padding is applied to height and width.
  - If tuple of 3 ints: interpreted as two different symmetric padding values for height and width: `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
  - If tuple of 3 tuples of 2 ints: interpreted as `((left_dim1_pad, right_dim1_pad),       (left_dim2_pad, right_dim2_pad),       (left_dim3_pad, right_dim3_pad))`

- data_format: 

###Input shape

5D tensor with shape:
- If `data_format` is `"channels_last"`: `(batch, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad,       depth)` 
- If `data_format` is `"channels_first"`: `(batch, depth,       first_axis_to_pad, second_axis_to_pad, third_axis_to_pad)`

###Output shape

5D tensor with shape: 
- If `data_format` is `"channels_last"`: `(batch, first_padded_axis, second_padded_axis, third_axis_to_pad, depth)`
- If `data_format` is `"channels_first"`: `(batch, depth, first_padded_axis, second_padded_axis, third_axis_to_pad)`