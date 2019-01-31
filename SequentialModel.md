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







