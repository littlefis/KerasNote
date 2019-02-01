# Sequential模型接口

## Sequential模型方法

### compile

```python
compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
```

- `optimizer`：string类型的优化器名或实例。
- `loss`：string类型的目标函数的名称或目标函数。如果有多个输出，可以通过传递字典或列表在每个输出上使用不同的损失，模型的优化值将是所有损失的总和。
- `metrics`：模型在训练和测试期间需要评估的度量列表，通常使用`metrics=['accuracy']`。为多输出模型的不同输出指定不同的度量标准可以传递字典，如`metrics={'output_a': 'accuracy'}`。
- `loss_weight`：指定标量系数的可选列表或字典（系数类型为python浮点数），用以加券不同模型输出的损失贡献，模型将最小化单个损失的加权和。
- `sample_weight_mode`：按时间步为样本赋权（2D权值矩阵），将该值设置为`temporal`。参照下面的fit函数。
- `weighted_metrics`：
- `target_tensors`：
- `**kwargs`：使用tf后端时，参数会被传递给`tf.Session.run`。

### fit

```python
fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
```

- `x`：np数组（单输入模型），np数组列表（多输入）。
- `y`：目标标签，数组或数组列表。
- `batch_size`：整数或`None`。
- `epochs`：迭代次数，结合`initial_epoch`考虑，`epochs`并不是训练了多少轮，而是训练终止时的总迭代次数。
- `verbose`：进度详情，整数，可以是0、1、2，0为无显示，1为进度条，2为每个epoch一行。
- `callbacks`：
- `validation_split`：0到1之间的浮点数，指定验证集比例。
- `validation_data`：(x, y)形式的元组，是指定的验证集，该参数会覆盖`validation_split`。
- `shuffle`：布尔或字符串，表示是否在训练过程中打乱样本顺序；若为字符串'batch'则是处理HDF5数据的特殊情况，此时将在batch内部打乱数据、
- `class_weight`：字典，用于加权损失函数，只能用于训练//?
- `sample_weight`：
- `initial_epoch`：整数，开始训练的epoch数，可用于恢复之前的训练
- `steps_per_epoch`：整数或`None`，结束一个epoch开始下一个epoch前的总步数（样本batch数）
- `validation_steps`：仅在上个参数确定时可用，停止前要验证的总步数

返回`History`对象，其`History.history`属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况

### evaluate

```python
evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None)
```

- `x`：
- `y`：
- `batch_size`：
- `verbose`：只能取0或1
- `sample_weight`：
- `steps`：

返回测试误差的标量值或标量list，`model.metrics_names`可给出list各个值的含义。

### predict

```python
predict(x, batch_size=None, verbose=0, steps=None)
```

按batch获取输出，返回预测值的np array

### train_on_batch

```python
train_on_batch(x, y, sample_weight=None, class_weight=None)
```

在一个batch的数据上进行一次参数更新，返回训练误差的标量值或标量值的list

### test_on_batch

```python
test_on_batch(x, y, sample_weight=None)
```

在一个batch的数据上对模型进行评估

### predict_on_batch

```python
predict_on_batch(x)
```

在一个batch的数据上对模型进行测试，返回模型在一个batch上的预测结果

### fit_generator

```python
fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
```



### evaluate_generator

```python
evaluate_generator(generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
```



### predict_generator

```python
predict_generator(generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
```



### get_layer

```python
get_layer(name=None, index=None)
```

