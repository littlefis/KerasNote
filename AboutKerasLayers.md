# 关于Keras的层

所有Keras层都有如下方法：

- `layer.get_weights()`：返回层的权重，np array
- `layer.set_weights(weights)`：从np array中加载权重
- `layer.get_config()`：返回当前层的配置字典，层可以通过配置信息重构

```python
layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.from_config(config)
```

如果层仅有一个计算节点（即该层不是共享层），则可以通过下列方法获得输入张量、输出张量、输入数据的形状和输出数据的形状：

- `layer.input`
- `layer.output`
- `layer.input_shape`
- `layer.output_shape`

如果该层有多个计算节点可以使用下面的方法

- `layer.get_input_at(node_index)`
- `layer.get_output_at(node_index)`
- `layer.get_input_shape_at(node_index)`
- `layer.get_output_shape_at(node_index)`

