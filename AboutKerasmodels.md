# 关于Keras模型

## Keras的两种主要模型：顺序模型和函数模型。

两种模型的共同方法和属性：

- `model.layers`是包含模型的各层的展平列表

- `model.inputs`是模型的输入张量列表

- `model.outputs`是模型的输出张量列表

- `model.summary()`可以打印模型的概要，实际调用的`utils.print_summary`

- `model.get_config`返回包含模型配置信息的字典，模型可以从config信息重构

```python
config = model.get_config()
model = Model.from_config(config)
# for sequential model
model = Sequential.from_config(config)
```

- `model.get_layer()`依据层名或下表获得层对象

- `model.get_weights()`获取权重列表，np array类型
- `model.set_weights()`载入权重，np array类型
- `model.to_json`返回包含网络结构的json，不含权重，可以从json中重构模型

```python
from models import model_from_json
json_string = model.to_json()
model = model_from_json(json_string)
```

- `model.to_yaml`与上述类似，将json替换为yaml即可
- `model.save_weights(filepath)` 保存权重，文件类型HDF5，拓展名.h5
- `model.load_weights(filepath, by_name=False)`加载权重，默认模型结构不变。若载入到某些层相同的不同模型中，可以设置`by_name=True`，此时只有名字匹配的层才会载入权重。

## 模型子类化

可以通过继承`Model`类，并在`call`方法中实现自己的正向传播来创建自己的完全可自定义的模型。

```python
import keras

class SimpleMLP(keras.Model):

    def __init__(self, use_bn=False, use_dp=False, num_classes=10):
        super(SimpleMLP, self).__init__(name='mlp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        self.num_classes = num_classes

        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(num_classes, activation='softmax')
        if self.use_dp:
            self.dp = keras.layers.Dropout(0.5)
        if self.use_bn:
            self.bn = keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs):
        x = self.dense1(inputs)
        if self.use_dp:
            x = self.dp(x)
        if self.use_bn:
            x = self.bn(x)
        return self.dense2(x)

model = SimpleMLP()
model.compile(...)
model.fit(...)
```

层定义在`\__init__(self, ...)`，指定正向传播`call(self, inputs)`。