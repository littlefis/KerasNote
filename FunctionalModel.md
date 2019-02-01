# 函数式模型 Functional

## 全连接网络

```python
from keras.layers import Input, Dense
from keras.models import Model
# 返回张量
input = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
model.fit(data, labels)
```

## 多输入多输出

![multi-input-multi-output-graph](https://keras-cn.readthedocs.io/en/latest/images/multi-input-multi-output-graph.png)

```python
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
main_input = Input(shape=(100,), dtype='int32', name='main_input')
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(100)
lstm_out = LSTM(32)(x)
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             loss_weight=[1., 0.2])
model.fit([headline_data, additional_data], [labels, labels], [epochs=50, batch_size=32])

model.compile(optimizer='rmsprop',
             loss={'main_output':'binary_crossentropy','aux_output':'binary_crossentropy'},
              loss_weights={'main_output':1., 'aux_output':0.2})
model.fit({'main_input':headline_data, 'aux_input':additional_data},
         {'main_output':labels, 'aux_output':labels},
         epochs=50, batch_size=32)
```

## 共享层

初始化该层一次，然后多次调用它

```python
shared_lstm = LSTM(64)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)
```



































