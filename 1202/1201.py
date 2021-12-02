import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.utils import np_utils

# 파일 불러오기
df = pd.read_csv('/Users/urim/Desktop/1202/github_dataset.txt',
                 delimiter='|', encoding='UTF-8')
df.columns = ['data', 'label']
# print(df)

x = df.iloc[:, 0:0]  # 문장 데이터
y = df.iloc[:, 1:1]  # 욕설 라벨링 데이터


# test, train으로 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
# print(len(x_train), "train +", len(x_test), "test")

# 글자 수 맞추기
x_train = sequence.pad_sequences(x_train, maxlen=300)
x_test = sequence.pad_sequences(x_test, maxlen=300)

# 모델 설정
model = Sequential()
model.add(Embedding(5826, 300))
model.add(Dropout(0.5))
model.add(LSTM(55))
model.add(Dense(1))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=100,
                    epochs=5, validation_data=(x_test, y_test))

# 테스트 정확도 출력
print("\n 정확도 : %.4f" % (model.evaluate(x_test, y_test)[1]))

# 테스트셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']
