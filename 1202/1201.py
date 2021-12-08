from os import X_OK
import fasttext
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, embeddings
from keras.utils import np_utils
import re
import matplotlib
import matplotlib.pyplot as plt
from jamo import h2j, j2hcj
from gensim.models import FastText
import json
import fasttext

# 파일 불러오기
df = pd.read_csv('/Users/urim/Desktop/Comment-Cleaner/1202/github_dataset.txt',
                 delimiter='|', encoding='UTF-8')
df.columns = ['data', 'label']
# print(df)


def preprocessing(sentence):  # preprocessing 특수문자, 이모티콘, 한글자 단어 제거
    sentence_split = sentence.split()
    sentence_split_result = []
    for word in sentence_split:
        if word[0] != '@' and word[:4] != 'http':
            sentence_split_result.append(word)
    sentence_result = ' '.join(sentence_split_result)
    text_result = ''.join(re.compile('[가-힣|0-9|a-z| ]+').
                          findall(sentence_result)).strip()

    text_list = []
    for index in range(len(text_result.split())):
        if len(text_result.split()[index]) > 1:
            text_list.append(text_result.split()[index])
    text_result = " ".join(text_list)

    return text_result


df["data"].apply(preprocessing)
# print(df)


def word_split(i):  # 한 문장 내에 2단어 미만 문장제거(띄어쓰기 되어있지 않은 문장제거),
    return len(i.split())


df['word_count'] = df['data'].apply(word_split)  # 단어 개수 세는 칼럼 추가
df = df[df.word_count >= 2]


# plt.hist(x['word_count'])
# plt.show()
df = df[df.word_count <= 120]  # 한 문장내에 단어 120개 초과 문장 제거 (통계보고 결정)
# print(df)


def divide(sentence):  # 초,중,종성으로 분리
    sentence_split = sentence.split()
    sentence_split_result = []
    for word in sentence_split:
        word = j2hcj(h2j(word))
        sentence_split_result.append(word)
    sentence_result = ' '.join(sentence_split_result)
    return sentence_result


df["divide"] = df["data"].apply(divide)
# print(df)


x = df.iloc[:, 3:]  # 문장 데이터
y = df.iloc[:, 1:2]  # 욕설 라벨링 데이터
# print(x)
# print(y)

# test, train으로 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
print(len(x_train), "train +", len(x_test), "test")

#x_train.to_csv('Fasttext_train.txt', index=False)

# FastText모델 학습
fast_model = fasttext.train_supervised(input="Fasttext_train.txt",
                                       lr=0.05, dim=100, ws=5, epoch=50, minn=1, word_ngrams=6)
# print(fast_model['서발'])

# print(x_train)
print('x_train ', len(x_train))
print('y_test ', y_test)
# 단어 임베딩

sentence_number = 25


def embedding(df):
    result_vec = []
    for sen in df:
        # print(sen)
        word_list_vec = []
        sen_split = sen.split()
        for w_index in range(sentence_number):
            if w_index < len(sen_split):
                word_list_vec.append(fast_model[sen_split[w_index]])
            else:
                word_list_vec.append(np.array([0]*100))  # 0으로 벡터 개수 맞춤
        word_list_vec = np.array(word_list_vec)
        result_vec.append(word_list_vec)
    result_vec = np.array(result_vec)
    print('result_vec ', result_vec.shape)
    return result_vec


train_vec = embedding(x_train["divide"])
test_vec = embedding(x_test["divide"])

# # 모델 설정

LSTM_model = Sequential()
LSTM_model.add(LSTM(units=1, input_shape=(25, 100)))
LSTM_model.add(Dense(1, activation='sigmoid'))
LSTM_model.compile(loss='binary_crossentropy',
                   optimizer='RMSprop', metrics=['accuracy'])
LSTM_model.fit(train_vec, y_train, epochs=2,
               validation_data=(test_vec, y_test))


def test_result(s, fasttest_model, lstm_model):
    test_word = divide(s)
    test_word_split = test_word.split()
    fast_vec = []
    for index in range(sentence_number):
        if index < len(test_word_split):
            fast_vec.append(fasttest_model[test_word_split[index]])
        else:
            fast_vec.append(np.array([0]*100))
    fast_vec = np.array(fast_vec)
    # 학습 데이터와 마찬가지로 3차원으로 크기 조절
    fast_vec = fast_vec.reshape(1, fast_vec.shape[0], fast_vec.shape[1])
    test_pre = lstm_model.predict([fast_vec])  # 비속어 판별
    print(fast_vec)
    if test_pre[0][0] == 0:
        print("lstm 결과 : 비속어가 포함되어 있지 않습니다.")
    else:
        print("lstm 결과 : 비속어가 포함되어 있습니다.")
    print(test_pre)


a = '야야 시발 니가 뭔데 존나 어이없네'
test_result(a, fast_model, LSTM_model)


b = '안녕 안뇽 만나서 반가웡~~'
test_result(b, fast_model, LSTM_model)

c = 'ㅇ'
test_result(c, fast_model, LSTM_model)

# test_word_run = divide(s)
# result = fast_model.\ get_nearest_neighbors(test_word_run)
# for _,word_temp in result :
#     print(run_inverse(word_temp))


# for s_split in s.split():
#     test_word_run = divide(s_split)
#     result = fast_model.get_nearest_neighbors(test_word_run)
#     count = 0
#     for _, word_temp in result:
#         for w in word_list:
#             if w in run_inverse(word_temp):
#                 count += 1
