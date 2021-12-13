from django.contrib.auth import authenticate, login
from django.http.response import HttpResponse
from django.shortcuts import render, redirect
from .forms import UserForm
from .forms import FilterForm
import numpy as np
from jamo import h2j, j2hcj
import fasttext
from keras.saving.save import load_model
output = ""


def signup(request):
    """
    계정생성
    """
    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('index')
    else:
        form = UserForm()
    return render(request, 'common/signup.html', {'form': form})


def filter(request):
    if request.method == "POST":
        form = FilterForm(request.POST)
        if form.is_valid():
            sentence = str(form.save('sentence'))
            output = test_result(sentence, fast_model, LSTM_model)
            return render(request, 'common/filter.html', context={'outputBox': output})

    return render(request, 'common/filter.html')


#########################
# 욕설 필터링


def divide(s):    # 초,중,종성으로 분리
    sentence_split = s.split()
    sentence_split_result = []
    for word in sentence_split:
        word = j2hcj(h2j(word))
        sentence_split_result.append(word)
    sentence_result = ' '.join(sentence_split_result)
    return sentence_result


# # FastText모델 학습
fast_model = fasttext.train_supervised(input="/Users/urim/Desktop/Comment-Cleaner/mysite/common/Fasttext_train.txt",
                                       lr=0.05, dim=100, ws=5, epoch=50, minn=1, word_ngrams=6)

sentence_number = 25
# lstm 모델 불러오기
LSTM_model = load_model(
    '/Users/urim/Desktop/Comment-Cleaner/mysite/common/LSTM_model.h5')


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
    # print(fast_vec)
    if test_pre[0][0] == 0:
        output = "비속어가_포함되어_있지_않습니다."
    else:
        output = "비속어가_포함되어_있습니다."
    return output


#########################
# 감정분석 모델
