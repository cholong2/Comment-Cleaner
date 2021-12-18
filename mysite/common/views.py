from django.contrib.auth import authenticate, login
from django.http.response import HttpResponse
from django.shortcuts import render, redirect

from .forms import UserForm
from .forms import FilterForm
import numpy as np
from jamo import h2j, j2hcj
import fasttext
import tensorflow as tf
from tensorflow import keras
# from keras.saving.save import load_model
import os, sys
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
    return render(request, 'common/signup.html', {'form': form})


def filter(request):
    if request.method == "POST":
        form = FilterForm(request.POST)
        if form.is_valid():
            sentence = request.POST.get("sentence")
            output = test_result(sentence, fast_model, LSTM_model)
            model = load_model()
            result = predict(model, sentence)
            return render(request, 'common/filter.html', context={'outputBox': result})

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

    module_dir = os.path.dirname(__file__)
    fasttext_data = str(os.path.join(module_dir, 'Fasttext_train.txt'))

# # FastText모델 학습
fast_model = fasttext.train_supervised('C:/Users/pcl10/PycharmProjects/pythonProject10/mysite/common/Fasttext_train.txt',
                                       lr=0.05, dim=100, ws=5, epoch=50, minn=1, word_ngrams=6)

sentence_number = 25
# lstm 모델 불러오기
LSTM_model = tf.keras.models.load_model(
    'C:/Users/pcl10/PycharmProjects/pythonProject10/mysite/common/LSTM_model.h5')


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
# Create your views here.
import os, sys
import re
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse


import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
# from tqdm import tqdm, tqdm_notebook
# from tqdm.notebook import tqdm


from .KoBERT.kobert.utils import get_tokenizer
from .KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model

import sys


RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"

# 실시간 입력 형식을 위한 데이터변환기
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=7,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

def load_model():
    device = torch.device('cpu')
    bertmodel, vocab = get_pytorch_kobert_model()

    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    model.eval()

    module_dir = os.path.dirname(__file__)
    file = os.path.join(module_dir, 'kobert7_ending_finale.pt')

    model.load_state_dict(torch.load(file, map_location=device))
    return model

def switch_feel(feel_num):
    if feel_num == 0:
        return "공포"
    elif feel_num == 1:
        return "놀람"
    elif feel_num == 2:
        return "분노"
    elif feel_num == 3:
        return "슬픔"
    elif feel_num == 4:
        return "중립"
    elif feel_num == 5:
        return "행복"
    elif feel_num == 6:
        return "혐오"



# 실시간 입력형 테스트방식
def predict(model, predict_sentence):
    # 파라미터 설정
    max_len = 128
    batch_size = 64
    warmup_ratio = 0.1
    num_epochs = 5
    max_grad_norm = 1
    log_interval = 200
    learning_rate = 5e-5

    device = torch.device('cpu')
    bertmodel, vocab = get_pytorch_kobert_model()

    # 토큰화
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)

    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()
            best = np.argmax(logits)
            test_eval.append(best)
            # 가장 높은것의 값을 -10으로 바꾸어 그 다음으로 높은값을 argmax로 쉽게 찾을 수 있게 함.
            logits[best] = -10
            test_eval.append(np.argmax(logits))

        # 두번째로 높은 값이 0보다 작을경우에는 입력값의 감정과 연관이 없다고 판단되어 결과출력에서 제외함.
        if logits[test_eval[1]] < 0:
            sys.stdout.write(GREEN)
            result = "입력내용에서 가장 크게 느껴지는 감정은 " + switch_feel(test_eval[0]) + " 입니다."
            return result
        else:
            sys.stdout.write(GREEN)
            result = "입력내용에서 가장 크게 느껴지는 감정은 " + switch_feel(test_eval[0]) + " 이고, 두번째는 " + switch_feel(
                test_eval[1]) + " 입니다."
            return result

# def emotion(request):
#    if request.method == 'POST':
#         form = FilterForm(request.POST)
#         if form.is_valid():
#
#             text = str(form.save('sentence'))
#             model = load_model()
#             result = predict(model, text)
#             return render(request, 'common/filter.html', context={'outputBox':result})
#
#    return render(request, 'common/filter.html')
