from django.contrib import admin
from .models import Question
#from .models import Answer


class QuestionAdmin(admin.ModelAdmin):
    search_fields = ['subject']


admin.site.register(Question, QuestionAdmin)
#admin.site.register(Answer)     # Answer 모델 등록
