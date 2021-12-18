from django.core.paginator import Paginator
from django.shortcuts import render, get_object_or_404



def index(request):
    return render(request, 'mysite/templates/index.html')


