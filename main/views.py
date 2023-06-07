from django.shortcuts import render
from .models import File
import helpers
from rest_framework.decorators import api_view
from rest_framework.response import Response


# Create your views here.


@api_view(["GET"])
def chat_bot(request):
    query = request.GET.get("query")
    if query is None:
        return Response("Please provide a query", status=400)

    response = helpers.get_chatbot_response(query)
    return Response(response)


@api_view(["POST"])
def get_ocr(request):
    file = request.FILES["files"]
    print("test")
    if file is None:
        return Response("Please provide a file", status=400)
    file = File.objects.create(file=file)

    response = helpers.handle_ocr(file.file.path)

    return Response(response)
