FROM python:3.11 #image de base 

WORKDIR /app 

COPY . . 

RUN pip install -r requirements.txt

CMD ["python" , "main.py"]