FROM python:3.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy  

COPY ["predict.py", "model_C=1.0.bin", "./" ] 

EXPOSE 8080

ENTRYPOINT [ "waitress-serve", "--listen=0.0.0.0:8080", "predict:app" ]