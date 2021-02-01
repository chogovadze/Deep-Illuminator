FROM python:3.7

WORKDIR /app

COPY ./app .

RUN pip install -e .
RUN pip install -r probe_relighting/requirements.txt

WORKDIR /app/probe_relighting

ENTRYPOINT ["python", "generate_images.py"]
