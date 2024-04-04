# Dockerfile, Image, Container
FROM python:3.11

ADD main.py .

RUN pip install mediapy huggingface-hub safetensors torch diffusers numba transformers accelerate

CMD ["python","main.py"]