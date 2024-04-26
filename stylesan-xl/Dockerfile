FROM nvcr.io/nvidia/pytorch:23.06-py3
COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc libsndfile1
