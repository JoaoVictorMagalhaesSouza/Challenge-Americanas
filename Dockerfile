FROM python:3.7.11

WORKDIR /usr/local/bin

RUN apt-get update && apt-get install -y \
    sudo \
    pandoc \
    pandoc-citeproc \
    libcurl4-gnutls-dev \
    libcairo2-dev \
    libxt-dev \
    libssl-dev \
    libssh2-1-dev 

RUN apt-get update && \
    apt-get upgrade -y

COPY . .

RUN python -m pip install --upgrade pip

EXPOSE 8050

RUN pip install -r requirements.txt

CMD [ "python", "-u", "main.py" ]