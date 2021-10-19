FROM openjdk:8

MAINTAINER Fokko Driesprong <fokkodriesprong@godatadriven.com>

RUN update-ca-certificates -f \
  && apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y \
    software-properties-common \
    wget \
    git \
    libatlas3-base \
    libopenblas-base \
    libatlas-base-dev \
    build-essential \
  && apt-get clean

# Spark
RUN cd /usr/ \
  && wget "https://archive.apache.org/dist/spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz" \
  && tar xzf spark-2.4.3-bin-hadoop2.7.tgz \
  && rm spark-2.4.3-bin-hadoop2.7.tgz \
  && mv spark-2.4.3-bin-hadoop2.7 spark

ENV SPARK_HOME /usr/spark
ENV SPARK_MAJOR_VERSION 2
ENV PYTHONPATH=$SPARK_HOME/python/lib/py4j-0.10.7-src.zip:$SPARK_HOME/python/:$PYTHONPATH

RUN mkdir -p /usr/spark/work/ \
  && chmod -R 777 /usr/spark/work/

ENV SPARK_MASTER_PORT 7077

# Miniconda
ENV CONDA_DIR=/opt/miniconda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod a+x miniconda.sh && \
    ./miniconda.sh -b -p $CONDA_DIR && \
    rm ./miniconda.sh && \
    conda install python=3.6


WORKDIR /app/

#RUN pip install --upgrade pip \

RUN conda  install -c conda-forge -y  pyspark==2.4.6 \
    ua-parser==0.10.0  \
    pyarrow==0.14  \
    pytest==6.2.4 \
    pytest-html==3.1.1 \
    Cython>=0.22 \
    numpy>=1.10.0 \
    pyyaml \
    loguru \
    pandas==0.25.3

# pystan>=2.14 \
# RUN pip  install -r requirements.txt

ENV PATH=$PATH:$SPARK_HOME/bin/



CMD /usr/spark/bin/spark-class org.apache.spark.deploy.master.Master
