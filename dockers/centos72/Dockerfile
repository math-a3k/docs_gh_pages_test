FROM centos:centos7.2.1511

############################ Untils packages
RUN yum -y update
RUN yum -y remove java
RUN yum install -y \
       java-1.8.0-openjdk \
       java-1.8.0-openjdk-devel \
       maven curl unzip git vim nano wget gcc gcc-c++ make\
       openssl-devel bzip2-devel libffi-devel zlib-devel xz-devel \
       && yum clean all \
       && rm -rf /var/cache/yum

############################ Spark
RUN cd /usr/ \
  && wget https://archive.apache.org/dist/spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz \
  && tar -xzvf spark-2.4.3-bin-hadoop2.7.tgz \
  && rm spark-2.4.3-bin-hadoop2.7.tgz \
  && mv spark-2.4.3-bin-hadoop2.7 spark

ENV SPARK_HOME /usr/spark
ENV SPARK_MAJOR_VERSION 2
ENV SPARK_MASTER_PORT 7077
ENV PYTHONPATH=$SPARK_HOME/python/lib/py4j-0.10.7-src.zip:$SPARK_HOME/python/:$PYTHONPATH

RUN mkdir -p /usr/spark/work/ \
  && chmod -R 777 /usr/spark/work/

######################## Python 3.7
RUN cd /usr/src \
     && wget https://www.python.org/ftp/python/3.7.11/Python-3.7.11.tgz  \
     && tar xzf Python-3.7.11.tgz \
     && rm -rf Python-3.7.11.tgz \
     && cd Python-3.7.11 \
     && ./configure --enable-optimizations \
     && make altinstall \
     && cd ../ && rm -rf Python-3.7.11

RUN ln -s /usr/local/bin/python3.7  /usr/bin/python3 \
    && ln -s /usr/local/bin/pip3.7 /usr/bin/pip3

WORKDIR /app/
COPY requirements.txt /app
RUN cd /app \
    && pip3 install --no-cache-dir  wheel pypandoc \
    && pip3 install --no-cache-dir  -r requirements.txt

######################## Miniconda
ENV CONDA_DIR=/opt/minicond
RUN cd /tmp \
    && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && chmod +x miniconda.sh \
    && ./miniconda.sh -b -p $CONDA_DIR \
    && rm -rf ./miniconda.sh


ENV PATH=$PATH:$SPARK_HOME/bin:$CONDA_DIR/bin

CMD $SPARK_HOME/bin/spark-class org.apache.spark.deploy.master.Master
