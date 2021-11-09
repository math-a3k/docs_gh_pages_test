FROM centos:centos7.2.1511

######################## Update system packages first
RUN yum -y update
RUN yum install -y epel-release

######################## GCC update 7 version - for fasttext
RUN yum install -y centos-release-scl scl-utils-build 
RUN yum install -y devtoolset-7

SHELL [ "/usr/bin/scl", "enable", "devtoolset-7"]

######################## Add bazel repo
RUN echo -e "\
[copr:copr.fedorainfracloud.org:vbatts:bazel] \n\
name=Copr repo for bazel owned by vbatts \n\
baseurl=https://download.copr.fedorainfracloud.org/results/vbatts/bazel/epel-7-\$basearch/ \n\
type=rpm-md \n\
skip_if_unavailable=True \n\
gpgcheck=1 \n\
gpgkey=https://download.copr.fedorainfracloud.org/results/vbatts/bazel/pubkey.gpg \n\
repo_gpgcheck=0 \n\
enabled=1 \n\
enabled_metadata=1\
" > /etc/yum.repos.d/bazel.repo

######################## Utils packages
RUN yum install -y \
       java-1.8.0-openjdk java-1.8.0-openjdk-devel \
       maven curl unzip git vim nano wget gcc gcc-c++ make cmake openblas-devel \
       openssl-devel bzip2-devel libffi-devel zlib-devel xz-devel bazel3 fasttext \
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


######################## Miniconda
ENV CONDA_DIR=/opt/minicond
ENV PATH=$PATH:$SPARK_HOME/bin:$CONDA_DIR/bin

RUN cd /tmp \
    && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && chmod +x miniconda.sh \
    && ./miniconda.sh -b -p $CONDA_DIR \
    && rm -rf ./miniconda.sh \
    && conda install python=3.7


######################## PIP packages install
WORKDIR /app/
COPY requirements.txt /app
RUN cd /app \
    && pip3 install --no-cache-dir  wheel pypandoc \
    && pip3 install --no-cache-dir  -r requirements.txt


CMD $SPARK_HOME/bin/spark-class org.apache.spark.deploy.master.Master
