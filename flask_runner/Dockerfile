FROM daliwar/base_tensorflow


MAINTAINER Oleg Taykalo <oleg.taykalo@gmail.com>

RUN pip --no-cache-dir install \
    flask \
    werkzeug

ADD /model /model

WORKDIR "/model"

ENTRYPOINT "python" server.py



