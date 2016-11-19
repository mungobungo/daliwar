FROM daliwar/base_tensorflow

MAINTAINER Oleg Taykalo <oleg.taykalo@gmail.com>

# Usage: ADD [source directory or URL] [destination directory]
ADD /model /model

WORKDIR "/model"

#CMD "ls"
CMD python submit_generator.py