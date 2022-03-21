# Using GPU accelerated tensorflow.
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Make a directory for the code.
WORKDIR /code

# Change user to the Docker host user.
ARG USER_NAME=sahel
ARG USER_ID=1002
ARG GROUP_ID=1002

RUN groupadd -g $GROUP_ID $USER_NAME \
  && useradd -m -l -r -u $USER_ID -g $USER_NAME $USER_NAME
USER $USER_NAME

# Install dependencies.
COPY . .
RUN pip install -r requirements.txt
RUN pip install -e .

CMD ["bash"]
