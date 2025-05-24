FROM python:3.12

ARG UID
ARG GID
ENV USER_NAME="Marek"
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

WORKDIR /app

RUN groupadd --gid $GID $USER_NAME && \
    useradd -m -u $UID --gid $GID $USER_NAME && \
    chown -R $USER_NAME /app


COPY requirements.txt .

RUN pip --no-cache-dir install -r requirements.txt

USER $USER_NAME
