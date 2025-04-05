# docker buildx create --use
# COMPOSE_PARALLEL_LIMIT=1 docker buildx build --platform linux/amd64,linux/arm64/v8 --push -t alexgkim/tfpv:dev . 

# due to local memory limitations try:
# docker buildx build --platform linux/arm64/v8 --push -t alexgkim/tfpv:arm64v8 . 
# docker buildx build --platform linux/amd64 --push -t alexgkim/tfpv:amd64 . 
# docker manifest create alexgkim/tfpv:dev alexgkim/tfpv:arm64v8 alexgkim/tfpv:amd64
# docker manifest push alexgkim:tfpv:dev


# docker run -v data:/data -v output:/output alexgkim/tfpv:dev  python /opt/TFVP/fitstojson.py

# buildx works differently from build
# docker buildx build --platform linux/arm64/v8  --load -t tfpv:dev . 

# docker run -v /Users/akim/Projects/TFPV/data:/data -v /Users/akim/Projects/TFPV/output:/output -v /Users/akim/Projects/DESI_SGA:/DESI_SGA alexgkim/tfpv:dev ./command.sh

# docker run -v /Users/akim/Projects/TFPV/data:/data -v /Users/akim/Projects/TFPV/output:/output -v /Users/akim/Projects/DESI_SGA:/DESI_SGA tfpv:dev ./command.sh


FROM docker.io/library/python:latest

SHELL ["/bin/bash", "-c"]

WORKDIR /opt

RUN apt-get update && apt-get install -y pip curl unzip tar vim g++ make && apt-get clean

RUN \
    pip3 install            \
        --no-cache-dir      \
        numpy               \
        scipy               \
        fitsio              \
        astropy             \
        matplotlib          \
        pandas


RUN curl -L https://github.com/AlexGKim/TFPV/archive/refs/heads/docker.tar.gz \
    -o TFPV.tar.gz \ 
    && tar xzf TFPV.tar.gz \
    && rm -rf TFPV.tar.gz

RUN git clone https://github.com/stan-dev/cmdstan.git --recursive       \
    && cd cmdstan                                                       \
    && make /opt/TFPV-docker/cluster                                    \
    && make clean

COPY command.sh /opt

RUN chmod +x /opt/command.sh

ENV DATA_DIR=/data
ENV OUTPUT_DIR=/output
ENV DESI_SGA_DIR=/DESI_SGA
ENV RELEASE=Y1

VOLUME /data
VOLUME /output
VOLUME /DESI_SGA
