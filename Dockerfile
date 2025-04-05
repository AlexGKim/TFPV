# docker buildx create --use
# docker buildx build --platform linux/amd64,linux/arm64/v8 --push -t alexgkim/tfpv:dev . 


# buildx works differently from build
# docker buildx build --platform linux/arm64/v8  --load -t tfpv:dev . 

# docker run -v $DATA_DIR:/data -v $OUTPUT_DIR:/output -v $DESI_SGA_DIR:/DESI_SGA alexgkim/tfpv:dev ./command.sh


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


RUN echo "Downloading and installing frequently updated package - $(date +%s)" \
    && curl -L https://github.com/AlexGKim/TFPV/archive/refs/heads/docker.tar.gz \
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
