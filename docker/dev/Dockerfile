FROM gcr.io/deeplearning-platform-release/pytorch-cpu

SHELL ["/bin/bash", "-c"]

RUN apt update
RUN apt install git tmux fish curl -y

RUN conda init bash
ENV TERM xterm-256color

RUN conda install -c conda-forge mamba -y
RUN mamba install -c conda-forge wandb -y
RUN mamba install starship -y
RUN yes | pip install transformers -U

RUN mkdir -p /root/.config/fish/
RUN touch /root/.config/fish/config.fish
RUN echo "starship init fish | source" >> ~/.config/fish/config.fish
RUN echo 'eval "$(starship init bash)"' >> ~/.bashrc

RUN git clone https://github.com/AntreasAntoniou/wandb_stateless_utils.git

RUN cd src; git pull

WORKDIR /wandb_stateless_utils
