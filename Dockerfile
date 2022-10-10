FROM nvcr.io/nvidia/pytorch:22.09-py3  

SHELL ["/bin/bash", "-c"]

RUN apt update
RUN apt install micro bat git tmux fish curl -y

RUN conda init bash
RUN conda activate

RUN conda install mamba -y
RUN mamba install wandb -y 
RUN mamba install starship -y
RUN yes | pip install transformers -U

RUN mkdir -p /root/.config/fish/
RUN touch /root/.config/fish/config.fish
RUN echo "starship init fish | source" >> ~/.config/fish/config.fish
RUN echo 'eval "$(starship init bash)"' >> ~/.bashrc

RUN git clone https://github.com/AntreasAntoniou/wandb_stateless_utils.git

RUN cd wandb_stateless_utils; git pull

WORKDIR /wandb_stateless_utils
