FROM nvcr.io/nvidia/pytorch:22.09-py3

SHELL ["/bin/bash", "-c"]

RUN apt update
RUN apt install micro bat git tmux fish curl -y

RUN conda init bash
ENV TERM xterm-256color
#RUN conda activate

#SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
#
#RUN conda install wandb -y
#RUN conda install starship -y
#RUN yes | pip install transformers -U
#
#RUN mkdir -p /root/.config/fish/
#RUN touch /root/.config/fish/config.fish
#RUN echo "starship init fish | source" >> ~/.config/fish/config.fish
#RUN echo 'eval "$(starship init bash)"' >> ~/.bashrc
#
#RUN git clone https://github.com/AntreasAntoniou/wandb_stateless_utils.git
#
#RUN cd src; git pull
#
#WORKDIR /src
