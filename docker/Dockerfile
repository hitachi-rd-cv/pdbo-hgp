FROM nvidia/cuda:10.2-runtime-ubuntu18.04

RUN apt-get update
RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd

# Locale setting
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8

# Time setting
RUN apt-get install -y tzdata
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# Package Instalation
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y libsm6 libxext6 libfontconfig1 libxrender1 libglib2.0-0
RUN apt-get install -y tmux nano iputils-ping
RUN apt-get install -y curl
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super

# git
RUN apt-get update
RUN apt-get install -y git

# Expose port for openssh
EXPOSE 22

# Add user
RUN apt-get install -y sudo
RUN useradd -m user
RUN echo 'user:user' | chpasswd
RUN usermod -aG sudo user
RUN chsh -s /bin/bash user
RUN apt-get install -y net-tools

# Pip installation
RUN pip3 install --upgrade pip
RUN pip3 install numpy==1.19.5
RUN pip3 install opencv-python==4.5.5.64
RUN pip3 install Pillow==8.4.0
RUN pip3 install scikit-learn==0.24.2
RUN pip3 install pandas==1.1.5
RUN pip3 install jupyter==1.0.0
RUN pip3 install matplotlib==3.3.4
RUN pip3 install scipy==1.5.4
RUN pip3 install sacred==0.8.2
RUN pip3 install pymongo==4.1.1
RUN pip3 install seaborn==0.11.2
RUN pip3 install PTable==0.9.2
RUN pip3 install keras==2.9.0
RUN pip3 install luigi==3.0.3
RUN pip3 install optuna==2.10.0
RUN pip3 install imgaug==0.4.0
RUN pip3 install ipdb==0.13.9
RUN pip3 install toml==0.10.2
RUN pip3 install sympy==1.9
RUN pip3 install gokart==0.3.24
RUN pip3 install python-dotenv==0.20.0
RUN pip3 install ray==1.12.1
RUN pip3 install bayesian-optimization==1.2.0
RUN pip3 install line-profiler-pycharm==1.1.0
RUN pip3 install torch==1.10.1+cu102 torchvision==0.11.2+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
RUN pip3 install cvxpy==1.1.20
RUN pip3 install tensorboard==2.9.0
RUN pip3 install incense==0.0.11
RUN pip3 install paramiko==2.11.0
RUN pip3 install dill==0.3.4
RUN pip3 install latex

# alias
RUN ln -s /usr/bin/python3 /usr/bin/python
