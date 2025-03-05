FROM python:3.12-slim

SHELL [ "/bin/bash", "-c" ]

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    git \
    git-lfs \
    libgtk-3-dev \
    libgl1-mesa-glx \
    libxkbcommon-x11-0 \
    sudo \
    unzip \
    wget \
    libcairo2-dev \
    libpango1.0-dev \
    ffmpeg \
    texlive \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-xetex \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables to prevent issues
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PYTHONUNBUFFERED=1



# Install required dependencies
RUN apt-get update && apt-get install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Setting up user inside the container
ARG USER_UID=1000
ARG USER_GID=1000
ARG USER_NAME

RUN groupadd --gid ${USER_GID} ${USER_NAME} && \
    useradd --create-home --gid ${USER_GID} --uid ${USER_UID} ${USER_NAME} && \
    usermod -aG sudo ${USER_NAME} && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    mkdir -p /home/${USER_NAME} && \
    chown -R ${USER_UID}:${USER_GID} /home/${USER_NAME} && \
    mkdir /home/${USER_NAME}/workspace && \
    chown -R ${USER_UID}:${USER_GID} /home/${USER_NAME}
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}/workspace

# Copy pyproject.toml
COPY pyproject.toml ./

# # Install dependencies using uv
RUN uv venv && source .venv/bin/activate && \
    uv pip install -r <(uv pip compile pyproject.toml)

ENV SHELL=/bin/bash