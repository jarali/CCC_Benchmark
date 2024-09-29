#!/bin/bash

# Define environment variables
ENV_NAME="benchmark-environment"
PYTHON_VERSION="3.10.12"

# Step 1: Install pyenv by running curl and install required packages
echo "Installing pyenv and necessary dependencies..."

# Install pyenv
curl https://pyenv.run | bash

# Append pyenv configuration to ~/.bashrc
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# Source ~/.bashrc to apply the changes
source ~/.bashrc

# Step 2: Update and install system dependencies required for building Python
echo "Updating system and installing dependencies..."
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git unzip

# Step 3: Check if the desired Python version is installed, and install if necessary
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo "Python $PYTHON_VERSION is not installed. Installing using pyenv..."
    pyenv install $PYTHON_VERSION
fi

# Step 4: Create and activate the virtual environment
echo "Creating and activating virtual environment: $ENV_NAME"
pyenv virtualenv $PYTHON_VERSION $ENV_NAME

# Step 5: Activate the virtual environment
pyenv activate $ENV_NAME

# Step 6: Check if activation was successful
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "Virtual environment $ENV_NAME activated."
else
    echo "Failed to activate virtual environment automatically, try enabling it by using 'pyenv activate $ENV_NAME'"
    exit 1
fi
