#!/bin/bash

# Define dependencies
dependencies=("python3" "python3-pip" "git")

# Install dependencies
for dependency in "${dependencies[@]}"; do
    if ! command -v "$dependency" &> /dev/null; then
        echo "Installing $dependency..."
        sudo apt-get install -y "$dependency"
    fi
done

# Install Python packages
echo "Installing Python packages..."
pip3 install torch transformers beautifulsoup4 selenium nltk qiskit openai datasets

# Verify the installation of selenium
pip3 show selenium

# Clone the repository
echo "Cloning the repository..."
git clone https://github.com/SilenceGeneric/DarkAI.git

# Navigate to the project directory
cd DarkAI

# Run the main script
echo "Running the main script..."
python3 main.py
