#!/bin/bash

# Define dependencies
dependencies=("python3" "pip" "git")

# Install dependencies
for dependency in "${dependencies[@]}"; do
    if ! command -v "$dependency" &> /dev/null; then
        echo "Installing $dependency..."
        sudo apt-get install -y "$dependency"
    fi
done

# Install Python packages
echo "Installing Python packages..."
pip install torch transformers beautifulsoup4 selenium nltk qiskit openai datasets

# Clone the repository
echo "Cloning the repository..."
git clone https://github.com/yourusername/darkweb-analysis.git

# Navigate to the project directory
cd DarkAI

# Run the main script
echo "Running the main script..."
python main.py