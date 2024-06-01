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
pip install torch transformers beautifulsoup4 selenium nltk qiskit openai datasets python3-pip

# Install Selenium
!pip install selenium

# Import Selenium
from selenium import webdriver

# Verify the installation
!pip show selenium

# Simple usage example
driver = webdriver.Chrome()
driver.get('http://www.google.com/')
print(driver.title)
driver.quit()

# Clone the repository
echo "Cloning the repository..."
git clone https://github.com/SilenceGeneric/DarkAI.git

# Navigate to the project directory
cd DarkAI

# Run the main script
echo "Running the main script..."
python main.py
