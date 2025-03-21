#!/bin/bash

# Check if OPENAI_API_KEY is already set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY not found in environment."
    echo "Please enter your OpenAI API Key: "
    read -s OPENAI_API_KEY
    export OPENAI_API_KEY
    echo "API Key set."
else
    echo "Using existing OPENAI_API_KEY from environment."
fi

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3 and try again."
    exit 1
fi

# Check if virtualenv is installed, install if needed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 not found. Please install pip and try again."
    exit 1
fi

if ! python3 -m pip show virtualenv &> /dev/null; then
    echo "Installing virtualenv package..."
    python3 -m pip install virtualenv
fi

# Create virtual environment if not exists
if [ ! -d "monocle_custom_env" ]; then
    echo "Creating virtual environment monocle_custom_env..."
    python3 -m virtualenv monocle_custom_env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source monocle_custom_env/bin/activate

# Check activation worked
if [ "$VIRTUAL_ENV" == "" ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt file not found."
fi

# Run example.py
if [ -f "example_custom.py" ]; then
    echo "Running example_custom.py..."
    python example_custom.py
else
    echo "Error: example_custom.py file not found."
    exit 1
fi

# Deactivate virtual environment
deactivate
echo "Script execution completed."