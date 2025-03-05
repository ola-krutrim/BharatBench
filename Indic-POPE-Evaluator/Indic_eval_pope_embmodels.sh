# Krutrim-AI-Labs Changes / Modifications done as follows:
# Bash script to run for 12 Indic languages using embedding models

#!/bin/bash

# List of Indian languages to test (adjust as needed)
languages=("hindi" "tamil" "telugu" "kannada" "malayalam" "marathi" "gujarati" "sanskrit" "bengali" "assamese" "odia" "english")

# List of available models from models_dict in your Python script
available_models=("muril" "xlm-roberta" "labse" "multilingual-bert")

# Default model if none is specified
default_model="multilingual-bert"

# Function to display available models
function show_available_models {
  echo "Available models:"
  for model in "${available_models[@]}"; do
    echo " - $model"
  done
}

# Parse the model argument
specified_model=$1

# Check if the specified model is in the available models list
if [[ " ${available_models[@]} " =~ " ${specified_model} " ]]; then
  model_to_use="$specified_model"
else
  echo "No valid model specified or invalid model provided. Defaulting to: $default_model"
  model_to_use="$default_model"
fi

# Define the path to your Python script
python_script="./scripts/Indic_eval_pope_embmodels.py"

# Loop over each language for the chosen model
for lang in "${languages[@]}"; do
  echo "Running for language: $lang and model: $model_to_use"
  
  # Call your Python script with the current language and the chosen model
  python3 "$python_script" --language "$lang" --model "$model_to_use"
  
  # Check for errors
  if [ $? -ne 0 ]; then
    echo "Error encountered for language: $lang and model: $model_to_use"
  fi
done
