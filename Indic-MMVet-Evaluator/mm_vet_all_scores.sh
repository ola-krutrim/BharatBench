# Krutrim-AI-Labs Changes / Modifications done as follows:
# Bash script to run for 12 Indic languages using embedding models

#!/bin/bash

# List of default languages
languages=("Hindi" "Telugu" "Marathi" "Gujarati" "Kannada" "Malayalam" "Tamil" "Sanskrit" "Odia" "Assamese" "Bengali", "English")

# Check if a specific language is passed as an argument
if [ -n "$1" ]; then
  # If a language is specified, run for that language only
  echo "Running mm-vet_evaluator_all.py for specified language: $1"
  python mm-vet_evaluator_all.py --lang "$1"
else
  # If no language is specified, run for all languages
  for lang in "${languages[@]}"
  do
    echo "Running mm-vet_evaluator_all.py for language: $lang"
    python mm-vet_evaluator_all.py --lang "$lang"
  done
fi

