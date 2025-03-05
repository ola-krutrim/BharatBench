# Krutrim-AI-Labs Changes / Modifications done as follows:
# Bash script to run for 12 Indic languages using embedding models

#!/bin/bash

# Define the list of major Indian languages
languages=("hindi" "telugu" "tamil" "kannada" "malayalam" "marathi" "gujarati" "bengali" "sanskrit" "assamese" "odia")

# Check if a language is specified as an argument
if [ $# -eq 0 ]; then
    # No language specified, run the script for all languages
    echo "No language specified. Running for all major Indian languages..."
    for lang in "${languages[@]}"; do
        echo "Running for language: $lang"
        python ./scripts/Indic_eval_pope.py --language "$lang"
    done
else
    # Language specified, run the script for the specified language
    echo "Running for specified language: $1"
    python ./scripts/Indic_eval_pope.py --language "$1"
fi
