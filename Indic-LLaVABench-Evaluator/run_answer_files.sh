# Krutrim-AI-Labs Changes / Modifications done as follows:
# Bash script to run for 12 Indic languages using embedding models

#!/bin/bash

# Define an array of supported languages
SUPPORTED_LANGUAGES=("Hindi" "Telugu" "Marathi" "Gujarati" "Kannada" "Malayalam" "Tamil" "Bengali" "Odia" "Assamese" "Sanskrit" "English")

# Check if the language argument is provided
if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [<language>]"
    exit 1
fi

# If a language is provided, use it; otherwise, loop through all supported languages
if [ "$#" -eq 1 ]; then
    LANGUAGES=("$1")
else
    LANGUAGES=("${SUPPORTED_LANGUAGES[@]}")
fi

# Loop through each language and run the Python script
for LANGUAGE in "${LANGUAGES[@]}"; do
    # Define input and output file paths based on the language
    RESPONSE_FILE="./model_responses/responses_${LANGUAGE}.jsonl"
    ANSWERS_FILE="./answers/answer_${LANGUAGE}.jsonl"

    # Check if the response file exists
    if [[ ! -f "$RESPONSE_FILE" ]]; then
        echo "Response file not found for language: $LANGUAGE"
        continue
    fi

    # Run the Python script
    echo "Running evaluation for language: $LANGUAGE"
    python ./scripts/answer_file.py --response-file "$RESPONSE_FILE" --answers-file "$ANSWERS_FILE"
done
