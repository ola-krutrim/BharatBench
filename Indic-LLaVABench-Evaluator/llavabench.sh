# Krutrim-AI-Labs Changes / Modifications done as follows:
# Bash script to run for 12 Indic languages using embedding models

#!/bin/bash

# List of major Indian languages
LANGUAGES=("Hindi" "Telugu" "Tamil" "Kannada" "Malayalam" "Marathi" "Gujarati" "Bengali" "Sanskrit" "Assamese" "Odia" "English")

# Get language input (if provided)
LANGUAGE=$1

# Function to process a single language
process_language() {
    local LANG=$1

    # Set default paths for questions and answers based on the language
    QUESTIONS="./inputs/$LANG/questions.jsonl"
    ANSWERS="./inputs/$LANG/answers_gpt4.jsonl"
    
    # Check if files exist for the given language
    if [[ -f "$QUESTIONS" && -f "$ANSWERS" ]]; then
        echo "Processing $LANG..."

        # Create directories if they do not exist
        mkdir -p "./reviews" "./answers"

        # Run evaluation script
        python ./scripts/eval_gpt_review_bench.py \
            --question ./inputs/$LANG/questions.jsonl \
            --context ./inputs/$LANG/context.jsonl \
            --rule ./scripts/table/rule.json \
            --answer-list \
                ./inputs/$LANG/answers_gpt4.jsonl \
                ./answers/answer_${LANG}.jsonl \
            --output \
                ./reviews/review_${LANG}.jsonl

        # Run summarization script
        python ./scripts/summarize_gpt_review.py -f ./reviews/review_${LANG}.jsonl
    else
        echo "Files for $LANG not found, skipping..."
    fi
}

# If a specific language is provided
if [[ -n "$LANGUAGE" ]]; then
    # Process only the specified language
    process_language "$LANGUAGE" "$CONTEXT" "$OUTPUT_FILE"
else
    # Process all languages
    for LANG in "${LANGUAGES[@]}"; do
        process_language "$LANG" "$CONTEXT" "$OUTPUT_FILE"
    done
fi






