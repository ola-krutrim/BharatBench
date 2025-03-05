# Krutrim-AI-Labs Changes / Modifications done as follows:
# Bash script to run for 12 Indic languages using embedding models

#!/bin/bash

# Check if a language argument is provided
if [ $# -eq 1 ]; then
    LANGUAGE="$1"
    echo "Running scripts for specified language: $LANGUAGE"

    # Call the run_answer_files.sh script with the specified language
    ./run_answer_files.sh "$LANGUAGE"

    # Call the llavabench.sh script with the specified language
    ./llavabench.sh "$LANGUAGE"
else
    echo "Running scripts for all languages"

    # Call the run_answer_files.sh script for all languages
    ./run_answer_files.sh

    # Call the llavabench.sh script for all languages
    ./llavabench.sh
fi

echo "All scripts have been executed."
