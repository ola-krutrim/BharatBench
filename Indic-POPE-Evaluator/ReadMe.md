
# GPT-4 and Embedding Model for POPE Evaluation

## Overview
GPT-4 as tool is used for evaluating the POPE (Position-based Object Prediction Evaluation) across multiple Indian languages. This guide provides instructions on running the evaluations either for a single language or for all supported languages.

Question files (for all Indian languages): `./question_files/`

## GPT-4 for POPE Evaluation

GPT-4 can be used for evaluating POPE. Below are the instructions for running GPT-4 based POPE evaluations. Please note that you will have to enter your GPT key in the `./scripts/Indic_eval_pope.py` file before running the evaluation.

### Usage

#### Evaluating for a Single Language
To run the evaluation for a specific language (e.g., Telugu), use the following command:
```bash
bash run_pope_gpt4.sh telugu
```

#### Evaluating for All Languages
To evaluate for all supported languages, use the command without specifying a language:
```bash
bash run_pope_gpt4.sh
```

### Input Requirements
- The model response file should follow the below format:
  - Place the file in the `model_responses` folder.
  - Name the file as `{language}_results.json`, for example, `telugu_results.json`.

### Results
Final scores are available in `./scores/results_language.txt `for GPT-4 based POPE evaluation 

# POPE Evaluator based on embedding model

## Available Models

The following models are available for evaluation:

- `muril`
- `xlm-roberta`
- `labse`
- `multilingual-bert` (default)



## Running the Script

### 1. Using the Default Model (`multilingual-bert`)

To run the script using the default model (`multilingual-bert`), simply execute the shell script without passing any model as an argument:

```bash
./Indic_eval_pope_embmodels.sh
```

This will automatically evaluate the `multilingual-bert` model across all predefined Indian languages.

### 2. Specifying a Model

To evaluate using a different model, you can pass the model's name as an argument when running the script.

#### Example: Running with `muril`

```bash
./Indic_eval_pope_embmodels.sh muril
```

#### Example: Running with `xlm-roberta`

```bash
./Indic_eval_pope_embmodels.sh xlm-roberta
```

#### Example: Running with `labse`

```bash
./Indic_eval_pope_embmodels.sh labse
```


### Results
Final scores are available in `./scores/{embedding_model_name}_results_language.txt` for Embedding model-based POPE evaluation 


## Acknowledgement

Indic-POPE-Evaluator is built with reference to the code of the following projects: [LLaVA-1.5](https://github.com/haotian-liu/LLaVA/tree/main/llava/eval) and Dataset from : https://huggingface.co/datasets/lmms-lab/POPE . Thanks for their awesome work!
