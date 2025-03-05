# MM-Vet Evaluation

## To compute model responses on indian version of MMVet benchmarks

Run your model on given Questions (available in the directory `./mm-vet_qa/`) and Images that are available at MMVet: https://huggingface.co/datasets/whyu/mm-vet

## Overview

The **MM-Vet Evaluation** repository provides a comprehensive framework for evaluating multimodal models across multiple Indian languages. The evaluation script compares model responses with ground truth answers and computes various evaluation scores.

### Supported Languages:
- Hindi
- Telugu
- Marathi
- Gujarati
- Kannada
- Malayalam
- Tamil
- Sanskrit
- Odia
- Assamese
- Bengali
- English

## Repository Structure

### 1. Ground Truth Files

Ground truth question and answer files for each language should be placed in the following directory: ```./mm-vet_qa/```


- **Naming Convention**:  
  The file names must follow this format:  
  `mm-vet_{lang}.json`  
  Where `{lang}` is the language name (e.g., `mm-vet_Hindi.json`).

### 2. Model Response Files

The responses of the model being evaluated should be stored in the following directory: `./model_responses/`


- **Naming Convention**:  
  The file names must follow this format:  
  `mm-vet_{lang}.json`  
  Where `{lang}` is the language name (e.g., `mm-vet_Hindi.json`).

### 3. Evaluation Scores

After running the evaluation, the results (scores) for each language will be stored in: `./grades/grades.txt`


This file will contain the evaluation metrics for each language, providing an overall assessment of the model’s performance.

## How to Run

### 1. Run for a Specific Language

To evaluate the model for a specific language, run the script and pass the desired language as an argument:
```bash
./mm_vet_all_scores.sh Hindi
```
### 2. Run for all Languages
```bash
./mm_vet_all_scores.sh 
```

## Acknowledgement

Indic-MMVet-Evaluator is built with reference to the code of the following projects: [MM-Vet](https://github.com/yuweihao/MM-Vet) and Dataset from: https://huggingface.co/datasets/whyu/mm-vet. Thanks for their awesome work!