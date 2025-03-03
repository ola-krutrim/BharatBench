#    Licensed under the MIT License

# Krutrim-AI-Labs Changes / Modifications done as follows:
# Pre-process Answers and Labels of Generated Responses by Using Different Embedding Models for Evaluations

import os
import json
import sys
import argparse
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import util

models_dict = {
    # MURIL (Multilingual Representations for Indian Languages): A model designed by Google for multilingual NLP, 
    # focusing on Indian languages. MuRIL is a multilingual transformer model developed by Google, fine-tuned for
    #  Indian languages and English. It handles transliteration and code-mixed Indian languages.
    "muril": "google/muril-base-cased",
    
    # XLM-RoBERTa: A multilingual version of the RoBERTa model trained on 100 languages.XLM-Roberta is a powerful transformer model that supports 100+ languages, including Indian languages. 
    # It provides high-quality multilingual embeddings.
    "xlm-roberta": "xlm-roberta-base",
    
    # LaBSE (Language-agnostic BERT Sentence Embedding): A sentence transformer model for generating multilingual 
    # sentence embeddings.
    #LaBSE is trained to produce language-agnostic sentence embeddings for 109 languages, including major Indian languages.
    "labse": "sentence-transformers/LaBSE",
    
    # Multilingual BERT: A version of BERT that supports multiple languages (104 languages).A multilingual BERT model supporting
    #  Indian languages along with many others. Works well for embedding generation for mixed-language contexts.
    "multilingual-bert": "bert-base-multilingual-cased"
}

def load_model_and_tokenizer(model_name: str):
    """
    Loads the tokenizer and model based on the provided model name.

    Args:
        model_name (str): The user-friendly model name to load (from models_dict).

    Returns:
        tokenizer: Tokenizer for the specified model.
        model: Pretrained Hugging Face model.
    """
    # Get the corresponding Hugging Face model identifier from the dictionary
    hf_model_name = models_dict.get(model_name)
    
    if not hf_model_name:
        raise ValueError(f"Model name '{model_name}' is not available. Choose from: {', '.join(models_dict.keys())}")
    
    # Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModel.from_pretrained(hf_model_name)
    
    return tokenizer, model

def get_embeddings(sentence, tokenizer, model):
    """
    Generates embeddings for the input sentence(s) using the specified model and tokenizer.

    Args:
        sentence (Union[str, List[str]]): The input sentence or list of sentences to generate embeddings for.
        tokenizer: Tokenizer for the specified model.
        model: Pretrained Hugging Face model.

    Returns:
        embeddings: A single embedding (for a single sentence) or a list of embeddings (for multiple sentences).
    """
    if isinstance(sentence, str):
        # Single sentence
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    elif isinstance(sentence, list):
        # Multiple sentences
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, padding_side="right")
    else:
        raise ValueError("Input should be either a string or a list of strings.")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract embeddings from the [CLS] token (or first token)
    embeddings = outputs.last_hidden_state[:, 0, :]
    
    # Return single embedding or list of embeddings based on input type
    if isinstance(sentence, str):
        return embeddings[0]  # Return single embedding
    else:
        return embeddings  # Return list of embeddings


# Load language statements from JSON file
def load_language_statements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

# Define the path for the language statements JSON file
statements_file = './inputs/language_statements.json'

# Check if the file exists
if not os.path.exists(statements_file):
    raise FileNotFoundError(f"{statements_file} not found. Please create the file with language statements.")

# Load the statements
statements = load_language_statements(statements_file)

# Function to classify based on embedding similarity and detected language
def classify_with_similarity(predicted_answer, lang, tokenizer, model):
    # Detect the language of the predicted answer
    detected_language = lang#detect(predicted_answer)

    # Get the language-specific affirmative and negative statements
    if detected_language in statements:
        language_affirmatives = statements[detected_language]['affirmative']
        language_negatives = statements[detected_language]['negative']
    else:
        # If language is not supported, default to English
        detected_language = 'english'
        language_affirmatives = statements[detected_language]['affirmative']
        language_negatives = statements[detected_language]['negative']

    # Create embeddings for the affirmative and negative statements for the detected language
    affirmative_embeddings = get_embeddings(language_affirmatives, tokenizer, model)#model.encode(language_affirmatives, convert_to_tensor=True)
    negative_embeddings = get_embeddings(language_negatives, tokenizer, model)

    # Generate the embedding for the predicted answer
    predicted_embedding = get_embeddings(predicted_answer, tokenizer, model)

    # Calculate cosine similarity with affirmative and negative reference embeddings
    affirmative_similarity = util.cos_sim(predicted_embedding, affirmative_embeddings).mean().item()
    negative_similarity = util.cos_sim(predicted_embedding, negative_embeddings).mean().item()

    # Determine the classification based on which similarity score is higher
    classification = "yes" if affirmative_similarity > negative_similarity else "no"

    return classification

def pre_process_answers(answers, lang, tokenizer, model):
    """
    Process Answers: The pre_process_answers function classifies each response as "yes" or "no".
    """
    for answer in answers:
        answer['text'] = classify_with_similarity(answer['text'], lang, tokenizer, model)
    return answers

def pre_process_labels(answers, lang, tokenizer, model):
    """
    Process Labels: The pre_process_labels function checks the Answer field, compares it 
    with language-specific words for "yes" and "no", and converts it into binary values (1 for "yes", 0 for "no").
    """
    for i in range(len(answers)):
        # Assuming 'answers' is a list of strings
        answers[i] = classify_with_similarity(answers[i], lang, tokenizer, model)
    return answers

def eval_pope(answers, label_list_):
    """
    Evaluation: The eval_pope function computes precision, recall, F1-score, and accuracy based on the classification results and the ground truth labels.
    """
    pred_list = []
    label_list=[]
    for answer in answers:
        pred_list.append(1 if answer['text'].lower() == 'yes' else 0)
    for answer in label_list_:
        label_list.append(1 if answer.lower() == 'yes' else 0)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == 1 and label == 1:
            TP += 1
        elif pred == 1 and label == 0:
            FP += 1
        elif pred == 0 and label == 0:
            TN += 1
        elif pred == 0 and label == 1:
            FN += 1

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    acc = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN > 0 else 0
    
    return acc#precision, recall, f1, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True, help="Language for annotation and question file.")
    # Add the --model argument with default set to "multilingual-bert"
    parser.add_argument(
        '--model', 
        type=str, 
        default="multilingual-bert",  # Default value set to multilingual-bert
        choices=models_dict.keys(),  # The allowed choices are the keys from the models_dict
        help=f"Choose one of the following models: {', '.join(models_dict.keys())}. Default is 'multilingual-bert'."
    )
    
    args = parser.parse_args()

    # Load the tokenizer and model
    tokenizer, model = load_model_and_tokenizer(args.model)
    
    # Define directory and file paths based on the language
    annotation_dir = f"inputs/coco"
    question_file = f"question_files/pope_test_{args.language}.jsonl"
    result_file = f"model_responses/{args.language}_results.jsonl"
    
    # Create scores directory if it doesn't exist
    os.makedirs("scores", exist_ok=True)
    
    # Load questions
    questions = [json.loads(line) for line in open(question_file, 'r', encoding='utf-8')]
    questions = {question['question_id']: question for question in questions}

    # Load answers
    answers = [json.loads(line) for line in open(result_file, 'r', encoding='utf-8')]
    answers = pre_process_answers(answers, args.language, tokenizer, model)

    acc_overall = 0.0
    annotation_files = [file for file in os.listdir(annotation_dir) if file.startswith('coco_pope_') and file.endswith('.json')]

    # Open a text file to log results
    scores_file_path = f"scores/{args.model}_results_{args.language}.txt"
    with open(scores_file_path, 'a', encoding='utf-8') as scores_file:

        for file in annotation_files:
            category = file[10:-5]
            cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
            label_list = [json.loads(q)['label'] for q in open(os.path.join(annotation_dir, file), 'r')]
            label_list = pre_process_labels(label_list, args.language, tokenizer, model)
            acc = eval_pope(cur_answers, label_list)
            acc_overall += acc
            scores_file.write(f"Category: {category}, Accuracy: {acc:.4f}\n")  # Log the accuracy for the current category

        # Log overall accuracy
        scores_file.write(f"Overall Accuracy: {acc_overall/3.0:.4f}\n")


