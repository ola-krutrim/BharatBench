#    Licensed under the MIT License

# Krutrim-AI-Labs Changes / Modifications done as follows:
# Pre-process Answers and Labels of Generated Responses by Implementation of OpenAI API calling for Evaluations.

import openai
import os
import json
import argparse
import sys
from tqdm import tqdm

openai.api_key = os.environ.get("OPENAI_API_KEY")


# Function to classify a predicted answer as 'Yes' or 'No' based on its meaning
def classify_predicted_answer(predicted_answer):
   
    # Updated classification prompt to handle descriptive statements
    classification_prompt = f"""
    Based solely on the following predicted answer, determine whether it should be classified as 'yes' or 'no'.
    The model should evaluate the semantic meaning of the prediction:
    - If the predicted answer indicates affirmation, agreement, supports a positive outcome, or expresses positive sentiment, classify it as 'yes'.
    - If the predicted answer indicates denial, contradiction, implies a negative outcome, or expresses negative sentiment, classify it as 'no'.

    Predicted Answer: {predicted_answer}

    Response (answer in English only):
    """

    # Call OpenAI GPT-4 model with the classification prompt
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": classification_prompt}
        ]
    )
    
    # Extract and return the classification ('Yes' or 'No') in English from the response
    return response['choices'][0]['message']['content'].strip()

'''
Process Answers: The pre_process_answers function classifies each response as "yes" or "no" using GPT-4.
'''


def pre_process_answers(answers):
    for answer in answers:
        answer['text'] = classify_predicted_answer(answer['text'])
    return answers

def pre_process_labels(answers):
    """
    Process Labels: The pre_process_labels function checks the Answer field, compares it 
    with language-specific words for "yes" and "no", and converts it into binary values (1 for "yes", 0 for "no").
    """
    for i in range(len(answers)):
        # Assuming 'answers' is a list of strings
        answers[i] = classify_predicted_answer(answers[i])
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
    
    return acc  #precision, recall, f1, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True, help="Language for annotation and question file.")
    
    args = parser.parse_args()

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
    answers = pre_process_answers(answers)

    acc_overall = 0.0
    annotation_files = [file for file in os.listdir(annotation_dir) if file.startswith('coco_pope_') and file.endswith('.json')]

    # Open a text file to log results
    scores_file_path = f"scores/results_{args.language}.txt"
    with open(scores_file_path, 'a', encoding='utf-8') as scores_file:

        for file in tqdm(annotation_files):
            category = file[10:-5]
            cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
            label_list = [json.loads(q)['label'] for q in open(os.path.join(annotation_dir, file), 'r')]
            label_list = pre_process_labels(label_list)
            acc = eval_pope(cur_answers, label_list)
            acc_overall += acc

            # Log the accuracy for the current category
            scores_file.write(f"Category: {category}, Accuracy: {acc:.4f}\n")

        # Log overall accuracy
        scores_file.write(f"Overall Accuracy: {acc_overall/3.0:.4f}\n")


