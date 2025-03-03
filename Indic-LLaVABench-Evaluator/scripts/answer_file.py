#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import argparse
import os
import json
from tqdm import tqdm
import shortuuid

'''
The input JSONL file (input.jsonl) should have the following structure:
Each line is a separate JSON object with the following keys:
- "image": A string representing the image filename (e.g., "001.jpg")
- "text": A string containing the question in a specific language (e.g., "ফটোখনত থকা এই বিখ্যাত দৃশ্যটোৰ নাম কি?")
- "category": A string representing the category (e.g., "conv")
- "question_id": An integer that uniquely identifies the question (e.g., 0)
- "response": A string containing the model's response (e.g., "ফটোখনত থকা এই বিখ্যাত দৃশ্যটোৰ নাম কি?")
- "model_name": A string representing the model's name (e.g., "xxx")

Example of input.jsonl:
{"image": "001.jpg", "text": "ফটোখনত থকা এই বিখ্যাত দৃশ্যটোৰ নাম কি?", "category": "conv", "question_id": 0, "Response": "ফটোখনত থকা এই বিখ্যাত দৃশ্যটোৰ নাম কি?", "model_name": "xxx"}
'''

def eval_model(args):
    # Load input JSONL file with questions and model responses
    with open(os.path.expanduser(args.response_file), "r") as f:
        questions = [json.loads(line) for line in f]

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    answered_questions = set()
    # Check if answer file exists and collect already processed question IDs
    if os.path.exists(answers_file):
        with open(answers_file, "r") as file:
            for line in file:
                data = json.loads(line)
                answered_questions.add(data["question_id"])

    # Open answer file for appending new results
    with open(answers_file, "a") as ans_file:
        for line in tqdm(questions):
            idx = line["question_id"]
            # Skip if already answered
            if idx in answered_questions:
                continue

            # Extract fields from input JSONL
            cur_prompt = line["text"]  # The text field
            outputs = line["response"].strip()  # The model response
            model_name = line["model_name"]  # Model name from input

            # Generate a unique answer ID
            ans_id = shortuuid.uuid()

            # Write the processed answer to the output file
            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt": cur_prompt,
                "text": outputs,
                "answer_id": ans_id,
                "model_id": model_name,
                "metadata": {}
            }, ensure_ascii=False) + "\n")
            ans_file.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Input and output file arguments
    parser.add_argument("--response-file", type=str, default="./model_responses/responses_Telugu.jsonl")
    parser.add_argument("--answers-file", type=str, default="./answers/answers.jsonl")
    
    args = parser.parse_args()

    eval_model(args)
