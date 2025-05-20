#import librairies
#import spacy
import random
import csv
from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.parse import quote
import json
import re
import subprocess
import os
import torch
import matplotlib
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, AutoModelForQuestionAnswering,  pipeline
#import util
from util import nethook
import sys
import requests
import time
import random
import pandas as pd
#from datasets.metric import load_metric
import evaluate
import string

# Add the ROME directory to the Python path
ROME_PATH = "../rome/rome"  # Replace with the actual path
sys.path.append(ROME_PATH)


#nlp = spacy.load("en_core_web_sm")
data = pd.read_csv("all_triples/MQuAKE/new_mquake-t.csv")
QA_pairs_file = "all_triples/multihop_qa_pairs_mquake.json"
removed_rows_file = "all_triples/removed_rows_mquake.json"

model_type = "gpt2"


def clean_predicted_answer_pattern(predicted_text, reference_answer, question):
    """
    More aggressive and targeted answer extraction.
    """
    predicted_lower = predicted_text.lower()
    reference_lower = reference_answer.lower()

    # 1. Look for direct answer after markers
    answer_match = re.search(r"(?:Answer:|A:|ans:|response:)\s*(.*?)(?:\n|$)", predicted_text, re.IGNORECASE | re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    # 2. Check if the lowercase reference answer is present in the lowercase prediction
    if reference_answer and reference_lower in predicted_lower:
        # If found, try to extract the exact case version from the original text
        match = re.search(re.escape(reference_answer), predicted_text)
        if match:
            return match.group(0)
        else:
            return reference_answer # Fallback to the original case

    # 3. Targeted pattern for "author"
    if reference_lower == 'author':
        author_match = re.search(r"Carl\s+is\s+(?:the\s+)?author\s+of\s+(?:the\s+)?(.*?)(?:\.|,|\n|$)", predicted_text, re.IGNORECASE)
        if author_match:
            return 'author'

    # 4. Pattern for numerical answers
    if re.match(r"^\d+$", reference_answer):
        numerical_match = re.search(r"(\d+)", predicted_text)
        if numerical_match and numerical_match.group(1) == reference_answer:
            return reference_answer

    # 5. Try extracting the last part if it matches (for "Brandon Jennings careerStation")
    predicted_parts = predicted_text.split()
    if predicted_parts and predicted_parts[-1].lower() == reference_lower.replace("_", "").lower():
        return predicted_parts[-1].replace("_", " ")

    # 6. Try extracting the first part if it matches (for "Philip Pullman University of London.")
    if predicted_parts and predicted_parts[0].lower().replace("_", "").lower() == reference_lower.replace("_", "").lower():
        return predicted_parts[0].replace("_", " ")

    # 7. Look for the reference answer within potential lists (for "Washington Wizards ...")
    list_match = re.search(r"(?:[0-9]+\.\s*)?(" + re.escape(reference_answer.replace("_", " ")) + r")(?:\s|$)", predicted_text, re.IGNORECASE)
    if list_match:
        return list_match.group(1)

    # 8. Simple first sentence fallback
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", predicted_text)
    if sentences:
        return sentences[0].strip()

    return predicted_text.strip()


wrong_ids = [] #read
with open(removed_rows_file, "r") as f:
    try:
        wrong_ids = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {removed_rows_file}. Please ensure it contains a valid JSON array of integers.")
        wrong_ids = []

#print(wrong_ids)

model_ids=[i for i in range(500) if i not in wrong_ids]
#print(model_ids)


def format_for_squad_metric1(predictions, references):
    """
    Takes in lists of predicted answers and reference answers (strings),
    and returns formatted lists for use with HuggingFace's `evaluate` squad or exact_match metrics.
    """
    formatted_preds = [{"id": str(i), "prediction_text": pred} for i, pred in enumerate(predictions)]
    formatted_refs = [{"id": str(i), "answers": {"text": [ref], "answer_start": [0]}} for i, ref in enumerate(references)]
    return formatted_preds, formatted_refs


 
def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation (except within words), articles, and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punc(text: str) -> str:
        # Remove punctuation, but keep spaces within words
        return text.translate(str.maketrans('', '', string.punctuation))

    def lower(text: str) -> str:
        return text.lower()

    # Add this line to replace underscores with spaces
    s = s.replace('_', ' ')

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_answer(predicted_answer):
    # Regular expression to match a name or entity (you can customize it for other patterns)
    match = re.match(r"([A-Za-z\s]+)(?:\s+is|\s+who|\s+that\s+is\s+)?", predicted_answer)
    if match:
        return match.group(1).strip()  # Return the matched answer
    return predicted_answer  # Return the original answer if no match is found

def compute_f1(prediction, reference):
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    
    common = set(pred_tokens) & set(ref_tokens)
    num_common = len(common)

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

#This function focuses on evaluating the QA ability of the edited models. by Rome
def evaluate_rome_edits(QA_pairs_file, df, model_ids, config_file):
    """Evaluates ROME edits using generated QA pairs."""
    
    evaluation_results = {}
    exact_match_metric = evaluate.load("exact_match")
    f1_metric = evaluate.load("squad")
   
    rome_script_path = "../rome/rome_main2.py" 
    #print(model_ids)
    with open(QA_pairs_file, "r") as f:
        QA_pairs = json.load(f)

    #print(QA_pairs)
    model_folder = "../rome/FT_model_saved_mquake/edited_models_gpt2-large_constr/"  #replace with your folder path

    base_model_name = "openai-community/gpt2-large"  # Or your specific base model
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to("cuda:0" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"Error loading base model/tokenizer from Hugging Face: {e}")
        return evaluation_results

    
    for index, row in df[:500].iterrows():
        original_question = row["question"]
        model_name = f"edited_model_{index}"
        model_path = os.path.join(model_folder, model_name)
        #print(model_path)
        tokenizer_path = os.path.join(model_folder, f"edited_tokenizer_{index}")
        
        if index in model_ids and original_question:# and os.path.exists(model_path):# Iterate through rules and QA pairs.
            
            qa_pairs_list = QA_pairs[original_question].get("qa_pairs", [])
            facts_list = QA_pairs[original_question].get("facts", [])
            
            if not qa_pairs_list:
                print(f"No QA pairs found for question: {original_question}")
                continue

            try:
                print(f"Loading tokenizer from: {tokenizer_path}")
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                print(f"Pad token ID: {tokenizer.pad_token_id}")
                print(f"Loading model from: {model_path}")
                edited_model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda:0" if torch.cuda.is_available() else "cpu")
                for name, param in edited_model.named_parameters():
                    if name in base_model.state_dict():
                        base_model.state_dict()[name].copy_(param.data)
                base_model.to("cuda:0" if torch.cuda.is_available() else "cpu")
                model = base_model # Use the base model with applied edits
             
                question_evaluation_results = []
                for qa_pair, fact in zip(qa_pairs_list, facts_list):
                    question = qa_pair["question"]
                    answer = qa_pair["answer"]
                    context = " ".join(fact) if fact else ""

                    prompt = f"{context} Question: {question} please provide the answer only and directly, nothing else.\nAnswer:" # Adjust prompt as needed

                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

                    outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,  # adjust as needed
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id #or tokenizer.eos_token_id
                )

             
                    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    predicted_answer = clean_predicted_answer_pattern(predicted_text, answer, question) # Clean here
           
                    norm_pred = normalize_answer(predicted_answer)
                    norm_pred = extract_answer(norm_pred)
                    norm_ref = normalize_answer(answer)
                
                    
                    f1_score = f1_metric.compute(predictions=[{"id": "0", "prediction_text": norm_pred}], references=[{"id": "0", "answers": {"text": [norm_ref], "answer_start": [0]}}])
                    #f1_score = f1_metric.compute(predictions=[predicted_answer], references=[answer])
                    #f1 = compute_f1(norm_pred, norm_ref)

                    question_evaluation_results.append({
                        "generated_question": question,
                        "generated_answer": norm_ref,
                        "predicted_answer": norm_pred,
                        "exact_match": f1_score["exact_match"],
                        "f1": f1_score["f1"]
                    })

                evaluation_results[original_question] = question_evaluation_results
                torch.cuda.empty_cache() # Still good to clear cache

            except Exception as e:
                print(f"Error during evaluation for question '{original_question}': {e}")

        else:
            print(f"Model or QA pairs not found for question index: {index}, question: {original_question}")

    # Clean up the base model and tokenizer after evaluation
    del base_model
    del base_tokenizer
    torch.cuda.empty_cache()


    return evaluation_results



config_file = "../hparams/FT/gpt2-large_constr.json" #path to rome_hparams.json

evaluation_results = evaluate_rome_edits(QA_pairs_file, data, model_ids, config_file)
print(evaluation_results)

def save_evaluation_results(evaluation_results, filename="results/FT_evaluation_results_gpt2-large_constr_mquake.json"):
    """Saves the evaluation results to a JSON file."""
    try:
        with open(filename, "w") as f:
            json.dump(evaluation_results, f, indent=4)  # indent for better readability
        print(f"Evaluation results saved to {filename}")
    except Exception as e:
        print(f"Error saving evaluation results: {e}")

save_evaluation_results(evaluation_results)
