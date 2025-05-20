from preprocess_rules import rules_list, extract_relations
from SparqlQuery import extract_entities, process_triples_file, query_sparql_endpoint, find_dbpedia_uri
import spacy
import random
import csv
from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.parse import quote
import json
import re
import subprocess
import os
import time
#import openai
import pandas as pd
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch


data = pd.read_csv("all_triples/MQuAKE/new_mquake-t.csv")

triples = 'all_triples/processed_triples_mquake.txt'

rules = 'all_triples/rules_mquake.txt'

#model="google/gemma-3-27b-it"
model = "distilbert-base-cased-distilled-squad"

#fact_file = "./all_triples/processed_triples.txt"



if torch.cuda.is_available():
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available on this machine.")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

def get_prompt(Rules, Facts):
    prompt = f"""Given the following rule and a set of facts, your task is to generate **one** multihop question whose answer is a single entity present in the facts. The question should require reasoning based on the provided rule and the connections within the facts.

Rule:
{Rules}

Facts:
{Facts}

Output the question and its corresponding single-entity answer in the following format:
Question: [your multihop question]
Answer: [the single entity answer from the facts]

For example, consider the rule: '?person bornIn ?place => ?person nationality ?place' and the facts: 'John bornIn Germany'. A possible multihop question is:
Question: John was born in which place, and what is the nationality associated with that place?
Answer: Germany

Now, based on the rule: {Rules} and the facts: {Facts}, generate one question and its answer:
Question:
Answer:
"""
    return prompt


def ensure_valid_uri(uri):
    if isinstance(uri, list):
        return uri[0] if uri else None
    return uri



def format_rule(rule_tuple):
    """
    Converts a rule tuple in the format ('antecedent', '=>', 'consequent')
    to a string in the format 'antecedent => consequent'.

    Args:
        rule_tuple: A tuple representing the rule.

    Returns:
        A string representing the formatted rule.
    """
    #if not isinstance(rule_tuple, tuple) or len(rule_tuple) != 3:
        #raise ValueError("Input must be a tuple of length 3.")

    antecedent = rule_tuple[0].strip()
    operator = rule_tuple[1].strip()
    consequent = rule_tuple[2].strip()

    if operator != "=>":
        raise ValueError("Tuple second element must be '=>'")

    return f"{antecedent} => {consequent}"


relations = extract_relations(data, triples)
filtered_rules = rules_list(relations, rules)
#print(f"Number of relations: {len(relations)}")
#print(f"Number of rules filterd: {len(filtered_rules)}")

def new_rules(filtered_rules_list):
    new_rules = []
    for rule in filtered_rules_list:
        rule  = format_rule(rule)
        new_rules.append(rule)

    return new_rules

Rules = new_rules(filtered_rules)
print(f"Number of rules loaded before function: {len(Rules)}")

def extract_rule_relations(rule_string):
    """Extracts relations from a rule string."""
    relations = set()  # Use a set to avoid duplicates
    parts = rule_string.split("=>")
    if len(parts) == 2:  # Check if it's a valid rule
        antecedent = parts[0].strip()
        consequent = parts[1].strip()

        # Extract relations from antecedent
        antecedent_relations = re.findall(r"\?\w+\s+(\w+)\s+\?\w+", antecedent)
        relations.update(antecedent_relations)

        # Extract relations from consequent
        consequent_relations = re.findall(r"\?\w+\s+(\w+)\s+\?\w+", consequent)
        relations.update(consequent_relations)

    return list(relations)

def generate_multihop_qa_huggingface_local(df, rules, model_name, max_retries=3, endpoint="https://dbpedia.data.dice-research.org/sparql", output_file="all_triples/processed_triples_mquake.txt", output_dir="all_triples", output_qa_file="all_triples/multihop_qa_pairs_mquake.json", max_qa_pairs_per_question=5):
    """
    Generates multihop questions and answers, ensuring each rule is used at most once per original question.
    """
    qa_pipeline = pipeline("question-answering", model=model_name, device=0)
    question_generation_model_name = "google/flan-t5-small"
    question_generator_tokenizer = AutoTokenizer.from_pretrained(question_generation_model_name)
    question_generator_model = AutoModelForSeq2SeqLM.from_pretrained(question_generation_model_name).to(0)

    dict_qa_pairs = {}
    all_triples = {}
    remove_indices = set()
    os.makedirs(output_dir, exist_ok=True)

    print(f"Number of rules loaded: {len(Rules)}")
    for index, row in df[:1000].iterrows():
        print(f"Processing index: {index}, question: {row['question']}")
        original_question = row["question"]

        if index in remove_indices:
            print(f"Skipping question '{original_question}' (index {index}) as it was marked for removal.")
            continue
             
        if original_question not in dict_qa_pairs:
            dict_qa_pairs[original_question] = {"rules": [], "facts": [], "qa_pairs": []}
        qa_pair_count = 0
        rules_used_for_current_question = set()
        print(f"Processing question: {original_question}, initial used rules: {rules_used_for_current_question}")

        entities = extract_entities(original_question)
        dbpedia_uris = [find_dbpedia_uri(ent) for ent in entities]
        valid_uris = [uri for uri in dbpedia_uris if uri]
        valid_uris = ensure_valid_uri(valid_uris)
        #print(valid_uris)
        #break

        if valid_uris:
            triples = query_sparql_endpoint(endpoint, valid_uris)
            all_triples[original_question] = triples
        else:
            print(f"No valid DBpedia URIs found for entities in question: {original_question}")
            remove_indices.add(index)
            continue

        processed_all_triples = process_triples_file(all_triples, output_file)

        if original_question not in processed_all_triples:
            remove_indices.add(index)
            continue

        processed_triples_for_question = processed_all_triples[original_question]

        if not processed_triples_for_question:
            print(f"No processed triples found for question: {original_question}")
            remove_indices.add(index)
            continue

        relations_in_question = set()
        for triple_str in processed_triples_for_question:
            triple_parts = triple_str.split('\t')
            if len(triple_parts) == 3:
                predicate = triple_parts[1]
                relations_in_question.add(predicate)

        relevant_rules = [rule for rule in rules if any(rel in relations_in_question for rel in extract_rule_relations(rule))]
        print(f"  Number of relevant rules: {len(relevant_rules)} / {len(rules)}")
        
       # print(f"Processing question: {original_question}, relevant rules: {relevant_rules}")

        if original_question not in dict_qa_pairs:
            dict_qa_pairs[original_question] = {"rules": [], "facts": [], "qa_pairs": [], "score": []}

        dict_qa_pairs[original_question]["rules"] = relevant_rules  # Save ALL relevant rules
        
        for rule in relevant_rules:
           # print(f"  Trying rule: {rule}") 
            if rule in rules_used_for_current_question:
                continue

            for original_triple_str in processed_triples_for_question:
                original_triple_parts = original_triple_str.split('\t')
                print(original_triple_parts)
                if len(original_triple_parts) == 3:
                    subject = original_triple_parts[0]
                    predicate = original_triple_parts[1]
                    object_ = original_triple_parts[2]
                    simplified_facts = [subject, predicate, object_]

                    prompt = get_prompt([rule], simplified_facts)

                    input_ids = question_generator_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(question_generator_model.device)

                    outputs = question_generator_model.generate(
                        input_ids,
                        max_length=150,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        top_k=50,
                        top_p=0.95,
                        temperature=0.7
                    )

                    generated_text = question_generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    #print(f"    Generated Text: {generated_text}")
                    #break

                    retries = 0
                    while retries < max_retries:
                        try:
                            print(f"    Generated Question for QA: '{generated_text}'")
                            print(f"    Context for QA: '{' '.join(simplified_facts)}'")
                            qa_result = qa_pipeline(question=generated_text, context=" ".join(simplified_facts))
                            print(f"      QA Result: {qa_result}")
                            #break
                            generated_answer = qa_result["answer"].strip()
                            generated_score = qa_result["score"]

                            #dict_qa_pairs[original_question]["rules"].append(relevant_rules)
                            dict_qa_pairs[original_question]["facts"].append(simplified_facts)
                            dict_qa_pairs[original_question]["qa_pairs"].append({"question": generated_text, "answer": generated_answer, "score": generated_score})
                            #print(dict_qa_pairs[original_question])
                            #break
                            #dict_qa_pairs[original_question]["score"].append(generated_score)
                            rules_used_for_current_question.add(rule)
                            qa_pair_count += 1
                            break  # Break retry loop

                        except Exception as e:
                            print(f"      Error during local QA model inference: {e}")
                            print(f"      Failing Question: '{generated_text}'")
                            print(f"      Failing Context: '{' '.join(simplified_facts)}'")
                            retries += 1
                            time.sleep(random.uniform(1, 3))
                        if qa_pair_count >= max_qa_pairs_per_question:
                            break # Break out of the inner while loop if limit reached

                if qa_pair_count >= max_qa_pairs_per_question:
                    break # Break out of the triples loop if limit reached

            if qa_pair_count >= max_qa_pairs_per_question:
                break # Break out of the rules loop if limit reached

    removed_rows_file = os.path.join(output_dir, "removed_rows_mquake.json")
    with open(removed_rows_file, "w") as f:
        json.dump(list(remove_indices), f)
    if remove_indices:
        print(f"Removing {len(remove_indices)} rows due to missing DBpedia entities.")
        df = df.drop(index=remove_indices).reset_index(drop=True)

    with open(output_qa_file, 'w', encoding='utf-8') as f:
        json.dump(dict_qa_pairs, f, indent=4)

    return dict_qa_pairs


result = generate_multihop_qa_huggingface_local(data, Rules,  model)


