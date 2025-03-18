from preprocess_rules import rules_list, extract_relations
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


data = pd.read_csv("all_triples/en_qa.csv")

triples = 'all_triples/processed_triples.txt'

rules = 'all_triples/triples_mined-rules1.txt'

model="google/gemma-3-27b-it"

def get_prompt(Rules, Facts):
    prompt = f"""Given rules and facts your task is to generate a multihop questions that is different from the example provided below, and whose answers should be one entity in the given facts.   The answer should be an entity, nothing else.
    The question should encode knowledge from the rules and the facts. You should also provide the answer to the question in the following format:
    <question> ... your question goes here </question>
    <answer> ... your answer goes here </answer>
    
    For example, when given the following rules and facts:
    
    Rules:
    '?a bornIn ?b  ?g citizenship ?a ==> ?a nationaity ?g',
    '?g travelsTo ?b ?g staysInHotel ?h ==> ?g notResidentOf ?b',
    '?a worksAt ?b ==> ?a livesIn ?b'                    
           
    Facts:
    ('John', 'worksAt', 'DICE'),
    ('John', 'BornIn', 'Germany'),
    ('John, 'TravelTo', 'Germany')
    You may write:
    <question>John was born in Germany. Which nationality does John have? </question>
    <answer>Germany</answer>
    
    Now please proceed the same way for the following rules and facts:   
    Rules: 
    '\n'.join(Rules[:10])

    Facts: 
    '\n'.join(list(map(str, Facts)))
    
    """

    return prompt


def generate_multihop_qa_huggingface(rule, facts, api_key, model, max_retries=10, num_questions=2):
    """
    Generates multihop questions and answers based on a given rule and facts using Hugging Face Inference API.

    Args:
        rule (str): The rule in the format "a patternLa b => a patternB b".
        facts (list): A list of triples, where each triple is a tuple (subject, predicate, object).
        api_key (str): Your Hugging Face API key.
        model (str): The Hugging Face model to use (default: "google/flan-t5-xxl").

    Returns:
        tuple: A tuple containing the generated question and answer, or None if an error occurs.
    """
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    qa_pairs = [] # Store multiple qa pairs
    #generated_prompts = set()
    for _ in range(num_questions):
        retries = 0
        while retries < max_retries:
            
            try:

                # Check if prompt was already generated
                #prompt_variation = f" Generate a question with variation {i+1}."
                prompt = get_prompt([rule], facts) #+ prompt_variation
                
                payload = {
                    "inputs": prompt,
                    "options": {"wait_for_model": True}
                }
        
                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
                response_json = response.json()
                output = response_json[0]["generated_text"].strip() #this can change based on the model.
        
                question_start = output.find("Question:") + len("Question:")
                answer_start = output.find("Answer:") + len("Answer:")
        
                question_end = output.find("Answer:")
                if question_end == -1:
                  question_end = len(output)
        
                question = output[question_start:question_end].strip()
                answer = output[answer_start:].strip()

                qa_pairs.append({"question": question, "answer": answer}) #store as dict
                break
        
                #return question, answer
    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 503:
                    retries += 1
                    delay = (2 ** retries) + random.random()
                    print(f"Received 503 error. Retrying in {delay:.2f} seconds.")
                    time.sleep(delay)
                else:
                    print(f"Error during Hugging Face API request: {e}")
                    return qa_pairs
            except Exception as e:
                    print(f"Error during Hugging Face API request: {e}")
                    return qa_pairs
        if retries == max_retries:
            print("Maximum retries reached. Unable to complete request.")
    return qa_pairs
    

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

def new_rules(filtered_rules_list):
    new_rules = []
    for rule in filtered_rules_list:
        rule  = format_rule(rule)
        new_rules.append(rule)

    return new_rules

Rules = new_rules(filtered_rules)
#print(Rules)

facts = [
    ("George_Gervin", "isPrimaryTopicOf", "George_Gervin"),
    ("George_Gervin", "team", "Chicago_Bulls"),
    ("John_LeRoy_Hennessy",	"wikiPageRedirects", "John_L._Hennessy")
]

api_key = os.getenv("HUGGINGFACE_API_KEY") #or input your api key directly.
if api_key is None:
    api_key = input("Please enter your Hugging Face API key: ")
    
result = generate_multihop_qa_huggingface(Rules, facts, api_key, model)
    
#print(result)
if result:
    for elt in result:
        question = elt["question"]
        answer = elt["answer"]
        print("Question:", question)
        print("Answer:", answer)
