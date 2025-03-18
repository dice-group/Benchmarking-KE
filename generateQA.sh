#!/bin/bash

# Set your Hugging Face API key as an environment variable or replace this with your key
HUGGINGFACE_API_KEY="${HUGGINGFACE_API_KEY:-$(read -sp "Enter Hugging Face API key: " API_KEY; echo "$API_KEY")}"
export HUGGINGFACE_API_KEY

# Set the model to use
MODEL="google/gemma-3-27b-it" # or try "another model"

# Path to the Python file containing the rule generation function
RULE_GENERATION_FILE="generateQA.py" # Replace with your file path

# Generate rules using the external Python function
RULES=$(python3 "$PYTHON_FILE" | jq -c '.rules')

# Check if rules were generated successfully
if [[ -z "$RULES" ]]; then
  echo "Error: Failed to generate rules from $RULE_GENERATION_FILE"
  exit 1
fi


FACTS=(
    "George_Gervin isPrimaryTopicOf George_Gervin"
    "George_Gervin team Chicago_Bulls"
    "John_LeRoy_Hennessy wikiPageRedirects John_L._Hennessy"
)

# Python script to generate questions and answers
PYTHON_SCRIPT=$(cat <<'EOF'
import os
import sys
import json
from generateQA import generate_multihop_qa_huggingface, new_rules


def generate_qa_from_rules_and_facts(rules, facts, api_key, model):
    qa_results = {}
    result = generate_multihop_qa_huggingface(rule, facts, api_key, model)
    if result:
        question, answer = result
        qa_results[rule] = (question, answer)
    else:
        qa_results[rule] = ("Failed to generate question/answer.", "Failed to generate question/answer.")
    return qa_results

rules = json.loads(sys.argv[1])
facts = json.loads(sys.argv[2])
api_key = os.environ.get("HUGGINGFACE_API_KEY")
model = sys.argv[3]

qa_dict = generate_qa_from_rules_and_facts(rules, facts, api_key, model)

print(json.dumps(qa_dict))
EOF
)

# Convert rules and facts to JSON strings for passing to Python script
FACTS_JSON="["
for fact in "${FACTS[@]}"; do
    IFS=' ' read -r subject predicate object <<< "$fact"
    FACTS_JSON+="[\"$subject\", \"$predicate\", \"$object\"],"
done
FACTS_JSON="${FACTS_JSON%,}]" #Remove last comma and close array.

# Execute the Python script and capture the JSON output
JSON_OUTPUT=$(echo "$RULES" | python3 -c "$PYTHON_SCRIPT" "$RULES" "$FACTS_JSON" "$MODEL")

# Parse the JSON output and print the results
QA_DICT=$(echo "$JSON_OUTPUT" | jq -r 'to_entries[] | [.key, .value[0], .value[1]] | @tsv')

echo "$QA_DICT" | while IFS=$'\t' read -r rule question answer; do
  echo "Rule: $rule"
  echo "Question: $question"
  echo "Answer: $answer"
  echo "--------------------"
done