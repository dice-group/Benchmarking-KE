from copy import deepcopy
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import warnings
import json
import os
import sys
from kn_hparams import KNHyperParams
from knowledge_neurons.knowledge_neurons import KnowledgeNeurons, model_type


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="KN_models_saved")
    return parser.parse_args()

def load_hparams(config_path):
    if not isinstance(config_path, str):
        raise TypeError(f"Config path must be a string, not {type(config_path)}")

    with open(config_path, "r") as f:
        config = json.load(f)
    return KNHyperParams(**config)

def load_model_and_tokenizer(model_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer pad_token: {tokenizer.pad_token}", file=sys.stderr)
    return model, tokenizer

def parse_request(request_str):
    return json.loads(request_str)
    

def apply_kn_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: KNHyperParams,
    copy=False,
    return_orig_weights=False,
) -> Tuple[AutoModelForCausalLM, List[str]]:

    kn = KnowledgeNeurons(
        model,
        tok,
        model_type=model_type(hparams.model_name),
        device="cuda",
    )
    request_rewrite = deepcopy(request)

    #print("Debugging: request_rewrite:", request_rewrite)  # Print the entire dictionary
    #print("Debugging: request_rewrite['prompt']:", request_rewrite["prompt"])  # Print the value of "prompt"
    #print("Debugging: Type of request_rewrite['prompt']:", type(request_rewrite["prompt"]))  # Print the data type
    
    #text = [request_rewrite["prompt"].format(request_rewrite["subject"])]
    text = [request_rewrite[0]["prompt"].format(request_rewrite[0]["subject"]) if isinstance(request_rewrite[0]["prompt"], str) else request_rewrite[0]["prompt"]]
    ground_truth = request_rewrite[0]["target_true"]#["str"]
    target = request_rewrite[0]["target_new"]#["str"]

    kn.model = kn.model.to(kn.device)
    refined_neurons = kn.get_refined_neurons(
        text,
        ground_truth,
        p=hparams.p,
        batch_size=hparams.batch_size,
        steps=hparams.steps,
        coarse_adaptive_threshold=hparams.adaptive_threshold,
        refine=hparams.refine,
    )

    results_dict, unpatch_fn = kn.edit_knowledge(
        text[0],
        target=target,
        neurons=refined_neurons,
        undo_modification=False,
    )
    updated_model = deepcopy(kn.model)
    with torch.no_grad():
        unpatch_fn()
    return updated_model, {}


def apply_kn_and_save(model, tokenizer, request, hparams, save_dir, prompt_index):
    model.to(hparams.device)
    edited_model, _ = apply_kn_to_model(model, tokenizer, [request], hparams)
    os.makedirs(save_dir, exist_ok=True)

    save_path_model = os.path.join(save_dir, f"edited_model_{prompt_index}") # change to directory.
    save_path_tokenizer = os.path.join(save_dir, f"edited_tokenizer_{prompt_index}")
    
    # Save the entire model, not just the state_dict
    edited_model.save_pretrained(save_path_model)
    tokenizer.save_pretrained(save_path_tokenizer)

    print(f"Model saved to {save_path_model}")
    print(f"Tokenizer saved to {save_path_tokenizer}")

def main(dataset_path, model, config_path, save_dir):
    if torch.cuda.is_available():
        device_id = 1
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    with open(dataset_path, "r") as f:
        data = json.load(f)
    args = parse_args()
    hparams = load_hparams(config_path)
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    for prompt_index, request in enumerate(data):
        print(request)
        apply_kn_and_save(model, tokenizer, request, hparams, save_dir, prompt_index)

if __name__ == "__main__":
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    main(args.dataset_path, args.model_name, args.config, args.save_dir)