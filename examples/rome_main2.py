from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
#import util
from util import nethook
from util.generate import generate_fast
import sys
from compute_u import compute_u
from compute_v import compute_v
from rome_hparams import ROMEHyperParams
import warnings
import argparse
import json
import os

CONTEXT_TEMPLATES_CACHE = None

#cmd: python rome_main2.py --model_name openai-community/gpt2-large --dataset_path ../evaluate_rules/all_triples/MLaKE/new_en-qa.json --config ../hparams/ROME/gpt2-large.json --save_dir edited_models

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="Rome_models_saved")
    return parser.parse_args()

def load_hparams(config_path):
    if not isinstance(config_path, str):
        raise TypeError(f"Config path must be a string, not {type(config_path)}")

    with open(config_path, "r") as f:
        config = json.load(f)
    return ROMEHyperParams(**config)

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

#def generate_context_templates(subject: str):
#    return [
#        f'{subject} {{}}',
#        f'A woman has been {{}} {subject}',
#        f'The first-ever {{}} for {subject}',
#    ]

#def apply_rome_to_model(
#    model: AutoModelForCausalLM,
#    tok: AutoTokenizer,
#    requests: List[Dict],
#    hparams: ROMEHyperParams,
#    copy=False,
#    return_orig_weights=False,
#) -> Tuple[AutoModelForCausalLM, List[str]]:
#    request = requests[0]
#
#    device = next(model.parameters()).device
#    if "subject" not in request or "target_new" not in request:
#        print("Error: subject or target_new key missing from request in apply_rome_to_model.", file=sys.stderr)
#        return None, None
#    subject = request["subject"]
#    if copy:
#        model = deepcopy(model)
#
#    weights_copy = {}
#    #context_templates = generate_context_templates(subject)
#    deltas = execute_rome(model, tok, request, hparams, context_templates)
#
#    with torch.no_grad():
#        for w_name, (delta_u, delta_v) in deltas.items():
#            upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
#            w = nethook.get_parameter(model, w_name)
#            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
#
#            if return_orig_weights and w_name not in weights_copy:
#                weights_copy[w_name] = w.detach().clone()
#            upd_matrix = delta_v.to(device)
#            w[...] += upd_matrix
#
#    print(f"New weights successfully inserted into {list(deltas.keys())}", file=sys.stderr)
#    return model, weights_copy

def apply_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    #subject = request["subject"]
    if copy:
        model = deepcopy(model)

    weights_copy = {}

    for i, request in enumerate(requests):
        deltas = execute_rome(model, tok, request, hparams)

        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                w = nethook.get_parameter(model, w_name)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

                if return_orig_weights and w_name not in weights_copy:
                    assert i == 0
                    weights_copy[w_name] = w.detach().clone()

                w[...] += upd_matrix

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy
    

def execute_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    #context_templates: List[str]
) -> Dict[str, Tuple[torch.Tensor]]:
    device = next(model.parameters()).device
    request = deepcopy(request)

    #print(f"Executing ROME algorithm for request: {request}", file=sys.stderr)
    #print(f"Hparams layers: {hparams.layers}, rewrite_module_tmp: {hparams.rewrite_module_tmp}", file=sys.stderr)

    if isinstance(request["target_new"], str):
        if request["target_new"][0] != " ":
            request["target_new"] = " " + request["target_new"]
        print(
            f"Executing ROME algorithm for the update: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]", file=sys.stderr
        )
    else:
        print("target_new is not a string, using default", file=sys.stderr)

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    deltas = {}
    for layer in sorted(hparams.layers):
        left_vector: torch.Tensor = compute_u(
            model,
            tok,
            request,
            hparams,
            layer,
            get_context_templates(model, tok, hparams.context_template_length_params),
        ).to(device)
        #print("Left vector shape:", left_vector.shape, file=sys.stderr)
        print(f"Layer {layer}: Left vector shape = {left_vector.shape}, max abs value = {torch.abs(left_vector).max()}", file=sys.stderr)
        right_vector: torch.Tensor = compute_v(
            model,
            tok,
            request,
            hparams,
            layer,
            left_vector,
            get_context_templates(model, tok, hparams.context_template_length_params),
        ).to(device)
        #print("Right vector shape:", right_vector.shape, file=sys.stderr)
        print(f"Layer {layer}: Right vector shape = {right_vector.shape}, max abs value = {torch.abs(right_vector).max()}", file=sys.stderr)

        with torch.no_grad():
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            w = nethook.get_parameter(model, weight_name)
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            print(f"Layer {layer}: Original weight shape = {w.shape}, update matrix shape before match = {upd_matrix.shape}", file=sys.stderr)
            #upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
            print(f"Layer {layer}: Update matrix shape after match = {upd_matrix.shape}, max abs value = {torch.abs(upd_matrix).max()}", file=sys.stderr)

            #print(f"Weights before update: {weights[weight_name]}", file=sys.stderr)
            upd_matrix = upd_matrix.to(device)
            #weights[weight_name][...] += upd_matrix
            w[...] += upd_matrix
            print(f"Layer {layer}: Updated weight max abs value = {torch.abs(w).max()}", file=sys.stderr)
            #print(f"Weights after update: {weights[weight_name]}", file=sys.stderr)
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}", file=sys.stderr)

    return deltas

#def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
#    if matrix.shape == shape:
#        return matrix
#    elif matrix.T.shape == shape:
#        return matrix.T
#    else:
#        raise ValueError(
#            "Update matrix computed by ROME does not match original weight shape. "
#            "Check for bugs in the code?"
#        )
def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )

def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["<|endoftext|>"],
                        n_gen_per_prompt=n_gen,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}", file=sys.stderr)

    return CONTEXT_TEMPLATES_CACHE

def generate_prediction(model, tokenizer, prompt):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=15)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction

def evaluate_prediction(prediction, ground_truth):
    return prediction.strip() == ground_truth.strip()

def apply_rome_and_save(model, tokenizer, request, hparams, save_dir, prompt_index):
    model.to(hparams.device)
    edited_model, _ = apply_rome_to_model(model, tokenizer, [request], hparams)
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
        device_id = 0
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
        apply_rome_and_save(model, tokenizer, request, hparams, save_dir, prompt_index)

if __name__ == "__main__":
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    main(args.dataset_path, args.model_name, args.config, args.save_dir)