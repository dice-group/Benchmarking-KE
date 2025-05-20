from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
import argparse
import warnings
import json
import os
import sys
from ft_hparams import FTHyperParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="FT_models_saved")
    return parser.parse_args()

def load_hparams(config_path):
    if not isinstance(config_path, str):
        raise TypeError(f"Config path must be a string, not {type(config_path)}")

    with open(config_path, "r") as f:
        config = json.load(f)
    return FTHyperParams(**config)

def load_model_and_tokenizer(model_name):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer pad_token: {tokenizer.pad_token}", file=sys.stderr)
    return model, tokenizer

def parse_request(request_str):
    return json.loads(request_str)

def apply_ft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    copy=False,
    return_orig_weights=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_ft(model, tok, requests, hparams)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"][0] != " ":
            # Space required for correct tokenization
            request["target_new"] = " " + request["target_new"]
        print(
            f"Executing FT algo for: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    # Define inputs
    texts = [r["prompt"].format(r["subject"]) for r in requests]
    targets = [r["target_new"] for r in requests]

    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt, tgt in zip(
            chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            inputs = tok(txt, return_tensors="pt", padding=True).to("cuda")
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                "cuda"
            )
            last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
            loss_mask = target_ids != tok.unk_token_id

            opt.zero_grad()
            bs = inputs["input_ids"].shape[0]
            probs = torch.nn.functional.log_softmax(
                model(**inputs).logits[torch.arange(bs), last_token_inds], dim=-1
            )
            loss = -(torch.gather(probs, 1, target_ids) * loss_mask).sum(
                1
            ) / loss_mask.sum(1)
            loss = loss.mean()
            print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=bs)

            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )

        print(f"Total loss {loss_meter.avg}")

        if loss_meter.avg < 1e-2:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def apply_ft_and_save(model, tokenizer, request, hparams, save_dir, prompt_index):
    model.to(hparams.device)
    edited_model, _ = apply_ft_to_model(model, tokenizer, [request], hparams)
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
        apply_ft_and_save(model, tokenizer, request, hparams, save_dir, prompt_index)

if __name__ == "__main__":
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    main(args.dataset_path, args.model_name, args.config, args.save_dir)