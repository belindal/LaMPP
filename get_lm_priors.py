"""
Queries LM for priors
"""
import argparse
import os
import numpy as np
from tqdm import tqdm
from gpt3_utils import (
    gpt3_score_prompt, save_gpt3_result, load_gpt3_cache,
)
import json
import pathlib
import itertools as it
import torch
import torch.nn.functional as F


def parse_query_config(query_config_fp, gpt3_cache_dir_fp):
    query_config = json.load(open(query_config_fp))
    gpt_scoring_cache_fn = gpt3_cache_dir_fp / f"{query_config['engine']}.jsonl"
    gpt3_scoring_cache = load_gpt3_cache(gpt_scoring_cache_fn)
    return query_config, gpt_scoring_cache_fn, gpt3_scoring_cache


def get_gpt3_cooccurrences(query_config, output_logits_path, gpt_scoring_cache_fn, gpt3_scoring_cache, print_prompt=False): #, items, prompt_type):
    all_inputs = query_config["inputs"]
    if not os.path.exists(output_logits_path):
        input_replace_key_value_pairs = [[(k, [inp_idx, inp]) for inp_idx, inp in enumerate(all_inputs[k])] for k in all_inputs]
        cooccur_matrix = []
        all_input_kv_combos = list(it.product(*input_replace_key_value_pairs))
        for input_kvs in tqdm(all_input_kv_combos):
            # get prompt for current example
            curr_prompt = query_config["prompt_format"]
            for input_kv in input_kvs:
                curr_prompt = curr_prompt.replace(f"{{{input_kv[0]}}}", input_kv[1][1])

            # get prompt for previous examples
            if "prompt_demos" in query_config:
                prompt_demos = query_config["prompt_demos"]
            else:
                prompt_demos = []
                for label in query_config["examples"].keys():
                    prompt_demos.extend([
                        ex for ex in query_config["examples"][label]
                        if ex != (curr_prompt + label)
                    ][:query_config["n_examples"][label]])
                prompt_demos = "\n".join(prompt_demos)

            full_prompt = f"{prompt_demos}\n{curr_prompt}"
            if print_prompt:
                # Print full prompt
                print("= Prompt =")
                print(full_prompt + "[" + "|".join(query_config["classes"]) + "]")
                print("= END =")

            # get scores per class
            cond_scores, new_results = gpt3_score_prompt(
                query_config["engine"],
                full_prompt,
                query_config["classes"],
                gpt3_scoring_cache,
            )
            save_gpt3_result(gpt_scoring_cache_fn, new_results)
            class_prob = F.softmax(torch.tensor(cond_scores),dim=0)

            selected_cell = cooccur_matrix
            for input_kv_dim in input_kvs:
                if len(selected_cell) <= input_kv_dim[1][0]:
                    selected_cell.append([])
                selected_cell = selected_cell[input_kv_dim[1][0]]
            if query_config["binary_pred"]:
                selected_cell.append(class_prob[0].item())
            else:
                # p(col|row)
                selected_cell.append(class_prob.tolist())

        cooccur_matrix = np.squeeze(np.array(cooccur_matrix), axis=-1)
        np.save(output_logits_path, cooccur_matrix)
    else:
        cooccur_matrix = np.load(output_logits_path)

    return cooccur_matrix


def main(args):
    query_config, gpt_scoring_cache_fn, gpt3_scoring_cache = parse_query_config(
        args.query_config, args.gpt3_cache_fp)
    gpt3_cooccurs = get_gpt3_cooccurrences(
        query_config, args.output_save_path,
        gpt_scoring_cache_fn, gpt3_scoring_cache,
        print_prompt=args.print_prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query priors from the LM')
    parser.add_argument('--query-config', type=str,
                        help='Config specifying form of LM query')
    parser.add_argument('--gpt3-cache-fp', type=str, default=(pathlib.Path(__file__).parent.resolve() / "gpt3_cache"),
                        help='Filepath to store a (local) cache of GPT3 query results')
    parser.add_argument('--output-save-path', type=str,
                        help='Specify filepath to save LM logits')
    parser.add_argument('--print-prompt', action='store_true', default=False,
                        help='Whether or not to print each prompt')
    args = parser.parse_args()
    main(args)