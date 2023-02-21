import openai
import os
import torch
from retry import retry
import json
from tqdm import tqdm


openai.api_key = os.getenv('OPENAI_API_KEY')


@retry(openai.error.RateLimitError, delay=3, backoff=1.5, max_delay=30, tries=10)
def openai_completion_query(**kwargs):
    if kwargs['engine'].startswith("code"):
        time.sleep(3)
        print(f"query time: {time.time()}")
    return openai.Completion.create(**kwargs)


def gpt3_score_prompt(engine, input_prefix, classes, cache=None, gpt3_file=None):
    new_cache_results = {}
    class_scores = []
    # optimal_class = None
    if engine.startswith("random"):
        return [1./len(classes) for _ in classes], new_cache_results
    for cl in classes:
        input_str = input_prefix + cl
        if cache and input_str in cache:
            result = cache[input_str]
        else:
            result = openai_completion_query(
                engine=engine,
                prompt=input_str,
                max_tokens=0,
                logprobs=0,
                echo=True,
            )
            if engine.startswith("code"):
                print("Got result")
            # gpt3_output = result["choices"][0]
            new_cache_results[input_str] = result
            cache[input_str] = result
            if gpt3_file:
                save_gpt3_result(gpt3_file, {input_str: result})
        for token_position in range(len(result['choices'][0]['logprobs']['tokens'])):
            if ''.join(
                result['choices'][0]['logprobs']['tokens'][:token_position]
            ).strip() == input_prefix.strip():
                break
        score = sum(result['choices'][0]['logprobs']['token_logprobs'][token_position:]) #/ len(result['choices'][0]['logprobs']['token_logprobs'][token_position:])
        class_scores.append(score)
    return class_scores, new_cache_results


def gpt3_generate(engine, input_prefix, cache=None, temperature=0.7, n_samples=1, gpt3_file=None, other_stop_tokens=[]):
    new_cache_results = {}
    class_scores = []
    input_str = input_prefix.rstrip()
    if temperature == 0 and cache and input_str in cache:
        result = cache[input_str]
    else:
        result = openai_completion_query(
            engine=engine,
            prompt=input_str,
            max_tokens=5,  # generate room name OR item
            logprobs=5,
            echo=True,
            stop=["\n", "<|endoftext|>", *other_stop_tokens],
            temperature=temperature,
            n=n_samples,
        )
        # gpt3_output = result["choices"][0]
        new_cache_results[input_str] = result
        cache[input_str] = result
    if temperature == 0 and gpt3_file:
        save_gpt3_result(gpt3_file, new_cache_results)
    choice_to_score = {}
    choice_to_score2 = {}
    for choice in result['choices']:
        index_of_prefix_token = choice['logprobs']['text_offset'].index(len(input_str))
        # gen_score = sum(choice['logprobs']['token_logprobs'][index_of_prefix_token:])
        gen_scores = choice['logprobs']['token_logprobs'][index_of_prefix_token:]
        gen_tokens = choice['logprobs']['tokens'][index_of_prefix_token:]
        if "<|endoftext|>" in gen_tokens:
            eot_token_idx = gen_tokens.index("<|endoftext|>")
            gen_scores = gen_scores[:eot_token_idx]
            gen_tokens = gen_tokens[:eot_token_idx]
        if "\n" in gen_tokens:
            nl_token_idx = gen_tokens.index("\n")
            gen_scores = gen_scores[:nl_token_idx]
            gen_tokens = gen_tokens[:nl_token_idx]
        if len(gen_scores) > 0: #breakpoint()
            gen_score = sum(gen_scores) / len(gen_scores)
            generation = ''.join(gen_tokens)
            choice_to_score[generation] = gen_score
            choice_to_score2[generation] = sum(gen_scores)
    return result['choices'][0]['text'][len(input_prefix):], choice_to_score, new_cache_results


def gpt3_constrained_generation(engine, input_prefix, classes, cache=None, gpt3_file=None):
    """
    First generate greedily (for expenses), otherwise score every class individually
    """
    greedy_generation, _, _ = gpt3_generate(engine, input_prefix, cache=cache, temperature=0, n_samples=1, gpt3_file=gpt3_file, other_stop_tokens=["."])
    if greedy_generation.strip().rstrip('.') in classes:
        return classes.index(greedy_generation.strip().rstrip('.'))
    else:
        scores, _ = gpt3_score_prompt(
            engine=engine,
            input_prefix=input_prefix,
            classes=classes,
            cache=cache,
            gpt3_file=gpt3_file,
        )
        return torch.tensor(scores).argmax()


def save_gpt3_result(gpt3_file, new_results):
    with open(gpt3_file, "a") as wf:
        for prompt in new_results:
            wf.write(json.dumps({"prompt": prompt, "result": new_results[prompt]}) + "\n")


def load_gpt3_cache(gpt3_file):
    os.makedirs(os.path.split(gpt3_file)[0], exist_ok=True)
    cache = {}
    if os.path.exists(gpt3_file):
        with open(gpt3_file) as f:
            for line in tqdm(f, desc="Loading GPT3 cache"):
                line = json.loads(line)
                cache[line['prompt']] = line['result']
    return cache