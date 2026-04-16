import argparse
import json
import os
import random
import re
from pathlib import Path

from math_verify import parse, verify

from lsrl_hint import LSRL
from ref_server import RefServer

os.environ["OMP_NUM_THREADS"] = "32"

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. A conversation takes place between the "
    "User and the Assistant. The User asks a question, and the Assistant "
    "solves it. Please help me solve this question. Wrap only the final "
    "answer in \\boxed{}."
)


def correct_fn(answer, item):
    ground_truth_str = item["A"]
    pattern = r"\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}"
    boxed_content_ans = re.findall(pattern, answer)
    if not boxed_content_ans:
        return 0.0

    final_ans_expr = "\\boxed{" + boxed_content_ans[-1] + "}"
    final_gt_expr = "\\boxed{" + ground_truth_str + "}"

    if final_ans_expr == final_gt_expr:
        return 1.0

    try:
        parsed_ans = parse(final_ans_expr)
        parsed_gt = parse(final_gt_expr)
        return 1.0 if verify(parsed_ans, parsed_gt) else 0.0
    except Exception:
        return 0.0


def make_prompt_fn(self, item):
    return self.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["Q"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def make_hint_fn(self, item):
    return self.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    item["Q"]
                    + "\nHere are some key pieces of information to assist your "
                    + "reasoning without revealing the solution directly:\n"
                    + item["H"]
                ),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_gen_devices(raw):
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def load_training_data(data_file_path):
    records = []
    with open(data_file_path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            item = json.loads(line)
            question = item.get("question") or item.get("problem") or item.get("Q")
            answer = item.get("answer") or item.get("A")
            hint = item.get("abstract_hint") or item.get("H")
            if not all([question, answer, hint]):
                raise ValueError(
                    f"Line {line_id} in {data_file_path} is missing one of "
                    "'question/problem', 'answer', or 'abstract_hint'."
                )
            records.append({"Q": question, "A": answer, "H": hint})
    return records


all_corrs = []


def rollout_monitor(samples):
    global all_corrs
    corrs = [1 if rewards.get("correct_fn", -1) > 0 else 0 for rewards in samples["rewards"]]
    all_corrs.extend(corrs)
    if len(all_corrs) > 1000:
        all_corrs = all_corrs[-1000:]
    acc = sum(all_corrs) / len(all_corrs)
    print(f"[ROLL] rollout monitor: acc: {acc:.2f}")


def run_ref_server(args):
    RefServer(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        force_cpu_offload=args.force_cpu_offload,
        nlayers_keep_in_gpu=args.nlayers_keep_in_gpu,
    ).start()


def run_gsm8k_test(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    qas = [
        {"Q": question, "A": answer.split("####")[-1].strip()}
        for question, answer in zip(dataset["question"], dataset["answer"])
    ]
    if args.limit is not None:
        qas = qas[: args.limit]

    vllm_gen = LLM(
        model=args.model_path,
        enable_chunked_prefill=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.no_enforce_eager,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    test_obj = lambda: None
    test_obj.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompts = [make_prompt_fn(test_obj, x) for x in qas]
    voutputs = vllm_gen.generate(prompts, sampling_params, use_tqdm=True)

    corrs = [int(correct_fn(output.outputs[0].text, item) > 0) for output, item in zip(voutputs, qas)]
    wrongs = [idx for idx, corr in enumerate(corrs) if corr == 0]
    acc = sum(corrs) / len(corrs) if corrs else 0.0
    print(f"Accuracy: {acc:.2%}")

    if args.results_file:
        with open(args.results_file, "w", encoding="utf-8") as f:
            for idx in wrongs:
                text = voutputs[idx].outputs[0].text
                item = qas[idx]
                f.write(f'Q: {item["Q"]}\nA: {item["A"]}\nModel: {text}\n\n')


def run_training(args):
    qas = load_training_data(args.data_file)
    random.seed(args.shuffle_seed)
    random.shuffle(qas)
    os.makedirs(args.save_path, exist_ok=True)

    lsrl = LSRL(
        args.model_path,
        epochs=args.epochs,
        train_data=qas,
        rollout_num=args.rollout_num,
        train_batch_size=args.train_batch_size,
        gen_batch_size=args.gen_batch_size,
        gen_max_tokens=args.gen_max_tokens,
        gen_update_steps=args.gen_update_steps,
        trainer=args.trainer,
        gen_temperature=args.gen_temperature,
        gen_device=parse_gen_devices(args.gen_devices),
        ref_server=args.ref_url,
        beta=args.beta,
        lr=args.lr,
        accum_steps=args.accum_steps,
        genlog_filename=args.genlog_filename,
        save_steps=args.save_steps,
        skip_zero_groups=args.skip_zero_groups,
        save_path=args.save_path,
        swanlab_project=args.swanlab_project,
        swanlab_experiment_name=args.swanlab_experiment_name,
        affinity_lambda=args.affinity_lambda,
    )
    lsrl.set_hook("after_rollout", rollout_monitor)
    lsrl.add_reward(correct_fn)
    lsrl.set_policy_prompt_fn(make_prompt_fn)
    lsrl.set_rollout_prompt_fn(make_prompt_fn)
    lsrl.set_hint_prompt_fn(make_hint_fn)
    lsrl.train()


def build_parser():
    parser = argparse.ArgumentParser(description="HINT training and evaluation entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ref_parser = subparsers.add_parser("serve-ref", help="Start the reference log-probability server.")
    ref_parser.add_argument("--model-path", type=str, required=True)
    ref_parser.add_argument("--host", type=str, default="0.0.0.0")
    ref_parser.add_argument("--port", type=int, default=59888)
    ref_parser.add_argument("--force-cpu-offload", action="store_true")
    ref_parser.add_argument("--nlayers-keep-in-gpu", type=int, default=12)
    ref_parser.set_defaults(func=run_ref_server)

    train_parser = subparsers.add_parser("train", help="Train HINT with Meta-Hints.")
    train_parser.add_argument("--model-path", type=str, required=True)
    train_parser.add_argument("--data-file", type=str, default="./data/dapo-selected-10k.jsonl")
    train_parser.add_argument("--save-path", type=str, default="./outputs/hint")
    train_parser.add_argument("--ref-url", type=str, default="http://127.0.0.1:59888")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--rollout-num", type=int, default=8)
    train_parser.add_argument("--train-batch-size", type=int, default=8)
    train_parser.add_argument("--gen-batch-size", type=int, default=32)
    train_parser.add_argument("--gen-update-steps", type=int, default=128)
    train_parser.add_argument("--save-steps", type=int, default=2560)
    train_parser.add_argument("--gen-max-tokens", type=int, default=3000)
    train_parser.add_argument("--gen-temperature", type=float, default=0.9)
    train_parser.add_argument("--gen-devices", type=str, default="0")
    train_parser.add_argument("--beta", type=float, default=0.0)
    train_parser.add_argument("--lr", type=float, default=1e-6)
    train_parser.add_argument("--accum-steps", type=int, default=64)
    train_parser.add_argument("--affinity-lambda", type=float, default=1.0)
    train_parser.add_argument("--trainer", type=str, default="LSCPU")
    train_parser.add_argument("--shuffle-seed", type=int, default=42)
    train_parser.add_argument("--skip-zero-groups", action="store_true")
    train_parser.add_argument("--genlog-filename", type=str, default="")
    train_parser.add_argument("--swanlab-project", type=str, default=None)
    train_parser.add_argument("--swanlab-experiment-name", type=str, default="")
    train_parser.set_defaults(func=run_training)

    test_parser = subparsers.add_parser("gsm8k-test", help="Run quick GSM8K evaluation for a checkpoint.")
    test_parser.add_argument("--model-path", type=str, required=True)
    test_parser.add_argument("--limit", type=int, default=None)
    test_parser.add_argument("--temperature", type=float, default=0.001)
    test_parser.add_argument("--max-tokens", type=int, default=1024)
    test_parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    test_parser.add_argument("--no-enforce-eager", action="store_true")
    test_parser.add_argument("--results-file", type=str, default="gsm8k_results.txt")
    test_parser.set_defaults(func=run_gsm8k_test)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if getattr(args, "save_path", None):
        args.save_path = str(Path(args.save_path))
    args.func(args)
