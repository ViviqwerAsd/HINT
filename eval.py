import os
import re
import json
import argparse
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import ray

os.environ["VLLM_USE_V1"] = "1"
from math_verify import parse, verify, ExprExtractionConfig

DEFAULT_PLAIN_SYSTEM_PROMPT = "{QUESTION} Please reason step by step, and put the reasoning process within <think> ... </think>, and your final answer within \\boxed{{}}. \n<think>"

MATH_SYSTEM_PROMPT = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\n\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines, and put your final answer within \\boxed{}."

DEFAULT_CHAT_SYSTEM_PROMPT = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
NON_MATH_SYSTEM_PROMPT = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."


SAMPLING_PARAMS_DICT = dict(
    temperature=0,
    max_tokens=32000,
    repetition_penalty=1,
)

DATA_ROOT = os.environ.get("HINT_DATA_ROOT", "./data")


TASK_PATHS = {
    "math500": os.path.join(DATA_ROOT, "MATH-500", "test.jsonl"),
    "aime24": os.path.join(DATA_ROOT, "aime24", "aime24.jsonl"),
    "aime25": os.path.join(DATA_ROOT, "aime25", "aime25.jsonl"),
    "amc": os.path.join(DATA_ROOT, "AMC", "amc.jsonl"),
    "minerva": os.path.join(DATA_ROOT, "minerva", "minerva.jsonl"),
    "olympiad": os.path.join(DATA_ROOT, "olympiad", "olympiad.jsonl"),
    "openr1": os.path.join(DATA_ROOT, "openr1", "openr1.jsonl"),
    "gpqa_d": os.path.join(DATA_ROOT, "GPQA-D", "gpqa.jsonl"),
    "arc_c": os.path.join(DATA_ROOT, "ARC_c", "arc_c.jsonl"),
    "mmlu_pro": os.path.join(DATA_ROOT, "mmlu_pro", "mmlu_pro.jsonl"),
    "mmlu_pro_1k": os.path.join(DATA_ROOT, "mmlu_pro", "mmlu_pro-1k.jsonl"),
}
TASK_CONFIG: Dict[str, Any] = {
    "default": {
        "sampling": {
            "temperature": 0.6,
            "max_tokens": 32000,
        },
        "repeat": 1,
    },
    "overrides": {
        "math500": {"sampling": {"temperature": 0.6, "max_tokens": 32000}, "repeat": 1},
        "aime24": {"sampling": {"temperature": 0.6, "max_tokens": 32000}, "repeat": 32},
        "aime25": {"sampling": {"temperature": 0.6, "max_tokens": 32000}, "repeat": 1},
        "amc": {"sampling": {"temperature": 0.6, "max_tokens": 32000}, "repeat": 1},
        "mmlu_pro": {
            "sampling": {
                "temperature": 0.6,
                "max_tokens": 16000,
            },
            "repeat": 1,
        },
        "mmlu_pro_1k": {
            "sampling": {
                "temperature": 0.6,
                "max_tokens": 32000,
            },
            "repeat": 1,
        },
        "arc_c": {
            "sampling": {
                "temperature": 0.6,
                "max_tokens": 32000,
            },
            "repeat": 1,
        },
        "gpqa_d": {
            "sampling": {
                "temperature": 0.6,
                "max_tokens": 32000,
            },
            "repeat": 1,
        },
    },
}


def load_dataset(path: str) -> Tuple[List[str], List[str]]:
    questions, ground_truths = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            questions.append(item.get("problem") or item.get("question"))
            ground_truths.append(item.get("answer") or item.get("target"))
    return questions, ground_truths


def reward_correct(ground_truth: str, answer: str) -> bool:
    pattern = r"\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}"
    boxed_matches_ans = re.findall(pattern, answer)
    if not boxed_matches_ans:
        return False

    boxed_ans = "\\boxed{" + boxed_matches_ans[-1] + "}"
    boxed_gt = "\\boxed{" + ground_truth + "}"
    if boxed_ans == boxed_gt:
        return True

    try:
        ans_expr = parse(boxed_ans)
        gt_expr = parse(boxed_gt)
        return verify(ans_expr, gt_expr)
    except Exception as e:
        print("Parse/verify error:", e)
        return False


import math


def _pass_at_k_from_counts(n: int, c: int, k: int) -> float:
    if k > n or n <= 0:
        return float("nan")
    if c == 0:
        return 0.0

    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def resolve_prompts_for_task(task: str) -> Tuple[str, str]:
    math_tasks = {"math500", "aime24", "aime25", "amc", "minerva", "olympiad", "openr1"}
    if task in math_tasks:
        return MATH_SYSTEM_PROMPT, DEFAULT_PLAIN_SYSTEM_PROMPT
    else:
        return NON_MATH_SYSTEM_PROMPT, DEFAULT_PLAIN_SYSTEM_PROMPT


def split_indices(n: int, k: int) -> List[List[int]]:
    if k <= 0:
        return [list(range(n))]
    k = min(k, n) if n > 0 else k
    q, r = divmod(n, k)
    parts, start = [], 0
    for i in range(k):
        end = start + q + (1 if i < r else 0)
        parts.append(list(range(start, end)))
        start = end
    return parts


def resolve_task_cfg(
    task: str, base_sampling: Dict[str, Any], task_config: Dict[str, Any]
) -> Tuple[Dict[str, Any], int]:
    default = task_config.get("default", {})
    overrides = (task_config.get("overrides", {}) or {}).get(task, {})

    sampling = dict(base_sampling)
    sampling.update(default.get("sampling", {}))
    sampling.update(overrides.get("sampling", {}))

    repeat = overrides.get("repeat", default.get("repeat", 1))
    return sampling, int(repeat)


@ray.remote(num_gpus=1)
class VLLMWorker:
    def __init__(
        self,
        model_path: str,
        sampling_params: Dict[str, Any],
        use_chat_template: bool,
        chat_system_prompt: str,
        plain_system_prompt: str,
        enforce_eager: bool = True,
        dtype: str = "bfloat16",
    ):

        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.use_chat_template = use_chat_template
        self.chat_system_prompt = chat_system_prompt
        self.plain_system_prompt = plain_system_prompt

        self.sampling_params = SamplingParams(**sampling_params)
        self.llm = LLM(
            model=model_path,
            dtype=dtype,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=0.95,
            enable_chunked_prefill=False,
        )

        self.tokenizer = None
        if self.use_chat_template:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

    def _build_inputs(self, qs: List[str]) -> List[str]:
        if not self.use_chat_template:

            return [self.plain_system_prompt.format(QUESTION=q) for q in qs]

        tip_text = []
        for x in qs:
            s = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.chat_system_prompt},
                    {"role": "user", "content": x},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            tip_text.append(s)
        return tip_text

    def set_sampling_params(self, sampling_params: Dict[str, Any]):
        from vllm import SamplingParams

        self.sampling_params = SamplingParams(**sampling_params)

    def set_prompts(self, chat_system_prompt: str, plain_system_prompt: str):
        self.chat_system_prompt = chat_system_prompt
        self.plain_system_prompt = plain_system_prompt

    def generate(self, qs: List[str]) -> Tuple[List[str], List[List[int]]]:
        tip_text = self._build_inputs(qs)
        voutputs = self.llm.generate(tip_text, self.sampling_params, use_tqdm=False)

        answers, ans_token_ids = [], []
        for v in voutputs:
            for z in v.outputs:
                answers.append(z.text)
                ans_token_ids.append(z.token_ids)
        return answers, ans_token_ids

    def generate_shard(
        self, qs: List[str], inner_bs: int = 100
    ) -> Tuple[List[str], List[List[int]]]:
        all_answers: List[str] = []
        all_token_ids: List[List[int]] = []
        for i in range(0, len(qs), inner_bs):
            sub_qs = qs[i : i + inner_bs]
            sub_ans, sub_ids = self.generate(sub_qs)
            all_answers.extend(sub_ans)
            all_token_ids.extend(sub_ids)
        return all_answers, all_token_ids


def run_single_task_parallel(
    task_name: str,
    path: str,
    inner_batch_size: int,
    workers: List[ray.actor.ActorHandle],
    repeat: int = 1,
    passk_k: int | None = None,
    output_file=None,
):
    print(f"\n🚀 Running Task: {task_name }")
    questions, gts = load_dataset(path)

    n0 = len(questions)
    if repeat > 1:
        questions = questions * repeat
        gts = gts * repeat

    k = max(1, min(len(workers), len(questions)))
    parts = split_indices(len(questions), k)

    pendings: List[Tuple[List[int], ray.ObjectRef]] = []
    for w, idxs in zip(workers[:k], parts):
        if not idxs:
            continue
        shard_qs = [questions[i] for i in idxs]
        ref = w.generate_shard.remote(shard_qs, inner_bs=inner_batch_size)
        pendings.append((idxs, ref))

    preds: Dict[int, str] = {}
    all_refs = [ref for _, ref in pendings]
    idx_bags = [idxs for idxs, _ in pendings]

    pbar = tqdm(total=len(all_refs), desc=f"{task_name } collecting (by worker)")
    remaining = list(range(len(all_refs)))
    ref_map = dict(enumerate(all_refs))

    inv_map = {ref: i for i, ref in ref_map.items()}

    ready_set = set()
    while ref_map:
        ready, not_ready = ray.wait(list(ref_map.values()), num_returns=1)
        for r in ready:
            i = inv_map[r]
            answers, _ = ray.get(r)
            for gi, ans in zip(idx_bags[i], answers):
                preds[gi] = ans

            pbar.update(1)
            del inv_map[r]
            del ref_map[i]
    pbar.close()

    results = []
    correct = total = 0
    per_item_total = [0] * n0
    per_item_correct = [0] * n0

    for i, (q, gt) in enumerate(zip(questions, gts)):
        pred = preds[i]
        is_correct = reward_correct(gt, pred)
        result_item = {
            "task": task_name,
            "question": q,
            "ground_truth": gt,
            "prediction": pred,
            "is_correct": is_correct,
        }
        results.append(result_item)

        if output_file is not None:
            output_file.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            output_file.flush()

        correct += int(is_correct)
        total += 1

        orig_idx = i % n0
        per_item_total[orig_idx] += 1
        per_item_correct[orig_idx] += int(is_correct)

    acc = correct / total if total else 0.0
    print(f"✅ Task [{task_name }] Accuracy: {correct }/{total } = {acc :.2%}")

    task_passk = None
    if passk_k is not None:
        vals = []
        for n_i, c_i in zip(per_item_total, per_item_correct):
            if n_i >= passk_k:
                vals.append(_pass_at_k_from_counts(n_i, c_i, passk_k))
        task_passk = (sum(vals) / len(vals)) if vals else float("nan")
        msg = f"{task_passk :.2%}" if not math.isnan(task_passk) else "N/A"
        print(f"📊 Task [{task_name }] pass@{passk_k }: {msg }")

    return results, correct, total, task_passk


def run_tasks_parallel(
    MODEL_PATH,
    tasks: List[str],
    batch_size: int = 200,
    output_path: str = f"./multi_task_results.jsonl",
    num_workers: int = 0,
    use_chat_template: bool = False,
    chat_system_prompt: str = DEFAULT_CHAT_SYSTEM_PROMPT,
    plain_system_prompt: str = DEFAULT_PLAIN_SYSTEM_PROMPT,
    enforce_eager: bool = True,
    dtype: str = "bfloat16",
    task_config_path: str = None,
):
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=True)

    task_cfg = dict(TASK_CONFIG)
    if task_config_path:
        with open(task_config_path, "r", encoding="utf-8") as f:
            external = json.load(f)

        for k, v in external.items():
            task_cfg[k] = v

    cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
    if cluster_gpus == 0 and num_workers == 0:
        raise RuntimeError(
            "No GPU resources visible to Ray. Please start Ray with GPUs."
        )

    if num_workers <= 0:
        num_workers = max(1, cluster_gpus)

    print(f"[Ray] GPUs visible: {cluster_gpus }, using workers: {num_workers }")
    print(f"[Prompt] use_chat_template={use_chat_template }")

    workers = [
        VLLMWorker.remote(
            MODEL_PATH,
            SAMPLING_PARAMS_DICT,
            use_chat_template,
            chat_system_prompt,
            plain_system_prompt,
            enforce_eager=enforce_eager,
            dtype=dtype,
        )
        for _ in range(num_workers)
    ]

    all_results = []
    total_correct = total_total = 0

    with open(output_path, "w", encoding="utf-8") as f:
        base_sampling = dict(SAMPLING_PARAMS_DICT)
        for task in tasks:
            path = TASK_PATHS.get(task)
            if path is None:
                print(f"[Warning] Unknown task: {task }")
                continue

            chat_system_prompt, plain_system_prompt = resolve_prompts_for_task(task)

            sampling_params, repeat = resolve_task_cfg(task, base_sampling, task_cfg)
            ray.get([w.set_sampling_params.remote(sampling_params) for w in workers])
            ray.get(
                [
                    w.set_prompts.remote(chat_system_prompt, plain_system_prompt)
                    for w in workers
                ]
            )

            passk_k = repeat if repeat > 1 else None

            task_results, c, t, task_passk = run_single_task_parallel(
                task_name=task,
                path=path,
                inner_batch_size=batch_size,
                workers=workers,
                repeat=repeat,
                passk_k=passk_k,
                output_file=f,
            )
            all_results.extend(task_results)
            total_correct += c
            total_total += t

    print(
        f"\n🎯 Overall Accuracy: {total_correct }/{total_total } = {total_correct /total_total :.2%}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["math500", "aime24"],
        help="Tasks to evaluate (e.g., math500 aime24)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Per-worker micro-batch size."
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of Ray GPU workers to launch. 0 = use all visible GPUs.",
    )
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Use tokenizer.apply_chat_template with a system+user chat prompt.",
    )
    parser.add_argument(
        "--chat_system_prompt",
        type=str,
        default=DEFAULT_CHAT_SYSTEM_PROMPT,
        help="System prompt used when --use_chat_template is on.",
    )
    parser.add_argument(
        "--plain_system_prompt",
        type=str,
        default=DEFAULT_PLAIN_SYSTEM_PROMPT,
        help="Plain text prompt (supports {QUESTION}) when --use_chat_template is off.",
    )

    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--no_enforce_eager",
        action="store_true",
        help="If set, use enforce_eager=False for VLLM LLM init.",
    )
    parser.add_argument(
        "--task_config_path",
        type=str,
        default=None,
        help="Optional JSON file that overrides per-task sampling config.",
    )

    args = parser.parse_args()
    if args.output is None:
        model_name = os.path.basename(args.model_path.rstrip("/"))
        args.output = f"./{model_name }+{','.join (args .tasks )}.jsonl"

    run_tasks_parallel(
        args.model_path,
        args.tasks,
        args.batch_size,
        args.output,
        args.num_workers,
        use_chat_template=args.use_chat_template,
        chat_system_prompt=args.chat_system_prompt,
        plain_system_prompt=args.plain_system_prompt,
        enforce_eager=not args.no_enforce_eager,
        dtype=args.dtype,
        task_config_path=args.task_config_path,
    )


if __name__ == "__main__":
    main()
