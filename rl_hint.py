import random, os, sys, re, time, requests, json
from math_verify import parse, verify
os.environ['OMP_NUM_THREADS'] = '32'

'''
Use Deepseek format, the testset accuracy is about 0.90
Use this scrip, and do not use format_fn (like verl), the testset accuracy is 0.92 (the first saving)

Note: Step counts vary across frameworks due to different definitions. 
Use training time for fair comparison.
'''

def correct_fn(answer, item):
    ground_truth_str = item['A']
    pattern = r'\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}'
    boxed_content_ans = re.findall(pattern, answer)

    if not boxed_content_ans:
        return 0
    final_ans_expr = "\\boxed{" + boxed_content_ans[-1] + "}"
    final_gt_expr = "\\boxed{" + ground_truth_str + "}"

    if final_ans_expr == final_gt_expr:
        return 1.0
    try:
        parsed_ans = parse(final_ans_expr)
        parsed_gt = parse(final_gt_expr)
        is_correct = verify(parsed_ans, parsed_gt)
        return 1.0 if is_correct else 0
    except Exception as e:
        return 0

system_prompt = 'You are a helpful AI assistant. A conversation takes place between the User and the Assistant. The User asks a question, and the Assistant solves it. Please help me solve this question. Wrap only the final answer in \\boxed{}.'
def make_prompt_fn(self, item):
    return self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item['Q']}], 
            tokenize=False, add_generation_prompt=True)

def make_hint_fn(self, item):
    return self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item['Q'] + '\n' + "Here are some key information provided to assist you in solving the problem:\n" + item['H']}], 
            tokenize=False, add_generation_prompt=True)

from lsrl_hint import LSRL
from ref_server import RefServer
model_path = "/data2/Qwen/Qwen2.5-7B/"
if 'ref' in sys.argv:
    RefServer(model_path, port=59888).start()
    sys.exit(0)
       
from math_verify import parse, verify, ExprExtractionConfig

if 'test' in sys.argv:
    number = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    print(f'Testing model at step {number}...')
    model_path = f'./step_{number}'

    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]
        
    from vllm import LLM, SamplingParams
    vllm_gen = LLM(model=model_path, enable_chunked_prefill=True, 
                gpu_memory_utilization=0.7, enforce_eager=True)
    sampling_params = SamplingParams(n=1, temperature=0.001, max_tokens=1024)

    from transformers import AutoTokenizer
    test_obj = lambda: None
    test_obj.tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = [make_prompt_fn(test_obj, x) for x in QAs]
    voutputs = vllm_gen.generate(prompts, sampling_params, use_tqdm=True)

    corrs = [1 * (correct_fn(x.outputs[0].text, item) > 0) for x, item in zip(voutputs, QAs)]
    print(corrs)
    wrongs = [k for k,v in enumerate(corrs) if v == 0]
    print('Wrong QA:', wrongs)
    acc = sum(corrs) / len(corrs)
    print(f'Accuracy: {acc:.2f}')

    with open('gsm8k_results.txt', 'w') as f:
        for k in wrongs:
            text = voutputs[k].outputs[0].text
            item = QAs[k]
            f.write(f'Q: {item["Q"]}\nA: {item["A"]}\nVLLM: {text}\n\n')
    sys.exit()


all_corrs = []
def rollout_monitor(samples):
    global all_corrs
    corrs = []
    for rewards in samples['rewards']: corrs.append(rewards.get('correct_fn', -1))
    corrs = [1 if x > 0 else 0 for x in corrs]
    all_corrs.extend(corrs)
    if len(all_corrs) > 1000: all_corrs = all_corrs[-1000:]  
    acc = sum(all_corrs) / len(all_corrs)
    print(f'[ROLL] rollout monitor: acc: {acc:.2f}')

if __name__ == '__main__':
    data_file_path = "/dataset/training_dataset/dapo-selected-30k.jsonl"
    QAs = []
    with open(data_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            QAs.append({
                'Q': item['question'],
                'A': item['answer'],
                'H': item['abstract_hint']
            })


    random.seed(42)
    random.shuffle(QAs)

    lsrl = LSRL(model_path, epochs=10, train_data=QAs, rollout_num=8, 
                train_batch_size=8, gen_batch_size=32, gen_max_tokens=3000,
                gen_update_steps=128, trainer='LSCPU', gen_temperature=0.9,
                gen_device=[4,5], ref_server="http://127.0.0.1:59888", beta=0,
                lr=1e-6, accum_steps=64, genlog_filename='',
                save_steps=2560, skip_zero_groups=False, 
                save_path = "", swanlab_project = "hint", swanlab_experiment_name = "")
    

    
    lsrl.set_hook('after_rollout', rollout_monitor)
    lsrl.add_reward(correct_fn)
    lsrl.set_policy_prompt_fn(make_prompt_fn)
    lsrl.set_rollout_prompt_fn(make_prompt_fn)
    lsrl.set_hint_prompt_fn(make_hint_fn)
    lsrl.train()