"""
从训练好的模型中采样
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # 'resume'（从out_dir恢复）或gpt2变体（例如'gpt2-xl'）
out_dir = 'out' # 如果init_from不是'resume'则忽略
start = "\n" # 或"<|endoftext|>"等。也可以指定文件，用法："FILE:prompt.txt"
num_samples = 10 # 要生成的样本数量
max_new_tokens = 500 # 每个样本中生成的令牌数量
temperature = 0.8 # 1.0 = 不变，< 1.0 = 更少随机性，> 1.0 = 更多随机性，在预测中
top_k = 200 # 仅保留top_k个最可能的令牌，将其他令牌设为0概率
seed = 1337
device = 'cuda' # 示例：'cpu', 'cuda', 'cuda:0', 'cuda:1'等
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # 使用PyTorch 2.0编译模型以提高速度
exec(open('configurator.py').read()) # 从命令行或配置文件覆盖
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # 在矩阵乘法中允许tf32
torch.backends.cudnn.allow_tf32 = True # 在cudnn中允许tf32
device_type = 'cuda' if 'cuda' in device else 'cpu' # 供后续在torch.autocast中使用
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 模型
if init_from == 'resume':
    # 从特定目录中保存的模型初始化
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # 从给定的GPT-2模型初始化
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # 需要PyTorch 2.0（可选）

# 查找数据集文件夹中的meta pickle文件（如果可用）
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # 旧检查点可能没有这些...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"从{meta_path}加载meta...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO 希望使这个对任意编码器/解码器方案更通用
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # 好的，让我们默认假设gpt-2编码
    print("未找到meta.pkl，假设使用GPT-2编码...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# 编码提示的开始部分
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# 运行生成
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
