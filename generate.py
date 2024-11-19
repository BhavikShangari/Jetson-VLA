from Models.model import VLAModel, VLAProcessor
import torch
import argparse
from PIL import Image
from transformers import GenerationConfig
import numpy as np

splits = np.linspace(-1,1, 256)
mid = (splits[:-1] + splits[1:]) / 2

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--image_path', type=str, required=True)
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--device', type=str, required=True, default='cpu')

args = parser.parse_args()

model = VLAModel()
model.load_state_dict(torch.load(args.model_path))
model.to(args.device)

processor = VLAProcessor()

splits = np.linspace(-1,1, 256)
mid = (splits[:-1] + splits[1:]) / 2
base= processor.tokenizer.vocab_size - 256-1

base_prompt = '''<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant assisting in task specified by user based on the Image provided in context<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nBased on Image plan: {user_prompt}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

base= processor.tokenizer.vocab_size - 256-1
image = Image.open(args.image_path)
image.show()

gen_config = GenerationConfig(max_new_tokens=30, use_cache=False)
processed = processor(images=image, text = base_prompt.format(user_prompt=args.prompt)).to(args.device)
initial_length = len(processed['input_ids'][0, :])
generation = model.generate(input_ids=processed['input_ids'], images=processed['image_features'].unsqueeze(0), gen_config=gen_config, tokenizer=processor.tokenizer)

# print(processor.tokenizer.batch_decode(generation[:, initial_length:][0], skip_special_tokens=True))
print(mid[np.clip(generation[: ,initial_length:][0].cpu()[:-1]-base, 0, 254)])




