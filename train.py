from transformers import TrainingArguments, Trainer
from Dataset.dataset import VLADataset
from Models.model import VLAModel, VLAProcessor
import torch
import argparse

base_prompt = '''<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant assisting in task specified by user based on the Image provided in context<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nBased on plan: {user_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<|eot_id|><|end_of_text|>'''

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, help='path to your model')
parser.add_argument('--per_device_batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--output_dir', type=str, default='./results')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--torch_compile', type=bool, default=False)
parser.add_argument('--save_strategy', type=str, default='no')
parser.add_argument('--report_to', type=str, default='wandb')
parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
parser.add_argument('--warmup_ratio', type=float, default=0.06)
parser.add_argument('--logging_steps', type=int, default=100)
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--save_file_name', type='str', default='model.pt')

args = parser.parse_args()

def collate_fxn(data):
    image_batch = []
    input_batch = []
    action_batch = []
    for item in data:
        image_batch.append(processor(item['images'])['image_features'])
        input_batch.append(base_prompt.format(user_question=item['text']))
        action_batch.append(item['action'])
    image_batch = torch.stack(image_batch)
    tokens_processed = processor(images = None, text = input_batch)
    input_ids = tokens_processed['input_ids']
    attention_mask = tokens_processed['attention_mask']
    attention_sum = attention_mask.sum(axis=-1)
    adding_pos = torch.max(attention_sum) - attention_sum + 2
    ls = []
    attention_ls = []
    for i in range(len(input_ids)):
        ls.append(torch.concat([input_ids[i, :-adding_pos[i]], torch.tensor(action_batch[i]), input_ids[i, -adding_pos[i]:]]))
        attention_ls.append(torch.concat([attention_mask[i, :-adding_pos[i]], torch.ones(len(action_batch[i])), attention_mask[i, -adding_pos[i]:]]))
    input_ids = torch.stack(ls)
    attention_mask = torch.stack(attention_ls)
    return {'input_ids': input_ids, 'images': image_batch, 'attention_mask':attention_mask}


processor = VLAProcessor()
model = VLAModel()
if args.model_path:
    model.load_state_dict(torch.load(args.model_path), weights_only=False)
if args.dataset_path:
    dataset = VLADataset()



# data_collator = DataCollatorWithPadding(processor.tokenizer)
training_args = TrainingArguments(output_dir=args.outpu_dir, learning_rate=args.learning_rate, per_device_train_batch_size=args.per_device_batch_size , num_train_epochs=args.epochs, remove_unused_columns=False, torch_compile=args.torch_compile, save_strategy=args.save_strategy, report_to=args.report_to, lr_scheduler_type=args.lr_scheduler_type, warmup_ratio = args.warmup_ratio, logging_steps=args.logging_steps)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collate_fxn)

trainer.train()

torch.save(model.state_dict(), args.save_file_name)