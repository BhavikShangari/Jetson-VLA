import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM
import os
import timm
from PIL import Image
import torch
import pandas as pd
import numpy as np

class Adapter(nn.Module):
    def __init__(self, input_dim, llm_dim=2048):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 4*input_dim, bias=True)
        self.linear2 = nn.Linear(4*input_dim, llm_dim, bias=True)
        self.linear3 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()

    def forward(self, image):
        image = self.linear1(image)
        image = self.gelu1(image)
        image = self.linear2(image)
        image = self.gelu2(image)
        image = self.linear3(image)
        return image

class VLAProcessor():
    def __init__(self, tokenizer_id = 'meta-llama/Llama-3.2-1B-Instruct', image_processor_id = 'vit_base_patch16_224.dino'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.tokenizer.pad_token = '<|finetune_right_pad_id|>'
        dino_data_config = timm.data.resolve_model_data_config(image_processor_id)
        self.image_processor = timm.data.create_transform(**dino_data_config, is_training=False)
        
    def __call__(self, images=None, text=None):
        ''''''
        inputs = {}
        if text is None and images is None:
            raise ValueError('Image and text cannot both be None')
        if text is not None:
            inputs = self.tokenizer(text, return_tensors='pt', max_length=250, truncation=True, padding=True)
            # inputs.update({'labels':self.tokenizer(labels, return_tensors='pt', max_length=250, truncation=True, padding=True)})
        
        # Process images (if any)
        if images is not None:
            image_inputs = self.image_processor(images)
            inputs.update({'image_features':image_inputs})
        
        return inputs
    
processor = VQAProcessor()
    

class VLAModel(nn.Module):
    def __init__(self,train_vision_backbone=False, train_adapter = True, train_llm_backbone = True):
        super().__init__()

        self.model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B-Instruct', trust_remote_code=False)
        for mat in self.model.parameters():
            mat.requires_grad = train_llm_backbone

        self.dino_model = timm.create_model('vit_base_patch16_224.dino', pretrained=True, num_classes=0)
        for mat in self.dino_model.parameters():
            mat.requires_grad=train_vision_backbone

        self.siglip_model = timm.create_model('vit_base_patch16_siglip_224', pretrained=True, num_classes=0)
        for mat in self.siglip_model.parameters():
            mat.requires_grad=train_vision_backbone

        self.adapter = Adapter(2*768, llm_dim=2048)
        for mat in self.adapter.parameters():
            mat.requires_grad=train_adapter


    def forward(self, input_ids=None, images=None, attention_mask=None):
        if input_ids is None or images is None:
            raise ValueError("prompt and image cannot be None")
        
        dino_embeddings = self.dino_model.forward_features(images)
        siglip_embeddings = self.siglip_model.forward_features(images)

        image_embeddings = torch.concat([dino_embeddings[:,:-1,:], siglip_embeddings], dim=2)   # one extra (dino: 197, siglip: 196 )in case of dino, removing that
        image_embeddings = self.adapter(image_embeddings)

        batch, seq_len, _ = image_embeddings.shape
        image_pad = torch.full((batch, seq_len), 128004, device='cuda')
        labels = torch.concat([input_ids[:, :31], image_pad, input_ids[:, 31:]], dim=1)
        
        x = self.model(input_ids, image_features = image_embeddings, labels=labels, attention_mask=attention_mask, tokenizer = processor.tokenizer)
        
        return x

    def generate(self, input_ids=None, images=None, gen_config=None, tokenizer=None):
        if input_ids is None or images is None:
            raise ValueError("prompt and image cannot be None")
        
        dino_embeddings = self.dino_model.forward_features(images)
        siglip_embeddings = self.siglip_model.forward_features(images)

        image_embeddings = torch.concat([dino_embeddings[:,:-1,:], siglip_embeddings], dim=2)   # one extra (dino: 197, siglip: 196 )in case of dino, removing that
        image_embeddings = self.adapter(image_embeddings)

        return self.model.generate(inputs=input_ids, generation_config=gen_config, image_features=image_embeddings, tokenizer=tokenizer)
