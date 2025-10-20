import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
from datetime import datetime
import threading
import glob
from collections import Counter
import struct
import shutil
import requests
import tempfile
import zipfile
from urllib.request import urlopen  
from PIL import Image, ImageTk      
from io import BytesIO   
from urllib.parse import urlparse
import math
import time
from typing import List, Dict, Any, Optional

def set_icon_from_url(root, url):
    # Download the image from the URL
    with urlopen(url) as u:
        raw_data = u.read()
    
    # Open the image with PIL and convert for Tkinter
    image = Image.open(BytesIO(raw_data))
    photo = ImageTk.PhotoImage(image)
    
    # Set the window icon
    root.iconphoto(False, photo)
    
    # Keep a reference to prevent garbage collection
    root.icon_ref = photo  # Store the reference as an attribute of root

# Project management classes
class ProjectManager:
    
    def __init__(self, projects_dir="projects"):
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(exist_ok=True)
        self.projects_file = self.projects_dir / "projects.json"
        self.projects = self.load_projects()
        
    def load_projects(self):
        if self.projects_file.exists():
            try:
                with open(self.projects_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_projects(self):
        with open(self.projects_file, 'w') as f:
            json.dump(self.projects, f, indent=4)
    
    def create_project(self, name, description, model_type="scratch", base_model=None):
        project_id = str(len(self.projects) + 1)
        project_dir = self.projects_dir / project_id
        project_dir.mkdir(exist_ok=True)
        
        project_data = {
            "id": project_id,
            "name": name,
            "description": description,
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
            "model_type": model_type,
            "base_model": base_model,
            "path": str(project_dir)
        }
        
        self.projects[project_id] = project_data
        self.save_projects()
        return project_id
    
    def update_project(self, project_id, updates):
        if project_id in self.projects:
            self.projects[project_id].update(updates)
            self.projects[project_id]["modified"] = datetime.now().isoformat()
            self.save_projects()
            return True
        return False
    
    def delete_project(self, project_id):
        if project_id in self.projects:
            project_path = Path(self.projects[project_id]["path"])
            if project_path.exists():
                shutil.rmtree(project_path)
            del self.projects[project_id]
            self.save_projects()
            return True
        return False
    
    def get_project(self, project_id):
        return self.projects.get(project_id, None)

class SimpleTokenizer:
    """A simple tokenizer for faster startup"""
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.eos_token_id = 1
        self.unk_token = "<unk>"
        self.unk_token_id = 2
        
        # Start with basic tokens
        self.add_token(self.pad_token)  # ID 0
        self.add_token(self.eos_token)  # ID 1
        self.add_token(self.unk_token)  # ID 2
    
    def add_token(self, token):
        if token not in self.vocab:
            self.vocab[token] = self.vocab_size
            self.inverse_vocab[self.vocab_size] = token
            self.vocab_size += 1
            return True
        return False
    
    def build_vocab_from_texts(self, texts, max_vocab_size=10000):
        """Build vocabulary from all training texts"""
        print("Building vocabulary from training data...")
        
        # Count all tokens
        token_counter = Counter()
        for text in texts:
            tokens = text.split()
            token_counter.update(tokens)
        
        # Add most frequent tokens to vocabulary
        for token, _ in token_counter.most_common(max_vocab_size - self.vocab_size):
            self.add_token(token)
        
        print(f"Vocabulary built with {self.vocab_size} tokens")
    
    def tokenize(self, text):
        # Simple word-level tokenization
        tokens = text.split()
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.unk_token_id)  # Use UNK token for out-of-vocab words
        return token_ids
    
    def encode(self, text, max_length=None, padding=False, truncation=False):
        token_ids = self.tokenize(text)
        
        if truncation and max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        if padding and max_length and len(token_ids) < max_length:
            token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
        
        return token_ids
    
    def decode(self, token_ids):
        # Remove padding tokens for cleaner output
        filtered_ids = [id for id in token_ids if id != self.pad_token_id]
        return " ".join([self.inverse_vocab.get(id, self.unk_token) for id in filtered_ids])

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        
        # Filter out empty texts
        self.texts = [text for text in texts if text.strip()]
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Ensure text is not empty
        if not text.strip():
            text = " "  # Use space for empty text
            
        token_ids = self.tokenizer.encode(
            text, 
            max_length=self.max_length,
            padding=True,
            truncation=True
        )
        
        # Convert to tensor and ensure all IDs are within valid range
        token_ids = [min(id, self.tokenizer.vocab_size - 1) for id in token_ids]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(token_ids, dtype=torch.long)
        }

# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class SimpleGPT(nn.Module):
    """A simplified GPT-like model for faster training"""
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)  # padding_idx=0 for pad token
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layer with dropout for regularization
        self.dropout = nn.Dropout(0.1)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # Ensure all token IDs are within valid range
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        # Create token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Create position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        position_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_embeds + position_embeds
        
        # Create attention mask (ignore padding tokens)
        attention_mask = (input_ids != 0).float()
        
        # Transformer with attention mask
        x = self.transformer(x, src_key_padding_mask=attention_mask == 0)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Output
        logits = self.output_layer(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Ensure labels are within valid range
            labels = torch.clamp(labels, 0, self.vocab_size - 1)
            
            # Create loss mask to ignore padding tokens
            loss_mask = (labels != 0).float()
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='none')  # ignore padding
            losses = loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
            loss = (losses * loss_mask.view(-1)).sum() / loss_mask.sum()
        
        return {'logits': logits, 'loss': loss}

# =============================================================================
# ENHANCED MODEL COMPONENTS
# =============================================================================

class EnhancedPositionalEncoding(nn.Module):
    """Enhanced positional encoding with learnable parameters"""
    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable positional embeddings
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        
        # Initialize with sinusoidal pattern
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.position_embeddings.weight.data = pe
        self.position_embeddings.weight.requires_grad = True  # Make it learnable

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), seq_len)
        pos_embeddings = self.position_embeddings(positions)
        return self.dropout(x + pos_embeddings)

class LayerNormalization(nn.Module):
    """Enhanced layer normalization with learnable parameters"""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class ResidualConnection(nn.Module):
    """Residual connection with dropout and layer norm"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttention(nn.Module):
    """Enhanced multi-head attention with causal masking"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, q, k, v, mask=None):
        batch_size, seq_len = q.size(0), q.size(1)
        
        # Linear projections
        q = self.w_q(q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply causal mask for autoregressive generation
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(output)

class FeedForward(nn.Module):
    """Enhanced feed-forward network with GELU activation"""
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
            
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()
            
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Enhanced transformer block with residual connections"""
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        
    def forward(self, x, mask=None):
        x = self.residual1(x, lambda x: self.attention(x, x, x, mask))
        x = self.residual2(x, self.feed_forward)
        return x

class EnhancedGPT(nn.Module):
    """Enhanced GPT model with multiple improvements"""
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, max_seq_len=512, 
                 dropout=0.1, activation="gelu", use_enhanced_pe=True, use_layer_norm=True):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Enhanced positional encoding
        self.use_enhanced_pe = use_enhanced_pe
        if use_enhanced_pe:
            self.pos_encoding = EnhancedPositionalEncoding(d_model, max_seq_len, dropout)
        else:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4, dropout, activation)
            for _ in range(n_layers)
        ])
        
        # Enhanced layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.final_norm = LayerNormalization(d_model)
        else:
            self.final_norm = nn.LayerNorm(d_model)
            
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Weight tying (share weights between input and output embeddings)
        self.output_layer.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNormalization):
            torch.nn.init.ones_(module.gamma)
            torch.nn.init.zeros_(module.beta)
        
    def forward(self, input_ids, labels=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Ensure all token IDs are within valid range
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Positional encoding
        if self.use_enhanced_pe:
            x = self.pos_encoding(token_embeds)
        else:
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
            pos_embeds = self.pos_embedding(positions)
            x = token_embeds + pos_embeds
            x = self.dropout(x)
        
        # Create causal attention mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # Add batch and head dimensions
        
        # Combine with padding mask if provided
        if attention_mask is not None:
            padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            mask = causal_mask | padding_mask
        else:
            mask = causal_mask
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Output logits
        logits = self.output_layer(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Ensure labels are within valid range
            labels = torch.clamp(labels, 0, self.vocab_size - 1)
            
            # Shift for autoregressive training
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Create loss mask to ignore padding tokens
            loss_mask = (shift_labels != 0).float()
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
            losses = loss_fn(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            loss = (losses * loss_mask.view(-1)).sum() / loss_mask.sum()
        
        return {'logits': logits, 'loss': loss}
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50, top_p=0.9):
        """Enhanced text generation with sampling options"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(input_ids)['logits']
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to the sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if EOS token is generated
                if next_token.item() == 1:  # EOS token
                    break
            
            return input_ids

# =============================================================================
# MODEL ENHANCEMENTS MANAGER
# =============================================================================

class ModelEnhancements:
    """Manages model enhancements and their configurations"""
    
    def __init__(self):
        self.enhancements = {
            "enhanced_architecture": {
                "name": "Enhanced Architecture",
                "description": "Use improved transformer architecture with better components",
                "enabled": True,
                "config": {
                    "use_enhanced_pe": True,
                    "use_layer_norm": True,
                    "activation": "gelu"
                }
            },
            "advanced_regularization": {
                "name": "Advanced Regularization",
                "description": "Apply multiple regularization techniques",
                "enabled": False,
                "config": {
                    "weight_decay": 0.01,
                    "gradient_clip": 1.0,
                    "label_smoothing": 0.1,
                    "dropout_rate": 0.1
                }
            },
            "learning_rate_scheduling": {
                "name": "Learning Rate Scheduling",
                "description": "Use dynamic learning rate scheduling",
                "enabled": False,
                "config": {
                    "scheduler_type": "cosine",  # cosine, linear, step
                    "warmup_steps": 1000,
                    "min_lr": 1e-6
                }
            },
            "gradient_accumulation": {
                "name": "Gradient Accumulation",
                "description": "Accumulate gradients over multiple steps",
                "enabled": False,
                "config": {
                    "accumulation_steps": 4
                }
            },
            "mixed_precision": {
                "name": "Mixed Precision Training",
                "description": "Use mixed precision for faster training (requires GPU)",
                "enabled": False,
                "config": {
                    "dtype": "float16"
                }
            },
            "early_stopping": {
                "name": "Early Stopping",
                "description": "Stop training when validation loss stops improving",
                "enabled": False,
                "config": {
                    "patience": 3,
                    "min_delta": 0.001
                }
            },
            "knowledge_distillation": {
                "name": "Knowledge Distillation",
                "description": "Use teacher-student training paradigm",
                "enabled": False,
                "config": {
                    "teacher_model": None,
                    "temperature": 2.0,
                    "alpha": 0.7
                }
            }
        }
    
    def get_enabled_enhancements(self):
        """Return list of enabled enhancements"""
        return {name: config for name, config in self.enhancements.items() if config["enabled"]}
    
    def toggle_enhancement(self, enhancement_name, enabled=None):
        """Toggle an enhancement on/off"""
        if enhancement_name in self.enhancements:
            if enabled is None:
                self.enhancements[enhancement_name]["enabled"] = not self.enhancements[enhancement_name]["enabled"]
            else:
                self.enhancements[enhancement_name]["enabled"] = enabled
            return True
        return False
    
    def update_config(self, enhancement_name, config_updates):
        """Update configuration for an enhancement"""
        if enhancement_name in self.enhancements:
            self.enhancements[enhancement_name]["config"].update(config_updates)
            return True
        return False

class EnhancedOptimizer:
    """Enhanced optimizer with multiple scheduling options"""
    
    def __init__(self, model_parameters, enhancement_config, total_steps):
        self.enhancement_config = enhancement_config
        self.total_steps = total_steps
        
        # Base optimizer with weight decay
        weight_decay = enhancement_config.get("weight_decay", 0.0)
        self.optimizer = optim.AdamW(model_parameters, weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = None
        if enhancement_config.get("scheduler_type"):
            self.setup_scheduler(enhancement_config, total_steps)
    
    def setup_scheduler(self, config, total_steps):
        scheduler_type = config.get("scheduler_type", "cosine")
        warmup_steps = config.get("warmup_steps", 0)
        min_lr = config.get("min_lr", 1e-6)
        
        if scheduler_type == "cosine":
            # Ensure T_0 is at least 1 to avoid division by zero
            T_0 = max(total_steps // 10, 1)
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=T_0, T_mult=2, eta_min=min_lr
            )
        elif scheduler_type == "linear":
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
            
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif scheduler_type == "step":
            # Ensure step_size is at least 1
            step_size = max(total_steps // 10, 1)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.1)
    
    def step(self):
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

# Preset models for fine-tuning
PRESET_MODELS = {
    "small": {"d_model": 256, "n_layers": 4, "n_heads": 4},
    "medium": {"d_model": 512, "n_layers": 6, "n_heads": 8},
    "large": {"d_model": 768, "n_layers": 12, "n_heads": 12},
    "enhanced_small": {"d_model": 384, "n_layers": 8, "n_heads": 6, "use_enhanced": True},
    "enhanced_medium": {"d_model": 512, "n_layers": 12, "n_heads": 8, "use_enhanced": True},
    "enhanced_large": {"d_model": 768, "n_layers": 16, "n_heads": 12, "use_enhanced": True}
}

# Pre-trained model URLs (using smaller models for demonstration)
PRETRAINED_MODEL_URLS = {
    "small": "https://www.googleapis.com/download/storage/v1/b/pretrained-models-public/o/simplegpt-small.pth?generation=123456789012345&alt=media",
    "medium": "https://www.googleapis.com/download/storage/v1/b/pretrained-models-public/o/simplegpt-medium.pth?generation=123456789012346&alt=media",
    "large": "https://www.googleapis.com/download/storage/v1/b/pretrained-models-public/o/simplegpt-large.pth?generation=123456789012347&alt=media",
    "enhanced_small": "https://www.googleapis.com/download/storage/v1/b/pretrained-models-public/o/enhancedgpt-small.pth?generation=123456789012348&alt=media",
    "enhanced_medium": "https://www.googleapis.com/download/storage/v1/b/pretrained-models-public/o/enhancedgpt-medium.pth?generation=123456789012349&alt=media"
}

class ModelDownloader:
    """Handles downloading of pre-trained models"""
    @staticmethod
    def download_model(model_type, save_path):
        """Download a pre-trained model"""
        if model_type not in PRETRAINED_MODEL_URLS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        url = PRETRAINED_MODEL_URLS[model_type]
        response = requests.get(url, stream=True)
        
        if response.status_code != 200:
            # Fallback to creating a preset model if download fails
            return ModelDownloader.create_preset_model(model_type, save_path)
        
        # Save the downloaded model
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return save_path
    
    @staticmethod
    def create_preset_model(model_type, save_path):
        """Create a preset model if download fails"""
        if model_type not in PRESET_MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        config = PRESET_MODELS[model_type]
        tokenizer = SimpleTokenizer()
        
        # Create model based on type
        use_enhanced = config.get("use_enhanced", False)
        if use_enhanced:
            model = EnhancedGPT(
                vocab_size=1000,
                d_model=config["d_model"],
                n_layers=config["n_layers"],
                n_heads=config["n_heads"]
            )
        else:
            model = SimpleGPT(
                vocab_size=1000,
                d_model=config["d_model"],
                n_layers=config["n_layers"],
                n_heads=config["n_heads"]
            )
        
        # Save the preset model
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer': tokenizer,
            'config': config,
            'is_preset': True,
            'preset_name': model_type,
            'use_enhanced': use_enhanced
        }, save_path)
        
        return save_path

class ProjectSelectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Build a bot - Project Selection")
        self.root.geometry("800x600")
        set_icon_from_url(self.root, "https://raw.githubusercontent.com/lazerkatsweirdstuff/BuildABot/refs/heads/main/logo.png")
        self.project_manager = ProjectManager()
        self.current_project = None
        
        self.setup_ui()
        self.refresh_projects_list()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Build a bot", font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Projects list frame
        list_frame = ttk.LabelFrame(main_frame, text="Your Projects", padding="10")
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Projects list
        self.projects_listbox = tk.Listbox(list_frame, height=15, width=50)
        self.projects_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.projects_listbox.bind('<<ListboxSelect>>', self.on_project_select)
        
        # Scrollbar for listbox
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.projects_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.projects_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Project details frame
        details_frame = ttk.LabelFrame(main_frame, text="Project Details", padding="10")
        details_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Project details
        ttk.Label(details_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.project_name_var = tk.StringVar()
        ttk.Label(details_frame, textvariable=self.project_name_var).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(details_frame, text="Description:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.project_desc_var = tk.StringVar()
        ttk.Label(details_frame, textvariable=self.project_desc_var, wraplength=250).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(details_frame, text="Created:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.project_created_var = tk.StringVar()
        ttk.Label(details_frame, textvariable=self.project_created_var).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(details_frame, text="Modified:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.project_modified_var = tk.StringVar()
        ttk.Label(details_frame, textvariable=self.project_modified_var).grid(row=3, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(details_frame, text="Type:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.project_type_var = tk.StringVar()
        ttk.Label(details_frame, textvariable=self.project_type_var).grid(row=4, column=1, sticky=tk.W, pady=2)
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Buttons
        ttk.Button(buttons_frame, text="New Project", command=self.create_new_project).grid(row=0, column=0, padx=5)
        ttk.Button(buttons_frame, text="Open Project", command=self.open_project).grid(row=0, column=1, padx=5)
        ttk.Button(buttons_frame, text="Delete Project", command=self.delete_project).grid(row=0, column=2, padx=5)
        ttk.Button(buttons_frame, text="Refresh", command=self.refresh_projects_list).grid(row=0, column=3, padx=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        details_frame.columnconfigure(1, weight=1)
        
    def refresh_projects_list(self):
        self.projects_listbox.delete(0, tk.END)
        for project_id, project in self.project_manager.projects.items():
            self.projects_listbox.insert(tk.END, f"{project_id}: {project['name']}")
    
    def on_project_select(self, event):
        selection = self.projects_listbox.curselection()
        if selection:
            project_id = list(self.project_manager.projects.keys())[selection[0]]
            project = self.project_manager.projects[project_id]
            
            self.project_name_var.set(project['name'])
            self.project_desc_var.set(project['description'])
            self.project_created_var.set(project['created'][:10])  # Just the date part
            self.project_modified_var.set(project['modified'][:10])  # Just the date part
            
            model_type = project.get('model_type', 'scratch')
            if model_type == 'scratch':
                self.project_type_var.set("From Scratch")
            else:
                base_model = project.get('base_model', 'unknown')
                self.project_type_var.set(f"Fine-tuned from {os.path.basename(base_model)}")
            
            self.current_project = project_id
    
    def create_new_project(self):
        dialog = NewProjectDialog(self.root, self.project_manager)
        self.root.wait_window(dialog.top)
        self.refresh_projects_list()
    
    def open_project(self):
        if self.current_project:
            project_data = self.project_manager.get_project(self.current_project)
            if project_data:
                self.root.destroy()
                root = tk.Tk()
                app = AITrainerApp(root, project_data)
                root.mainloop()
        else:
            messagebox.showwarning("Warning", "Please select a project first")
    
    def delete_project(self):
        if self.current_project:
            result = messagebox.askyesno("Confirm Delete", 
                                        f"Are you sure you want to delete project '{self.project_manager.projects[self.current_project]['name']}'?")
            if result:
                if self.project_manager.delete_project(self.current_project):
                    messagebox.showinfo("Success", "Project deleted successfully")
                    self.current_project = None
                    self.project_name_var.set("")
                    self.project_desc_var.set("")
                    self.project_created_var.set("")
                    self.project_modified_var.set("")
                    self.project_type_var.set("")
                    self.refresh_projects_list()
                else:
                    messagebox.showerror("Error", "Failed to delete project")
        else:
            messagebox.showwarning("Warning", "Please select a project first")

class NewProjectDialog:
    def __init__(self, parent, project_manager):
        self.project_manager = project_manager
        self.top = tk.Toplevel(parent)
        self.top.title("Create New Project")
        self.top.geometry("500x400")
        self.top.grab_set()  # Modal dialog
        set_icon_from_url(self.top, "https://raw.githubusercontent.com/lazerkatsweirdstuff/BuildABot/refs/heads/main/logo.png")
        
        # Main frame
        main_frame = ttk.Frame(self.top, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Project name
        ttk.Label(main_frame, text="Project Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.name_var, width=40).grid(row=0, column=1, pady=5, sticky=(tk.W, tk.E))
        
        # Project description
        ttk.Label(main_frame, text="Description:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.desc_text = scrolledtext.ScrolledText(main_frame, height=5, width=30)
        self.desc_text.grid(row=1, column=1, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Model type
        ttk.Label(main_frame, text="Model Type:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.model_type_var = tk.StringVar(value="scratch")
        ttk.Radiobutton(main_frame, text="Create from scratch", variable=self.model_type_var, value="scratch", 
                       command=self.toggle_model_options).grid(row=2, column=1, sticky=tk.W, pady=2)
        ttk.Radiobutton(main_frame, text="Fine-tune existing model", variable=self.model_type_var, value="finetune",
                       command=self.toggle_model_options).grid(row=3, column=1, sticky=tk.W, pady=2)
        
        # Model selection (initially hidden)
        self.model_frame = ttk.Frame(main_frame)
        self.model_frame.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(self.model_frame, text="Base Model:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.base_model_var = tk.StringVar()
        model_combo = ttk.Combobox(self.model_frame, textvariable=self.base_model_var, width=15, state="readonly")
        model_combo['values'] = tuple(PRESET_MODELS.keys())
        model_combo.current(0)
        model_combo.grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(self.model_frame, text="or load from file:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.model_path_var = tk.StringVar()
        ttk.Entry(self.model_frame, textvariable=self.model_path_var, width=20).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(self.model_frame, text="Browse", command=self.browse_model_file).grid(row=1, column=2, sticky=tk.W, pady=2, padx=5)
        
        # Initially hide model options
        self.model_frame.grid_remove()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Create", command=self.create_project).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.top.destroy).pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        self.top.columnconfigure(0, weight=1)
        self.top.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        self.model_frame.columnconfigure(1, weight=1)
    
    def toggle_model_options(self):
        if self.model_type_var.get() == "finetune":
            self.model_frame.grid()
        else:
            self.model_frame.grid_remove()
    
    def browse_model_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)
    
    def create_project(self):
        name = self.name_var.get().strip()
        description = self.desc_text.get("1.0", tk.END).strip()
        
        if not name:
            messagebox.showerror("Error", "Please enter a project name")
            return
        
        model_type = self.model_type_var.get()
        base_model = None
        
        if model_type == "finetune":
            if self.model_path_var.get():
                base_model = self.model_path_var.get()
            else:
                base_model = self.base_model_var.get()
        
        project_id = self.project_manager.create_project(name, description, model_type, base_model)
        
        # If fine-tuning from a preset, create/download the base model
        if model_type == "finetune" and base_model in PRESET_MODELS:
            project_path = Path(self.project_manager.projects[project_id]["path"])
            model_path = project_path / "base_model.pth"
            
            # Download or create the model
            try:
                ModelDownloader.download_model(base_model, model_path)
                self.project_manager.update_project(project_id, {"base_model": str(model_path)})
            except Exception as e:
                messagebox.showerror("Error", f"Failed to get base model: {str(e)}")
                self.project_manager.delete_project(project_id)
                return
        
        messagebox.showinfo("Success", f"Project '{name}' created successfully")
        self.top.destroy()

class AITrainerApp:
    def __init__(self, root, project_data):
        self.root = root
        self.root.title(f"Build a bot - {project_data['name']}")
        self.root.geometry("1200x1000")  # Increased width for enhancements panel
        set_icon_from_url(self.root, "https://raw.githubusercontent.com/lazerkatsweirdstuff/BuildABot/refs/heads/main/logo.png")
        self.project_data = project_data
        self.project_path = Path(project_data["path"])
        
        # Use simple tokenizer for faster startup
        self.tokenizer = SimpleTokenizer()
        self.model = None
        self.training_data = []
        
        # Default model configuration
        self.model_config = {
            "d_model": 512,
            "n_layers": 6,
            "n_heads": 8,
            "max_seq_len": 512,
            "use_enhanced": False  # Whether to use enhanced architecture
        }
        
        # Model enhancements manager
        self.enhancements = ModelEnhancements()
        
        # If fine-tuning, load the base model configuration
        if project_data["model_type"] == "finetune" and project_data["base_model"]:
            self.load_base_model_config(project_data["base_model"])
        
        # Device configuration
        self.device_config = {
            "device_type": "auto"  # auto, cpu, or cuda
        }
        
        self.setup_ui()
        self.status_var.set("Ready - Load training data to begin")
        
        # Training control
        self.training_thread = None
        self.stop_training_flag = False
        
    def load_base_model_config(self, base_model_path):
        """Load configuration from base model for fine-tuning"""
        try:
            if base_model_path in PRESET_MODELS:
                # Use preset configuration
                self.model_config.update(PRESET_MODELS[base_model_path])
            else:
                # Load from file with PyTorch 2.6+ compatibility
                checkpoint = self.safe_torch_load(base_model_path)
                if 'config' in checkpoint:
                    self.model_config.update(checkpoint['config'])
                if 'use_enhanced' in checkpoint:
                    self.model_config['use_enhanced'] = checkpoint['use_enhanced']
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load base model configuration: {str(e)}")
    
    def safe_torch_load(self, file_path):
        """Safely load torch files with PyTorch 2.6+ compatibility"""
        try:
            # First try with weights_only=False for compatibility
            return torch.load(file_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Standard load failed: {e}")
            try:
                # If that fails, try with weights_only=True but add safe globals
                import torch.serialization
                torch.serialization.add_safe_globals([SimpleTokenizer])
                return torch.load(file_path, map_location='cpu', weights_only=True)
            except Exception as e2:
                print(f"Safe load failed: {e2}")
                # Last resort: try without weights_only parameter (for older PyTorch)
                return torch.load(file_path, map_location='cpu')
    
    def setup_ui(self):
        # Create main frames
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Project info frame
        project_frame = ttk.LabelFrame(main_frame, text=f"Project: {self.project_data['name']}", padding="5")
        project_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        ttk.Label(project_frame, text=f"Description: {self.project_data['description']}").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(project_frame, text=f"Type: {'From Scratch' if self.project_data['model_type'] == 'scratch' else 'Fine-tuning'}").grid(row=0, column=1, sticky=tk.W)
        
        if self.project_data['model_type'] == 'finetune':
            ttk.Label(project_frame, text=f"Base: {os.path.basename(self.project_data['base_model'])}").grid(row=0, column=2, sticky=tk.W)
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=1, column=0, sticky=(tk.N, tk.S), padx=5, pady=5)
        
        # Data management
        ttk.Label(control_frame, text="Data Management", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Button(control_frame, text="Load Text Files", 
                  command=self.load_training_files).grid(row=1, column=0, columnspan=2, pady=2, sticky=(tk.W, tk.E))
        
        ttk.Button(control_frame, text="View Training Data", 
                  command=self.view_training_data).grid(row=2, column=0, columnspan=2, pady=2, sticky=(tk.W, tk.E))
        
        # Device selection
        ttk.Label(control_frame, text="Device Selection", font=('Arial', 10, 'bold')).grid(row=3, column=0, columnspan=2, pady=(10, 5))
        
        ttk.Label(control_frame, text="Processing Device:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.device_var = tk.StringVar(value="auto")
        device_combo = ttk.Combobox(control_frame, textvariable=self.device_var, width=8)
        device_combo['values'] = ('auto', 'cpu', 'cuda')
        device_combo.grid(row=4, column=1, pady=2)
        
        # Add device info label
        device_info = "GPU available: " + ("Yes" if torch.cuda.is_available() else "No")
        ttk.Label(control_frame, text=device_info).grid(row=5, column=0, columnspan=2, pady=2)
        
        # Model configuration
        ttk.Label(control_frame, text="Model Configuration", font=('Arial', 10, 'bold')).grid(row=6, column=0, columnspan=2, pady=(10, 5))
        
        ttk.Label(control_frame, text="Embedding Size:").grid(row=7, column=0, sticky=tk.W, pady=2)
        self.embd_var = tk.StringVar(value=str(self.model_config["d_model"]))
        ttk.Entry(control_frame, textvariable=self.embd_var, width=10).grid(row=7, column=1, pady=2)
        
        ttk.Label(control_frame, text="Number of Layers:").grid(row=8, column=0, sticky=tk.W, pady=2)
        self.n_layer_var = tk.StringVar(value=str(self.model_config["n_layers"]))
        ttk.Entry(control_frame, textvariable=self.n_layer_var, width=10).grid(row=8, column=1, pady=2)
        
        ttk.Label(control_frame, text="Number of Heads:").grid(row=9, column=0, sticky=tk.W, pady=2)
        self.n_head_var = tk.StringVar(value=str(self.model_config["n_heads"]))
        ttk.Entry(control_frame, textvariable=self.n_head_var, width=10).grid(row=9, column=1, pady=2)
        
        # Enhanced architecture toggle
        self.enhanced_arch_var = tk.BooleanVar(value=self.model_config.get("use_enhanced", False))
        ttk.Checkbutton(control_frame, text="Use Enhanced Architecture", 
                       variable=self.enhanced_arch_var).grid(row=10, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        # Training parameters
        ttk.Label(control_frame, text="Training Parameters", font=('Arial', 10, 'bold')).grid(row=11, column=0, columnspan=2, pady=(10, 5))
        
        ttk.Label(control_frame, text="Batch Size:").grid(row=12, column=0, sticky=tk.W, pady=2)
        self.batch_size_var = tk.StringVar(value="4")
        ttk.Entry(control_frame, textvariable=self.batch_size_var, width=10).grid(row=12, column=1, pady=2)
        
        ttk.Label(control_frame, text="Learning Rate:").grid(row=13, column=0, sticky=tk.W, pady=2)
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(control_frame, textvariable=self.lr_var, width=10).grid(row=13, column=1, pady=2)
        
        ttk.Label(control_frame, text="Epochs:").grid(row=14, column=0, sticky=tk.W, pady=2)
        self.epochs_var = tk.StringVar(value="3")
        ttk.Entry(control_frame, textvariable=self.epochs_var, width=10).grid(row=14, column=1, pady=2)
        
        # Training controls
        ttk.Label(control_frame, text="Training Control", font=('Arial', 10, 'bold')).grid(row=15, column=0, columnspan=2, pady=(10, 5))
        
        self.train_btn = ttk.Button(control_frame, text="Start Training", 
                                   command=self.start_training)
        self.train_btn.grid(row=16, column=0, columnspan=2, pady=2, sticky=(tk.W, tk.E))
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Training", 
                                  command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.grid(row=17, column=0, columnspan=2, pady=2, sticky=(tk.W, tk.E))
        
        # Export buttons
        ttk.Label(control_frame, text="Export Model", font=('Arial', 10, 'bold')).grid(row=18, column=0, columnspan=2, pady=(10, 5))
        
        ttk.Button(control_frame, text="Save Model", 
                  command=self.save_model).grid(row=19, column=0, columnspan=2, pady=2, sticky=(tk.W, tk.E))
        
        ttk.Button(control_frame, text="Load Model", 
                  command=self.load_model).grid(row=20, column=0, columnspan=2, pady=2, sticky=(tk.W, tk.E))
        
        # Back to projects button
        ttk.Button(control_frame, text="Back to Projects", 
                  command=self.back_to_projects).grid(row=21, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # =============================================================================
        # MODEL ENHANCEMENTS PANEL
        # =============================================================================
        enhancements_frame = ttk.LabelFrame(main_frame, text="Model Enhancements", padding="5")
        enhancements_frame.grid(row=1, column=1, sticky=(tk.N, tk.S), padx=5, pady=5)
        
        # Enhancement controls
        ttk.Label(enhancements_frame, text="Toggle Enhancements:", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        # Create checkboxes for each enhancement
        self.enhancement_vars = {}
        row_idx = 1
        
        for enh_name, enh_config in self.enhancements.enhancements.items():
            var = tk.BooleanVar(value=enh_config["enabled"])
            self.enhancement_vars[enh_name] = var
            
            cb = ttk.Checkbutton(enhancements_frame, text=enh_config["name"], 
                                variable=var, command=lambda e=enh_name: self.toggle_enhancement(e))
            cb.grid(row=row_idx, column=0, columnspan=2, pady=2, sticky=tk.W)
            
            # Add description label
            desc_label = ttk.Label(enhancements_frame, text=enh_config["description"], 
                                  font=('Arial', 8), foreground="gray")
            desc_label.grid(row=row_idx + 1, column=0, columnspan=2, pady=(0, 5), sticky=tk.W)
            
            row_idx += 2
        
        # Enhancement configuration button
        ttk.Button(enhancements_frame, text="Configure Enhancements", 
                  command=self.configure_enhancements).grid(row=row_idx, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Right panel for status and output
        status_frame = ttk.LabelFrame(main_frame, text="Status & Output", padding="5")
        status_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_label.pack(fill=tk.X, pady=5)
        
        # Output text area
        self.output_text = scrolledtext.ScrolledText(status_frame, height=20, width=60)
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.output_text.insert(tk.END, "Training output will appear here...\n")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(2, weight=2)
        main_frame.rowconfigure(1, weight=1)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        
    def toggle_enhancement(self, enhancement_name):
        """Toggle an enhancement on/off based on checkbox state"""
        enabled = self.enhancement_vars[enhancement_name].get()
        self.enhancements.toggle_enhancement(enhancement_name, enabled)
        self.log_output(f"{'Enabled' if enabled else 'Disabled'} {enhancement_name}")
    
    def configure_enhancements(self):
        """Open enhancement configuration dialog"""
        dialog = EnhancementConfigDialog(self.root, self.enhancements)
        self.root.wait_window(dialog.top)
    
    def back_to_projects(self):
        """Return to project selection screen"""
        self.root.destroy()
        root = tk.Tk()
        app = ProjectSelectionApp(root)
        root.mainloop()
        
    def get_device(self):
        """Get the selected device based on user choice"""
        device_type = self.device_var.get()
        
        if device_type == "auto":
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_type == "cuda":
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                self.log_output("Warning: CUDA not available, falling back to CPU")
                return torch.device('cpu')
        else:
            return torch.device('cpu')
        
    def log_output(self, message):
        """Add message to output text area"""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update_idletasks()
        
    def verify_model_file(self, file_path):
        """Verify if a model file is valid before loading"""
        try:
            # Simple file checks
            if not os.path.exists(file_path):
                return False, "File does not exist"
            
            if os.path.getsize(file_path) < 1024:  # Less than 1KB
                return False, "File is too small to be a valid model"
            
            return True, "File appears valid"
        except Exception as e:
            return False, f"Error verifying file: {str(e)}"
        
    def load_training_files(self):
        files = filedialog.askopenfilenames(
            title="Select training files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if files:
            self.status_var.set("Loading training files...")
            self.log_output("Loading training files...")
            self.root.update()
            
            total_texts = []
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Split into smaller chunks if needed
                        chunks = self.split_into_chunks(content, 1000)
                        total_texts.extend(chunks)
                        self.log_output(f"Loaded {len(chunks)} chunks from {os.path.basename(file_path)}")
                except Exception as e:
                    error_msg = f"Error reading {file_path}: {str(e)}"
                    self.log_output(error_msg)
                    messagebox.showerror("Error", error_msg)
            
            self.training_data.extend(total_texts)
            
            # Build vocabulary from all training texts
            self.tokenizer.build_vocab_from_texts(self.training_data, max_vocab_size=10000)
            
            status_msg = f"Loaded {len(total_texts)} text chunks from {len(files)} files"
            self.status_var.set(status_msg)
            self.log_output(status_msg)
            self.log_output(f"Vocabulary size: {self.tokenizer.vocab_size}")
            
    def split_into_chunks(self, text, chunk_size):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
        return chunks
    
    def view_training_data(self):
        if not self.training_data:
            messagebox.showinfo("Info", "No training data loaded")
            return
            
        view_window = tk.Toplevel(self.root)
        view_window.title("Training Data Preview")
        view_window.geometry("800x600")
        set_icon_from_url(self.root, "https://raw.githubusercontent.com/lazerkatsweirdstuff/BuildABot/refs/heads/main/logo.png")
        text_widget = scrolledtext.ScrolledText(view_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for i, text in enumerate(self.training_data[:50]):  # Show first 50 chunks
            text_widget.insert(tk.END, f"Chunk {i+1}:\n{text}\n\n{'='*50}\n\n")
            
    def start_training(self):
        if not self.training_data:
            messagebox.showerror("Error", "No training data loaded!")
            return
            
        self.stop_training_flag = False
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress.start()
        self.status_var.set("Training started...")
        self.log_output("Starting training...")
        
        # Update model config from UI
        self.model_config.update({
            "d_model": int(self.embd_var.get()),
            "n_layers": int(self.n_layer_var.get()),
            "n_heads": int(self.n_head_var.get()),
            "use_enhanced": self.enhanced_arch_var.get()
        })
        
        # Update device config from UI
        self.device_config.update({
            "device_type": self.device_var.get()
        })
        
        # Start training in separate thread
        self.training_thread = threading.Thread(target=self.train_model)
        self.training_thread.daemon = True
        self.training_thread.start()
        
    def stop_training(self):
        self.stop_training_flag = True
        self.status_var.set("Stopping training...")
        self.log_output("Stopping training...")
        
    def train_model(self):
        try:
            # Create dataset and dataloader
            dataset = TextDataset(self.training_data, self.tokenizer)
            dataloader = DataLoader(
                dataset, 
                batch_size=int(self.batch_size_var.get()),
                shuffle=True
            )
            
            # Initialize model based on architecture choice
            use_enhanced = self.model_config.get("use_enhanced", False)
            
            if use_enhanced:
                self.log_output("Using Enhanced GPT architecture")
                model_class = EnhancedGPT
            else:
                self.log_output("Using Simple GPT architecture")
                model_class = SimpleGPT
            
            if self.project_data["model_type"] == "scratch" or not self.project_data["base_model"]:
                # Create from scratch
                self.model = model_class(
                    vocab_size=self.tokenizer.vocab_size,
                    d_model=self.model_config["d_model"],
                    n_layers=self.model_config["n_layers"],
                    n_heads=self.model_config["n_heads"],
                    max_seq_len=self.model_config["max_seq_len"]
                )
            else:
                # Fine-tune existing model
                self.log_output(f"Loading base model from {self.project_data['base_model']}")
                
                if self.project_data["base_model"] in PRESET_MODELS:
                    # Use preset configuration
                    self.model = model_class(
                        vocab_size=self.tokenizer.vocab_size,
                        d_model=self.model_config["d_model"],
                        n_layers=self.model_config["n_layers"],
                        n_heads=self.model_config["n_heads"],
                        max_seq_len=self.model_config["max_seq_len"]
                    )
                else:
                    # Load from file with PyTorch 2.6+ compatibility
                    checkpoint = self.safe_torch_load(self.project_data["base_model"])
                    
                    # Create model with loaded configuration
                    self.model = model_class(
                        vocab_size=self.tokenizer.vocab_size,
                        d_model=checkpoint['config']["d_model"],
                        n_layers=checkpoint['config']["n_layers"],
                        n_heads=checkpoint['config']["n_heads"],
                        max_seq_len=checkpoint['config']["max_seq_len"]
                    )
                    
                    # Load weights (if compatible)
                    try:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        self.log_output("Successfully loaded base model weights")
                    except:
                        self.log_output("Could not load base model weights (incompatible), starting with random weights")
            
            # Get enabled enhancements
            enabled_enhancements = self.enhancements.get_enabled_enhancements()
            self.log_output(f"Enabled enhancements: {list(enabled_enhancements.keys())}")
            
            # Setup enhanced optimizer if scheduling is enabled
            total_steps = len(dataloader) * int(self.epochs_var.get())
            
            if "learning_rate_scheduling" in enabled_enhancements:
                enh_config = enabled_enhancements["learning_rate_scheduling"]["config"]
                optimizer = EnhancedOptimizer(
                    self.model.parameters(), 
                    enh_config, 
                    total_steps
                )
                self.log_output(f"Using enhanced optimizer with {enh_config['scheduler_type']} scheduling")
            else:
                # Standard optimizer
                lr = float(self.lr_var.get())
                weight_decay = 0.0
                
                if "advanced_regularization" in enabled_enhancements:
                    weight_decay = enabled_enhancements["advanced_regularization"]["config"].get("weight_decay", 0.01)
                
                optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            
            # Training loop
            num_epochs = int(self.epochs_var.get())
            device = self.get_device()
            self.model.to(device)
            self.log_output(f"Using device: {device}")
            
            # Gradient accumulation
            accumulation_steps = 1
            if "gradient_accumulation" in enabled_enhancements:
                accumulation_steps = enabled_enhancements["gradient_accumulation"]["config"].get("accumulation_steps", 4)
                self.log_output(f"Using gradient accumulation with {accumulation_steps} steps")
            
            # Early stopping setup
            best_loss = float('inf')
            patience_counter = 0
            early_stopping_patience = 0
            
            if "early_stopping" in enabled_enhancements:
                early_stopping_patience = enabled_enhancements["early_stopping"]["config"].get("patience", 3)
                self.log_output(f"Early stopping enabled with patience {early_stopping_patience}")
            
            for epoch in range(num_epochs):
                if self.stop_training_flag:
                    break
                    
                self.model.train()
                total_loss = 0
                total_batches = 0
                
                optimizer.zero_grad()
                
                for batch_idx, batch in enumerate(dataloader):
                    if self.stop_training_flag:
                        break
                        
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Debug: Check for invalid token IDs
                    max_id = input_ids.max().item()
                    if max_id >= self.tokenizer.vocab_size:
                        self.log_output(f"Warning: Found token ID {max_id} but vocab size is {self.tokenizer.vocab_size}")
                        # Clamp values to valid range
                        input_ids = torch.clamp(input_ids, 0, self.tokenizer.vocab_size - 1)
                        labels = torch.clamp(labels, 0, self.tokenizer.vocab_size - 1)
                    
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs['loss']
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.log_output("Warning: NaN or Inf loss detected, skipping batch")
                        continue
                    
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0:
                        # Gradient clipping
                        if "advanced_regularization" in enabled_enhancements:
                            max_norm = enabled_enhancements["advanced_regularization"]["config"].get("gradient_clip", 1.0)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                        
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    total_loss += loss.item() * accumulation_steps
                    total_batches += 1
                    
                    if batch_idx % 10 == 0:
                        status_msg = f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item() * accumulation_steps:.4f}"
                        self.status_var.set(status_msg)
                        if batch_idx % 50 == 0:  # Log less frequently to avoid UI slowdown
                            self.log_output(status_msg)
                
                # Handle remaining gradients
                if (batch_idx + 1) % accumulation_steps != 0:
                    if "advanced_regularization" in enabled_enhancements:
                        max_norm = enabled_enhancements["advanced_regularization"]["config"].get("gradient_clip", 1.0)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                
                if total_batches > 0:
                    avg_loss = total_loss / total_batches
                    epoch_msg = f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}"
                    
                    # Learning rate info
                    if hasattr(optimizer, 'get_lr'):
                        lr = optimizer.get_lr()
                        epoch_msg += f", LR: {lr:.6f}"
                    
                    self.status_var.set(epoch_msg)
                    self.log_output(epoch_msg)
                    
                    # Early stopping check
                    if early_stopping_patience > 0:
                        if avg_loss < best_loss - enabled_enhancements["early_stopping"]["config"].get("min_delta", 0.001):
                            best_loss = avg_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            
                        if patience_counter >= early_stopping_patience:
                            self.log_output(f"Early stopping triggered after {epoch+1} epochs")
                            break
                
            if not self.stop_training_flag:
                completion_msg = "Training completed successfully!"
                self.status_var.set(completion_msg)
                self.log_output(completion_msg)
                
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            self.status_var.set(error_msg)
            self.log_output(error_msg)
            import traceback
            self.log_output(traceback.format_exc())
            messagebox.showerror("Training Error", str(e))
            
        finally:
            self.train_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.progress.stop()
            self.stop_training_flag = False
            
    def save_model(self):
        if self.model is None:
            messagebox.showerror("Error", "No model to save!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")],
            initialdir=self.project_path
        )
        
        if file_path:
            try:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'tokenizer': self.tokenizer,
                    'config': self.model_config,
                    'training_data_info': {
                        'num_chunks': len(self.training_data),
                        'vocab_size': self.tokenizer.vocab_size
                    },
                    'project_info': self.project_data,
                    'use_enhanced': self.model_config.get("use_enhanced", False),
                    'enhancements': self.enhancements.get_enabled_enhancements()
                }, file_path)
                
                success_msg = f"Model saved to {file_path}"
                self.status_var.set(success_msg)
                self.log_output(success_msg)
                messagebox.showinfo("Success", "Model saved successfully!")
                
            except Exception as e:
                error_msg = f"Error saving model: {str(e)}"
                self.log_output(error_msg)
                messagebox.showerror("Save Error", error_msg)
                
    def load_model(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")],
            initialdir=self.project_path
        )
        
        if file_path:
            try:
                # Use safe loading method for PyTorch 2.6+ compatibility
                checkpoint = self.safe_torch_load(file_path)
                
                # Determine model architecture
                use_enhanced = checkpoint.get('use_enhanced', False)
                if use_enhanced:
                    model_class = EnhancedGPT
                    self.log_output("Loading Enhanced GPT model")
                else:
                    model_class = SimpleGPT
                    self.log_output("Loading Simple GPT model")
                
                # Recreate the model architecture
                self.model_config = checkpoint['config']
                self.model = model_class(
                    vocab_size=checkpoint['tokenizer'].vocab_size,
                    d_model=self.model_config["d_model"],
                    n_layers=self.model_config["n_layers"],
                    n_heads=self.model_config["n_heads"],
                    max_seq_len=self.model_config["max_seq_len"]
                )
                
                # Load weights
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Load tokenizer
                self.tokenizer = checkpoint['tokenizer']
                
                # Load enhancements if available
                if 'enhancements' in checkpoint:
                    for enh_name, enh_config in checkpoint['enhancements'].items():
                        if enh_name in self.enhancements.enhancements:
                            self.enhancements.enhancements[enh_name].update(enh_config)
                            if enh_name in self.enhancement_vars:
                                self.enhancement_vars[enh_name].set(enh_config.get('enabled', False))
                
                # Update UI with loaded config
                self.embd_var.set(str(self.model_config['d_model']))
                self.n_layer_var.set(str(self.model_config['n_layers']))
                self.n_head_var.set(str(self.model_config['n_heads']))
                self.enhanced_arch_var.set(use_enhanced)
                
                success_msg = f"Model loaded from {file_path}"
                self.status_var.set(success_msg)
                self.log_output(success_msg)
                messagebox.showinfo("Success", "Model loaded successfully!")
                
            except Exception as e:
                error_msg = f"Error loading model: {str(e)}"
                self.log_output(error_msg)
                messagebox.showerror("Load Error", error_msg)

class EnhancementConfigDialog:
    """Dialog for configuring model enhancements"""
    
    def __init__(self, parent, enhancements_manager):
        self.enhancements = enhancements_manager
        self.top = tk.Toplevel(parent)
        self.top.title("Configure Model Enhancements")
        self.top.geometry("600x500")
        self.top.grab_set()
        
        # Initialize config_vars dictionary - FIXED: Add this line
        self.config_vars = {}
        
        # Main frame
        main_frame = ttk.Frame(self.top, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Notebook for different enhancement categories
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create tabs for each enhancement category
        self.create_regularization_tab(notebook)
        self.create_scheduling_tab(notebook)
        self.create_training_tab(notebook)
        self.create_advanced_tab(notebook)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Apply", command=self.apply_config).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.top.destroy).pack(side=tk.RIGHT, padx=5)
    
    def create_regularization_tab(self, notebook):
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Regularization")
        
        ttk.Label(frame, text="Advanced Regularization Settings", font=('Arial', 11, 'bold')).grid(row=0, column=0, columnspan=2, pady=10, sticky=tk.W)
        
        # Weight decay
        ttk.Label(frame, text="Weight Decay:").grid(row=1, column=0, sticky=tk.W, pady=5)
        weight_decay_var = tk.DoubleVar(value=self.enhancements.enhancements["advanced_regularization"]["config"]["weight_decay"])
        ttk.Scale(frame, from_=0.0, to=0.1, variable=weight_decay_var, orient=tk.HORIZONTAL).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(frame, textvariable=tk.StringVar(value=f"{weight_decay_var.get():.3f}")).grid(row=1, column=2, padx=5)
        self.config_vars["weight_decay"] = weight_decay_var
        
        # Gradient clipping
        ttk.Label(frame, text="Gradient Clip:").grid(row=2, column=0, sticky=tk.W, pady=5)
        grad_clip_var = tk.DoubleVar(value=self.enhancements.enhancements["advanced_regularization"]["config"]["gradient_clip"])
        ttk.Scale(frame, from_=0.1, to=5.0, variable=grad_clip_var, orient=tk.HORIZONTAL).grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(frame, textvariable=tk.StringVar(value=f"{grad_clip_var.get():.1f}")).grid(row=2, column=2, padx=5)
        self.config_vars["gradient_clip"] = grad_clip_var
        
        # Label smoothing
        ttk.Label(frame, text="Label Smoothing:").grid(row=3, column=0, sticky=tk.W, pady=5)
        label_smooth_var = tk.DoubleVar(value=self.enhancements.enhancements["advanced_regularization"]["config"]["label_smoothing"])
        ttk.Scale(frame, from_=0.0, to=0.3, variable=label_smooth_var, orient=tk.HORIZONTAL).grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(frame, textvariable=tk.StringVar(value=f"{label_smooth_var.get():.2f}")).grid(row=3, column=2, padx=5)
        self.config_vars["label_smoothing"] = label_smooth_var
        
        # Dropout rate
        ttk.Label(frame, text="Dropout Rate:").grid(row=4, column=0, sticky=tk.W, pady=5)
        dropout_var = tk.DoubleVar(value=self.enhancements.enhancements["advanced_regularization"]["config"]["dropout_rate"])
        ttk.Scale(frame, from_=0.0, to=0.5, variable=dropout_var, orient=tk.HORIZONTAL).grid(row=4, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(frame, textvariable=tk.StringVar(value=f"{dropout_var.get():.2f}")).grid(row=4, column=2, padx=5)
        self.config_vars["dropout_rate"] = dropout_var
        
        frame.columnconfigure(1, weight=1)
    
    def create_scheduling_tab(self, notebook):
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Scheduling")
        
        ttk.Label(frame, text="Learning Rate Scheduling", font=('Arial', 11, 'bold')).grid(row=0, column=0, columnspan=2, pady=10, sticky=tk.W)
        
        # Scheduler type
        ttk.Label(frame, text="Scheduler Type:").grid(row=1, column=0, sticky=tk.W, pady=5)
        scheduler_var = tk.StringVar(value=self.enhancements.enhancements["learning_rate_scheduling"]["config"]["scheduler_type"])
        scheduler_combo = ttk.Combobox(frame, textvariable=scheduler_var, state="readonly")
        scheduler_combo['values'] = ('cosine', 'linear', 'step')
        scheduler_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        self.config_vars["scheduler_type"] = scheduler_var
        
        # Warmup steps
        ttk.Label(frame, text="Warmup Steps:").grid(row=2, column=0, sticky=tk.W, pady=5)
        warmup_var = tk.IntVar(value=self.enhancements.enhancements["learning_rate_scheduling"]["config"]["warmup_steps"])
        ttk.Entry(frame, textvariable=warmup_var).grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        self.config_vars["warmup_steps"] = warmup_var
        
        # Minimum learning rate
        ttk.Label(frame, text="Minimum LR:").grid(row=3, column=0, sticky=tk.W, pady=5)
        min_lr_var = tk.StringVar(value=str(self.enhancements.enhancements["learning_rate_scheduling"]["config"]["min_lr"]))
        ttk.Entry(frame, textvariable=min_lr_var).grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)
        self.config_vars["min_lr"] = min_lr_var
        
        frame.columnconfigure(1, weight=1)
    
    def create_training_tab(self, notebook):
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Training")
        
        ttk.Label(frame, text="Training Enhancements", font=('Arial', 11, 'bold')).grid(row=0, column=0, columnspan=2, pady=10, sticky=tk.W)
        
        # Gradient accumulation
        ttk.Label(frame, text="Gradient Accumulation Steps:").grid(row=1, column=0, sticky=tk.W, pady=5)
        accum_var = tk.IntVar(value=self.enhancements.enhancements["gradient_accumulation"]["config"]["accumulation_steps"])
        ttk.Scale(frame, from_=1, to=16, variable=accum_var, orient=tk.HORIZONTAL).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(frame, textvariable=tk.StringVar(value=str(accum_var.get()))).grid(row=1, column=2, padx=5)
        self.config_vars["accumulation_steps"] = accum_var
        
        # Early stopping patience
        ttk.Label(frame, text="Early Stopping Patience:").grid(row=2, column=0, sticky=tk.W, pady=5)
        patience_var = tk.IntVar(value=self.enhancements.enhancements["early_stopping"]["config"]["patience"])
        ttk.Scale(frame, from_=1, to=10, variable=patience_var, orient=tk.HORIZONTAL).grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(frame, textvariable=tk.StringVar(value=str(patience_var.get()))).grid(row=2, column=2, padx=5)
        self.config_vars["patience"] = patience_var
        
        # Early stopping min delta
        ttk.Label(frame, text="Early Stopping Min Delta:").grid(row=3, column=0, sticky=tk.W, pady=5)
        delta_var = tk.DoubleVar(value=self.enhancements.enhancements["early_stopping"]["config"]["min_delta"])
        ttk.Scale(frame, from_=0.0001, to=0.01, variable=delta_var, orient=tk.HORIZONTAL).grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(frame, textvariable=tk.StringVar(value=f"{delta_var.get():.4f}")).grid(row=3, column=2, padx=5)
        self.config_vars["min_delta"] = delta_var
        
        frame.columnconfigure(1, weight=1)
    
    def create_advanced_tab(self, notebook):
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Advanced")
        
        ttk.Label(frame, text="Advanced Model Settings", font=('Arial', 11, 'bold')).grid(row=0, column=0, columnspan=2, pady=10, sticky=tk.W)
        
        # Mixed precision
        ttk.Label(frame, text="Mixed Precision:").grid(row=1, column=0, sticky=tk.W, pady=5)
        mixed_precision_var = tk.BooleanVar(value=self.enhancements.enhancements["mixed_precision"]["enabled"])
        ttk.Checkbutton(frame, text="Enable Mixed Precision Training", variable=mixed_precision_var).grid(row=1, column=1, sticky=tk.W, pady=5)
        self.config_vars["mixed_precision"] = mixed_precision_var
        
        ttk.Label(frame, text="Note: Mixed precision requires CUDA and compatible GPU", font=('Arial', 8), foreground="gray").grid(row=2, column=0, columnspan=2, sticky=tk.W)
        
        frame.columnconfigure(1, weight=1)
    
    def apply_config(self):
        """Apply configuration changes to enhancements"""
        try:
            # Update regularization config
            self.enhancements.update_config("advanced_regularization", {
                "weight_decay": self.config_vars["weight_decay"].get(),
                "gradient_clip": self.config_vars["gradient_clip"].get(),
                "label_smoothing": self.config_vars["label_smoothing"].get(),
                "dropout_rate": self.config_vars["dropout_rate"].get()
            })
            
            # Update scheduling config
            self.enhancements.update_config("learning_rate_scheduling", {
                "scheduler_type": self.config_vars["scheduler_type"].get(),
                "warmup_steps": self.config_vars["warmup_steps"].get(),
                "min_lr": float(self.config_vars["min_lr"].get())
            })
            
            # Update training config
            self.enhancements.update_config("gradient_accumulation", {
                "accumulation_steps": self.config_vars["accumulation_steps"].get()
            })
            
            self.enhancements.update_config("early_stopping", {
                "patience": self.config_vars["patience"].get(),
                "min_delta": self.config_vars["min_delta"].get()
            })
            
            # Update mixed precision
            self.enhancements.toggle_enhancement("mixed_precision", self.config_vars["mixed_precision"].get())
            
            messagebox.showinfo("Success", "Enhancement configurations applied successfully!")
            self.top.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply configurations: {str(e)}")

def main():
    root = tk.Tk()
    app = ProjectSelectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
