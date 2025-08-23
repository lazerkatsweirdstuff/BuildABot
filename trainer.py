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

class AITrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Text Generation Trainer")
        self.root.geometry("1000x700")
        
        # Use simple tokenizer for faster startup
        self.tokenizer = SimpleTokenizer()
        self.model = None
        self.training_data = []
        
        # Default model configuration
        self.model_config = {
            "d_model": 512,
            "n_layers": 6,
            "n_heads": 8,
            "max_seq_len": 512
        }
        
        # Device configuration
        self.device_config = {
            "device_type": "auto"  # auto, cpu, or cuda
        }
        
        self.setup_ui()
        self.status_var.set("Ready - Load training data to begin")
        
        # Training control
        self.training_thread = None
        self.stop_training_flag = False
        
    def setup_ui(self):
        # Create main frames
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, sticky=(tk.N, tk.S), padx=5, pady=5)
        
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
        self.embd_var = tk.StringVar(value="512")
        ttk.Entry(control_frame, textvariable=self.embd_var, width=10).grid(row=7, column=1, pady=2)
        
        ttk.Label(control_frame, text="Number of Layers:").grid(row=8, column=0, sticky=tk.W, pady=2)
        self.n_layer_var = tk.StringVar(value="6")
        ttk.Entry(control_frame, textvariable=self.n_layer_var, width=10).grid(row=8, column=1, pady=2)
        
        ttk.Label(control_frame, text="Number of Heads:").grid(row=9, column=0, sticky=tk.W, pady=2)
        self.n_head_var = tk.StringVar(value="8")
        ttk.Entry(control_frame, textvariable=self.n_head_var, width=10).grid(row=9, column=1, pady=2)
        
        # Training parameters
        ttk.Label(control_frame, text="Training Parameters", font=('Arial', 10, 'bold')).grid(row=10, column=0, columnspan=2, pady=(10, 5))
        
        ttk.Label(control_frame, text="Batch Size:").grid(row=11, column=0, sticky=tk.W, pady=2)
        self.batch_size_var = tk.StringVar(value="4")
        ttk.Entry(control_frame, textvariable=self.batch_size_var, width=10).grid(row=11, column=1, pady=2)
        
        ttk.Label(control_frame, text="Learning Rate:").grid(row=12, column=0, sticky=tk.W, pady=2)
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(control_frame, textvariable=self.lr_var, width=10).grid(row=12, column=1, pady=2)
        
        ttk.Label(control_frame, text="Epochs:").grid(row=13, column=0, sticky=tk.W, pady=2)
        self.epochs_var = tk.StringVar(value="3")
        ttk.Entry(control_frame, textvariable=self.epochs_var, width=10).grid(row=13, column=1, pady=2)
        
        # Training controls
        ttk.Label(control_frame, text="Training Control", font=('Arial', 10, 'bold')).grid(row=14, column=0, columnspan=2, pady=(10, 5))
        
        self.train_btn = ttk.Button(control_frame, text="Start Training", 
                                   command=self.start_training)
        self.train_btn.grid(row=15, column=0, columnspan=2, pady=2, sticky=(tk.W, tk.E))
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Training", 
                                  command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.grid(row=16, column=0, columnspan=2, pady=2, sticky=(tk.W, tk.E))
        
        # Export buttons
        ttk.Label(control_frame, text="Export Model", font=('Arial', 10, 'bold')).grid(row=17, column=0, columnspan=2, pady=(10, 5))
        
        ttk.Button(control_frame, text="Save Model", 
                  command=self.save_model).grid(row=18, column=0, columnspan=2, pady=2, sticky=(tk.W, tk.E))
        
        ttk.Button(control_frame, text="Load Model", 
                  command=self.load_model).grid(row=19, column=0, columnspan=2, pady=2, sticky=(tk.W, tk.E))
        
        # Right panel for status and output
        status_frame = ttk.LabelFrame(main_frame, text="Status & Output", padding="5")
        status_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
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
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        
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
            "n_heads": int(self.n_head_var.get())
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
            
            # Initialize model
            self.model = SimpleGPT(
                vocab_size=self.tokenizer.vocab_size,
                d_model=self.model_config["d_model"],
                n_layers=self.model_config["n_layers"],
                n_heads=self.model_config["n_heads"],
                max_seq_len=self.model_config["max_seq_len"]
            )
            
            # Setup optimizer
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=float(self.lr_var.get())
            )
            
            # Training loop
            num_epochs = int(self.epochs_var.get())
            device = self.get_device()
            self.model.to(device)
            self.log_output(f"Using device: {device}")
            
            for epoch in range(num_epochs):
                if self.stop_training_flag:
                    break
                    
                self.model.train()
                total_loss = 0
                total_batches = 0
                
                for batch_idx, batch in enumerate(dataloader):
                    if self.stop_training_flag:
                        break
                        
                    optimizer.zero_grad()
                    
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
                    
                    loss.backward()
                    
                    # Gradient clipping to prevent explosions
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_batches += 1
                    
                    if batch_idx % 10 == 0:
                        status_msg = f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}"
                        self.status_var.set(status_msg)
                        if batch_idx % 50 == 0:  # Log less frequently to avoid UI slowdown
                            self.log_output(status_msg)
                
                if total_batches > 0:
                    avg_loss = total_loss / total_batches
                    epoch_msg = f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}"
                    self.status_var.set(epoch_msg)
                    self.log_output(epoch_msg)
                
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
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
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
                    }
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
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                checkpoint = torch.load(file_path, map_location='cpu')
                
                # Recreate the model architecture
                self.model_config = checkpoint['config']
                self.model = SimpleGPT(
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
                
                # Update UI with loaded config
                self.embd_var.set(str(self.model_config['d_model']))
                self.n_layer_var.set(str(self.model_config['n_layers']))
                self.n_head_var.set(str(self.model_config['n_heads']))
                
                success_msg = f"Model loaded from {file_path}"
                self.status_var.set(success_msg)
                self.log_output(success_msg)
                messagebox.showinfo("Success", "Model loaded successfully!")
                
            except Exception as e:
                error_msg = f"Error loading model: {str(e)}"
                self.log_output(error_msg)
                messagebox.showerror("Load Error", error_msg)

def main():
    root = tk.Tk()
    app = AITrainerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()