`#!/usr/bin/env python3
"""
Persian QA Processor - Telegram Chat to Q&A Converter
-----------------------------------------------------
Processes Telegram chat history from JSON into structured Q&A pairs.
"""

import re
import json
import requests
import numpy as np
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.rich import tqdm
from parsivar import Normalizer, Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import hdbscan
import os

# =====================
# Configuration Settings
# =====================
CONFIG = {
    # File Paths (updated for your specific locations)
    "paths": {
        "input": "/Users/shahriarfarzadi/Documents/telegram-project/chat.json",
        "output": "/Users/shahriarfarzadi/Documents/telegram-project/qa.txt"
    },
    
    # Text Processing
    "text": {
        "max_length": 350,
        "allowed_chars": r"A-Za-z0-9@#%_\-\/\.:",
        "persian_range": r"\u0600-\u06FF",
        "stemming_rules": [
            (r"^(ن?می)", ""),
            (r"ها$", ""),
            (r"تر((ین)?)$", r"\1")
        ]
    },
    
    # Model Configuration
    "model": {
        "embed": "dariush7/parsbert-qa-uncased",
        "ollama": "gemma3:4b-instruct-q4_K_M",
        "cluster_min_size": 5
    },
    
    # Performance Settings
    "performance": {
        "batch_size": {
            "clean": 512,
            "embed": 128,
            "infer": 8
        },
        "timeout": 120
    }
}

class TelegramQAProcessor:
    """Process Telegram JSON export into Q&A pairs"""
    
    def __init__(self):
        """Initialize processing components"""
        # Persian NLP Tools
        self.normalizer = Normalizer()
        self.tokenizer = Tokenizer()
        
        # Regex Components
        self.clean_regex = re.compile(
            f"(?![{CONFIG['text']['allowed_chars']}])"
            f"[^{CONFIG['text']['persian_range']}\\s]",
            re.UNICODE
        )
        self.stem_patterns = [
            (re.compile(pattern), repl)
            for pattern, repl in CONFIG["text"]["stemming_rules"]
        ]
        
        # Embedding Model
        self.embedder = SentenceTransformer(
            CONFIG["model"]["embed"],
            device="mps"
        )

    def load_chat_data(self) -> List[Dict]:
        """Load and preprocess Telegram JSON data"""
        with open(CONFIG["paths"]["input"], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = []
        for msg in tqdm(data["messages"], desc="📥 Loading messages"):
            if "text" not in msg:
                continue
                
            # Handle both string and list-type messages
            if isinstance(msg["text"], list):
                text = " ".join(
                    segment["text"] 
                    for segment in msg["text"] 
                    if isinstance(segment, dict) and "text" in segment
                )
            else:
                text = msg["text"]
            
            messages.append({
                "id": msg["id"],
                "date": msg["date"],
                "text": text,
                "clean_text": self.clean_text(text)
            })
        
        return messages

    def clean_text(self, text: str) -> str:
        """Clean and normalize text while preserving English content"""
        text = self.normalizer.normalize(str(text))[:CONFIG["text"]["max_length"]]
        text = self.clean_regex.sub('', text)
        for pattern, repl in self.stem_patterns:
            text = pattern.sub(repl, text)
        return text.strip()

    def generate_qa_pairs(self, messages: List[Dict]) -> List[Dict]:
        """Main processing pipeline"""
        # 1. Generate embeddings
        embeddings = self.embedder.encode(
            [msg["clean_text"] for msg in messages],
            batch_size=CONFIG["performance"]["batch_size"]["embed"],
            show_progress_bar=False
        )
        
        # 2. Cluster messages
        clusters = hdbscan.HDBSCAN(
            min_cluster_size=CONFIG["model"]["cluster_min_size"],
            metric="cosine",
            core_dist_n_jobs=4
        ).fit_predict(normalize(embeddings))
        
        # 3. Generate QA for each cluster
        cluster_texts = [
            msg["text"] for msg, cluster 
            in zip(messages, clusters) 
            if cluster != -1
        ]
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for text in cluster_texts:
                futures.append(
                    executor.submit(
                        self._generate_single_qa,
                        text
                    )
                )
            
            return [
                result for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="🧠 Generating Q&A"
                )
                if (result := future.result()) is not None
            ]

    def _generate_single_qa(self, text: str) -> Dict:
        """Generate QA pair via Ollama API"""
        prompt = f"""
        به عنوان یک دانشجوی تحصیلات تکمیلی، این متن را به یک پرسش و پاسخ دانشگاهی تبدیل کن:
        متن: {text[:1500]}
        
        ساختار الزامی:
        - پرسش با "پرسش:" شروع شود
        - پاسخ با "پاسخ:" شروع شود
        - از اصطلاحات آکادمیک استفاده شود
        """
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": CONFIG["model"]["ollama"],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_ctx": 4096
                    }
                },
                timeout=CONFIG["performance"]["timeout"]
            )
            return self._parse_qa(response.json()["response"])
        except Exception as e:
            tqdm.write(f"⚠️ Error generating QA: {str(e)}")
            return None

    def _parse_qa(self, text: str) -> Dict:
        """Parse model response into Q&A dictionary"""
        q, a = "", ""
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith("پرسش:"):
                q = line[5:].strip()
            elif line.startswith("پاسخ:"):
                a = line[5:].strip()
        return {"question": q, "answer": a} if q and a else None

    def save_results(self, qa_pairs: List[Dict]):
        """Save QA pairs to text file with Persian formatting"""
        with open(CONFIG["paths"]["output"], 'w', encoding='utf-8') as f:
            for i, pair in enumerate(qa_pairs, 1):
                f.write(f"پرسش {i}:\n{pair['question']}\n\n")
                f.write(f"پاسخ {i}:\n{pair['answer']}\n\n")
                f.write("=" * 50 + "\n\n")

# =============
# Main Execution
# =============
if __name__ == "__main__":
    processor = TelegramQAProcessor()
    
    print("🔍 Starting Telegram chat processing...")
    messages = processor.load_chat_data()
    
    print("🧠 Generating Q&A pairs...")
    qa_pairs = processor.generate_qa_pairs(messages)
    
    print("💾 Saving results...")
    processor.save_results(qa_pairs)
    
    print(f"✅ Successfully generated {len(qa_pairs)} Q&A pairs at:")
    print(f"📄 {CONFIG['paths']['output']}")

# ================
# Installation
# ================
"""
1. Install requirements:
pip install parsivar sentence-transformers hdbscan tqdm requests

2. Install Ollama:
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:4b-instruct-q4_K_M

3. Run:
python chat-process.py
""" 