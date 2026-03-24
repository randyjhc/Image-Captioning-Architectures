🏗️ Architecture
Encoder
Pretrained ViT (vit_base_patch16_224) from timm
Outputs patch/token embeddings instead of classification logits
CLS token is removed (only patch tokens used)
Linear projection maps encoder dimension → decoder dimension

Output shape:

[B, N_patches, d_model]
Decoder
Transformer decoder (nn.TransformerDecoder)
Components:
Token embedding
Positional embedding
Multi-head self-attention (causal masked)
Cross-attention with image features
Feedforward layers
Final linear layer maps to vocabulary logits

Input:

[B, T] (caption tokens)

Output:

[B, T, vocab_size]
Full Model

Pipeline:

Image → ViT Encoder → Visual Tokens → Transformer Decoder → Caption
🔧 Key Design Decisions

1. Pretrained Backbone Integration
   Used pretrained ViT to leverage strong visual features
   Removed classification head
   Used patch embeddings for fine-grained cross-attention
2. Positional Encoding Strategy
   Reused pretrained ViT positional embeddings
   Input images resized to 224×224 to match pretrained configuration
3. Cross-Attention Mechanism
   Decoder attends to visual tokens via cross-attention
   No causal masking applied to image memory
   Standard causal masking applied to caption tokens
4. Dimension Alignment
   Added linear projection layer:
   ViT_dim → d_model (decoder)
5. Masking Strategy
   Causal mask prevents decoder from seeing future tokens
   Padding mask ignores <PAD> tokens in captions
   🧪 Training Setup
   Dataset
   Flickr8k dataset
   Each image paired with 5 captions
   Data split by image (train / val / test)
   Preprocessing
   Lowercasing and punctuation removal
   Tokenization by whitespace
   Special tokens:
   <SOS> start
   <EOS> end
   <PAD> padding
   <UNK> unknown
   Training Objective
   Cross-entropy loss with teacher forcing:
   input = captions[:, :-1]
   target = captions[:, 1:]
   📊 Results (Sanity Check)

After a short training run:

Training loss decreased:
~9.2 → ~2.9
Model generates reasonable captions:

Reference:

A black dog leaps over a log.

Generated:

<SOS> a black and white dog is jumping over a tree <EOS>

This demonstrates:

Successful encoder-decoder integration
Functional cross-attention
Correct sequence generation behavior
📁 Relevant Files
image_captioning_architectures/
│
├── models/
│ ├── vit_encoder.py
│ ├── decoder.py
│ └── model.py
│
├── data/
│ ├── flickr8k_dataset.py
│ └── vocab.py
│
└── scripts/
└── train.py

🧠 Summary

This module successfully establishes the backbone of the image captioning system by bridging visual and textual modalities using a Transformer-based architecture. The model demonstrates correct learning behavior and produces semantically meaningful captions after minimal training.
