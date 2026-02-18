# MiniTransformer AI

### A from-scratch Transformer Language Model in pure C# — CPU only, no external AI APIs

-----

## What It Is

A complete, working transformer language model written entirely in C# with:

- **Full transformer architecture** — multi-head causal self-attention, feed-forward (GELU), residual connections, pre-LayerNorm, sinusoidal positional encoding
- **Bigram+character tokenizer** — builds vocabulary from your corpus automatically
- **Adam optimizer** with gradient clipping
- **Backpropagation** fully hand-coded (no autograd library)
- **Parallel matrix multiplication** via `Parallel.For` to use all CPU cores
- **Top-K sampling** with temperature for generation
- **Save/load** model weights to disk (binary format)
- **Interactive console UI** — train, chat, configure, all in one terminal

-----

## Requirements

- [.NET 8 SDK](https://dotnet.microsoft.com/en-us/download) (or .NET 6/7)
- Windows / Linux / macOS (CPU only)

-----

## Build & Run

```bash
# 1. Navigate to the project folder
cd MiniTransformerAI

# 2. Build (optimized release)
dotnet build -c Release

# 3. Run
dotnet run -c Release
```

Or build a standalone Windows .exe:

```bash
dotnet publish -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true
# Output: bin\Release\net8.0\win-x64\publish\MiniTransformerAI.exe
```

-----

## Usage

When you run the program you get a menu:

```
[1] Train on corpus file      ← point at a .txt file, trains the model
[2] Train on typed text       ← paste text directly, then type END
[3] Chat / Generate           ← talk to your trained model
[4] Show model info           ← layer count, params, vocab size
[5] Create new model          ← custom size (bigger = smarter, slower)
[6] Save model                ← saves weights to ./model_weights/
[7] Exit
```

### Quick Start

1. Run the program → choose **[1]**
1. Press Enter (uses the bundled `corpus.txt`)
1. Set epochs to `30`, sequence length `32`
1. After training → choose **[3]** to chat

### Training on Your Own Text

Put any `.txt` file next to the executable and train on it.  
Larger and more varied text = smarter model.  
Good sources: books (Project Gutenberg), Wikipedia dumps, code files, conversation logs.

### Chat Commands

Inside chat mode:

```
/temp 0.7    — lower = more focused, higher = more creative
/topk 20     — limit sampling to top 20 tokens
/len 300     — max tokens to generate
/quit        — exit chat
```

-----

## Model Sizes

|Preset|dModel|Layers|Heads|dFF |Params|Speed           |
|------|------|------|-----|----|------|----------------|
|Tiny  |64    |2     |2    |256 |~0.5M |Fast            |
|Small |128   |4     |4    |512 |~3M   |Good            |
|Medium|256   |6     |8    |1024|~15M  |Slow            |
|Large |512   |8     |8    |2048|~80M  |Very slow on CPU|

Default is **Small** — good for learning on modest hardware.

-----

## Architecture Details

```
Input tokens
    │
    ▼
Token Embeddings (VocabSize × dModel)
    +
Positional Encoding (sinusoidal)
    │
    ▼
┌─────────────────────────────────────────┐
│ Transformer Block × N                  │
│                                         │
│  x = x + MultiHeadAttention(LN(x))     │
│  x = x + FFN(LN(x))                    │
│                                         │
│  MHA: Q,K,V projections                 │
│       Causal (masked) dot-product attn  │
│       Top-K softmax per head            │
│       Output projection                 │
│                                         │
│  FFN: Linear → GELU → Linear            │
└─────────────────────────────────────────┘
    │
    ▼
LM Head (dModel × VocabSize)
    │
    ▼
Softmax → Cross-Entropy Loss (training)
Top-K Temperature Sampling (inference)
```

-----

## Files Created After Training

```
model_weights/        ← binary weight files
vocab.json            ← tokenizer vocabulary
corpus.txt            ← sample training text (replace with your own)
```

-----

## Tips

- **More data = better model.** Try training on a full novel or Wikipedia article.
- **More epochs** help on small datasets. Watch for loss going below 1.5.
- **Temperature 0.5–0.9** gives best results — too low = repetitive, too high = gibberish.
- The model learns to continue/complete text in the style of whatever you trained it on.
- There are **no content restrictions** baked in — it generates whatever it learned from your data.

-----

*Built from scratch — no PyTorch, no TensorFlow, no ML.NET, no APIs.*
