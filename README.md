# 🧠 Simple GPT-2 Implementation in JAX/Flax 🚀

This project implements a simplified version of the GPT-2 language model using JAX and [Flax](https://flax.readthedocs.io/en/latest/). Its based on the PyTorch GPT2 implementation by Andrej Karpathy - https://github.com/karpathy/ng-video-lecture

## 🌟 Features

- 📚 Trains on the TinyShakespeare dataset
- 🧮 Implements multi-head attention and transformer blocks
- 🔧 Customizable hyperparameters
- 📈 Training and validation loss tracking
- 🎭 Text generation capability

## 🛠️ Requirements

You can install the requirements using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## 🚀 Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/suryavanshi/jax_llm.git
   cd jax_llm
   ```

2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Jupyter lab:
   ```
   jupyter lab 
   ```

## 🎛️ Hyperparameters

You can adjust the following hyperparameters in the script:

- `batch_size`: Number of sequences per batch
- `block_size`: Maximum context length for predictions
- `max_iters`: Number of training iterations
- `eval_interval`: How often to evaluate the model
- `learning_rate`: Learning rate for the Adam optimizer
- `n_embd`: Embedding dimension
- `n_head`: Number of attention heads
- `n_layer`: Number of transformer layers
- `dropout`: Dropout rate

## 📊 Model Architecture

The model consists of:
- Token and position embeddings
- Multiple transformer blocks with:
  - Multi-head self-attention
  - Feed-forward neural networks
- Layer normalization
- Final linear layer for next token prediction

## 🎭 Generating Text

After training, you can generate text using the `generate` method of the `GPTLanguageModel` class.

## 📜 License

This project is [MIT](https://choosealicense.com/licenses/mit/) licensed.

## 🙏 Acknowledgements

- [Andrej Karpathy's](https://github.com/karpathy) educational content on language models
- The JAX and Flax teams for their excellent libraries

