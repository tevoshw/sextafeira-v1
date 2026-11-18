## Tokenizer

This project uses the pre-trained **[Tucano2-0.6B-Base](https://huggingface.co/Polygl0t/Tucano2-0.6B-Base)** tokenizer from [Polygl0t](https://huggingface.co/Polygl0t). **No tokenizer was trained from scratch**; I reuse the existing one as-is.

### Why this tokenizer?

- **Portuguese-native:** it was built for Portuguese text, so it handles accents, contractions, and common word forms efficiently, with fewer tokens per sentence than a generic multilingual tokenizer.
- **Vocabulary size:** 49,152 tokens (`vocab_size = 49152`), which keeps the embedding matrix and output layer compact while still giving good coverage of the language.
- **Ready to use:** since it is already trained, there is no extra tokenizer-training step, and the vocabulary stays consistent with the original Tucano2 models.


> **Note:** because the tokenizer is pre-trained, the model's `vocab_size` must match it (49,152). Changing the vocabulary would invalidate the token-to-embedding mapping.