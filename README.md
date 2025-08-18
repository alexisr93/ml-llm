# ml-llm

A learning repository focused on exploring **Machine Learning** with an emphasis on **Large Language Models**. This repository will contain Python scripts, and experiments, built primarily using **PyTorch**.

## Overview

This repo is intended for:

- Hands-on learning and experimentation with machine learning concepts.
- Implementing and training models from scratch.
- Exploring the mechanics and applications of large language models.
- Testing different architectures, training techniques, and optimizations in PyTorch.

## Learning Goals

- Understand the fundamentals of machine learning and deep learning.
- Explore transformer architectures and attention mechanisms.
- Implement and train small-scale language models using PyTorch.
- Experiment with model optimization techniques.

## Contents

- **`byte_pair_encoding.py`**
A minimal, standalone implementation of a Byte Pair Encoding (BPE) tokenizer. Trains a tokenizer directly on a small corpus and merges character pairs into subword tokens.

- **`bpe_tokenizer_pipeline.py`**
A full BPE tokenizer training pipeline built around the Tiny Shakespeare dataset. Handles corpus download, learning merges, building vocab/token mappings, and preparing tokenized train/validation splits for model training.

- **`wordpiece_tokenizer.py`**
Implements a WordPiece subword tokenizer that can be trained on a custom corpus. Provides functionality to encode text into subword tokens using a greedy longest-match-first approach, decode token sequences back to text, save/load the vocabulary, and handles unknown and padding tokens.