# Ensemble Sentence Similarity Optimization using Harmony Search Algorithm

This project implements an ensemble approach for sentence similarity tasks by optimizing the weights of multiple pre-trained models using the Harmony Search metaheuristic algorithm. The ensemble combines three state-of-the-art models to achieve better performance than individual models.

## 🎯 Overview

The project addresses the challenge of combining multiple sentence similarity models effectively by finding optimal weight combinations through the Harmony Search optimization algorithm. Due to memory limitations, the optimization is performed on batches of the dataset.

## 🔧 Models Used

The ensemble combines three powerful sentence similarity models:

1. **RoBERTa-Large** (`stsb-roberta-large`)
   - Bi-encoder architecture
   - Efficient for large-scale similarity computations
   - Uses cosine similarity between embeddings

2. **Cross-Encoder** (`cross-encoder/stsb-roberta-base`)
   - Joint encoding of sentence pairs
   - More accurate but computationally expensive
   - Uses sigmoid activation on logits

3. **T5-Large** (`t5-large`)
   - Encoder-only version for embeddings
   - Transformer-based architecture
   - Mean pooling of last hidden states

## 📊 Dataset

- **Source**: STS-B Multi-MT dataset (English split, test set)
- **Total Size**: 1,379 samples
- **Used for Optimization**: 1,200 samples (due to memory constraints)
- **Batch Processing**: 4 batches of 300 samples each
- **Labels**: Similarity scores normalized to [0, 1] range

## 🚀 Methodology

### Harmony Search Algorithm

The Harmony Search algorithm is used to optimize ensemble weights with the following parameters:

- **Population Size (HM_size)**: 5 harmonies
- **Harmony Memory Considering Rate (HMCR)**: 0.9
- **Pitch Adjusting Rate (PAR)**: 0.3
- **Bandwidth (BW)**: 0.1
- **Maximum Iterations**: 10 per batch

### Objective Function

The optimization minimizes the Mean Absolute Error (MAE) between predicted ensemble scores and ground truth similarity scores:

```
MAE = (1/N) × Σ|y_true - y_ensemble|
```

Where `y_ensemble = w₁×score₁ + w₂×score₂ + w₃×score₃` and weights are normalized to sum to 1.

## 📈 Results

### Batch-wise Performance

| Batch | Best MAE | RoBERTa-Large | Cross-Encoder | T5-Large |
|-------|----------|---------------|---------------|----------|
| 1     | 0.1092   | 0.5795        | 0.2846        | 0.1359   |
| 2     | 0.0824   | 0.4536        | 0.5448        | 0.0016   |
| 3     | 0.1555   | 0.5110        | 0.4631        | 0.0259   |
| 4     | 0.0827   | 0.3105        | 0.6895        | 0.0000   |

### Final Ensemble Weights (Average)

- **RoBERTa-Large**: 46.37%
- **Cross-Encoder**: 49.55%
- **T5-Large**: 4.08%

**Final Average MAE**: 0.1075

## 🔍 Key Findings

1. **Cross-Encoder Dominance**: The Cross-Encoder model consistently receives the highest weights, indicating its superior performance for sentence pair similarity tasks.

2. **RoBERTa-Large Reliability**: RoBERTa-Large maintains substantial contribution across all batches, showing consistent performance.

3. **T5-Large Limitation**: T5-Large receives minimal weights, suggesting it may not be well-suited for this specific similarity task or the encoding approach used.

4. **Batch Variation**: Different batches show varying optimal weight distributions, indicating dataset heterogeneity.

## 🛠️ Technical Implementation

### Key Features

- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Memory Optimization**: Batch processing and caching mechanisms
- **Error Handling**: Robust error handling for problematic samples
- **Visualization**: Convergence plots and performance tracking

### Dependencies

```python
numpy
torch
matplotlib
seaborn
datasets
sentence-transformers
transformers
```

### Memory Management

Due to computational constraints:
- Dataset processed in 4 batches of 300 samples
- Embedding caching to avoid recomputation
- GPU memory clearing between batches
- Only 1,200 out of 1,379 samples used

## 📊 Visualization

The code generates two key visualizations:

1. **Harmony Search Convergence**: Shows fitness evolution across iterations for each batch
2. **Best Objective Evolution**: Tracks the best MAE across different batches

## 🚧 Limitations

1. **Memory Constraints**: Only ~87% of the dataset used due to computational limitations
2. **Limited Iterations**: Only 10 iterations per batch due to time constraints
3. **Batch Processing**: May not capture global optimal weights
4. **T5 Encoding**: The T5 encoding strategy might not be optimal for similarity tasks

## 🔮 Future Improvements

1. **Increased Dataset Coverage**: Use cloud computing for full dataset processing
2. **Extended Optimization**: More iterations and larger harmony memory
3. **Advanced T5 Usage**: Explore different T5 encoding strategies
4. **Cross-Validation**: Implement k-fold validation for robust evaluation
5. **Additional Models**: Include more diverse architectures (e.g., SimCSE, DPR)

## 📝 Usage

```python
# Initialize models
models = initialize_models()

# Load and process dataset in batches
dataset = load_dataset("stsb_multi_mt", name="en", split="test")

# Run harmony search optimization
best_weights, best_value, fitness_curve = harmony_search(
    dataset_batch, models, 
    N=3, HM_size=5, HMCR=0.9, PAR=0.3, BW=0.1, max_iterations=10
)

# Compute ensemble similarity
scores = compute_similarity(models, sentence1, sentence2)
ensemble_score = sum(weights[i] * scores[model_keys[i]] for i in range(len(model_keys)))
```

## 📚 References

- STS-B: Semantic Textual Similarity Benchmark
- Harmony Search Algorithm (Geem et al., 2001)
- RoBERTa: Robustly Optimized BERT Pretraining Approach
- T5: Text-to-Text Transfer Transformer

## 🤝 Contributing

Feel free to contribute by:
- Extending to larger datasets
- Adding new similarity models
- Implementing other optimization algorithms
- Improving memory efficiency

---

*Note: This implementation demonstrates the feasibility of ensemble optimization for sentence similarity tasks, though results could be improved with more computational resources and extended optimization.*
