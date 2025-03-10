import time
import numpy as np
import torch
import psutil
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from huggingface_hub import login

login(token="YOUR HuggingFace TOKEN HERE", add_to_git_credential=True)  # ADD YOUR TOKEN HERE

# Load author's fine-tuned model from Hugging Face Hub
fine_tuned_model = SentenceTransformer("arad1367/technographics-marketing-matryoshka")

# Load test dataset for sample texts
test_dataset = load_dataset("json", data_files="test_dataset.json", split="train")
sample_texts = test_dataset["anchor"][:100]  # Use 100 samples from test set

# Define dimensions to evaluate
dimensions = [768, 512, 256, 128, 64]

# Storage analysis - theoretical calculation
def calculate_storage_requirements(num_vectors, dimensions, bytes_per_value=4):
    """Calculate storage requirements for embedding vectors in bytes.
    Using 32-bit float (4 bytes) as standard for embeddings."""
    return num_vectors * dimensions * bytes_per_value

# Results containers
storage_requirements = []
inference_times = []
memory_usages = []

# Run benchmark for each dimension
for dim in dimensions:
    # Storage analysis for 10,000 hypothetical vectors (common retrieval corpus size)
    storage_mb = calculate_storage_requirements(10000, dim) / (1024 * 1024)
    storage_requirements.append(storage_mb)
    
    # Inference time and memory usage
    start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    start_time = time.time()
    
    # Generate embeddings
    with torch.no_grad():
        embeddings = fine_tuned_model.encode(
            sample_texts, 
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        # Truncate to current dimension
        embeddings = embeddings[:, :dim]
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    
    inference_times.append((end_time - start_time) / len(sample_texts) * 1000)  # ms per sample
    memory_usages.append(end_memory - start_memory)  # Additional MB used

# Create a DataFrame for tabular results
efficiency_df = pd.DataFrame({
    'Dimension': dimensions,
    'Storage (MB for 10K vectors)': storage_requirements,
    'Inference Time (ms/sample)': inference_times,
    'Memory Usage (MB)': memory_usages,
    'Storage Reduction (%)': [0] + [(1 - dim/dimensions[0])*100 for dim in dimensions[1:]],
    'Speed Improvement (%)': [0] + [(1 - t/inference_times[0])*100 for t in inference_times[1:]]
})

print(efficiency_df)

# Visualize results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

ax1.plot(dimensions, storage_requirements)
ax1.set_xlabel('Embedding Dimension')
ax1.set_ylabel('Storage (MB) for 10K vectors')
ax1.set_title('Storage Requirements')

ax2.plot(dimensions, inference_times)
ax2.set_xlabel('Embedding Dimension')
ax2.set_ylabel('Time (ms) per sample')
ax2.set_title('Inference Time')

ax3.plot(dimensions, memory_usages)
ax3.set_xlabel('Embedding Dimension')
ax3.set_ylabel('Additional Memory (MB)')
ax3.set_title('Memory Usage')

plt.tight_layout()
plt.savefig('computational_efficiency.png')

# Calculate theoretical speedup based on dimensionality reduction
# The theoretical speedup in vector operations is proportional to dimension reduction
theoretical_speedup = [dimensions[0]/dim for dim in dimensions]
actual_speedup = [inference_times[0]/t for t in inference_times]

# Compare theoretical vs actual speedup
plt.figure(figsize=(10, 6))
plt.plot(dimensions, theoretical_speedup, label='Theoretical Speedup')
plt.plot(dimensions, actual_speedup, label='Actual Speedup')
plt.xlabel('Embedding Dimension')
plt.ylabel('Speedup Factor (vs. 768 dimension)')
plt.title('Theoretical vs. Actual Computational Speedup')
plt.legend()
plt.grid(True)
plt.savefig('speedup_comparison.png')