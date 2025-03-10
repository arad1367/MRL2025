import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from scipy.stats import kendalltau, spearmanr
import time
import torch
from tabulate import tabulate
from huggingface_hub import login

login(token="YOUR HuggingFace TOKEN HERE", add_to_git_credential=True)  # ADD YOUR TOKEN HERE

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
colors = sns.color_palette("colorblind", 10)

# Load the fine-tuned model
fine_tuned_model = SentenceTransformer("arad1367/technographics-marketing-matryoshka")
base_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Load test dataset
test_dataset = load_dataset("json", data_files="test_dataset.json", split="train")

# Define dimensions to analyze - Order from smallest to largest for consistent plotting
dimensions = [64, 128, 256, 512, 768]

# 1. Ranking Consistency Analysis
def analyze_ranking_consistency(model, texts, dimensions, query_text=None):
    """Analyze ranking consistency across dimensions"""
    if query_text is None:
        # Use the first text as query if none provided
        query_text = texts[0]
        corpus_texts = texts[1:]
    else:
        corpus_texts = texts
    
    # Get full dimensional embeddings
    with torch.no_grad():
        query_embedding_full = model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
        corpus_embeddings_full = model.encode(corpus_texts, convert_to_numpy=True, normalize_embeddings=True)
    
    # Get rankings at full dimension (768)
    similarities_full = util.dot_score(query_embedding_full, corpus_embeddings_full)[0].numpy()
    ranks_full = np.argsort(-similarities_full)  # Descending order
    
    # Store results
    kendall_tau_scores = []
    spearman_scores = []
    ndcg_scores = []
    precision_at_k_scores = []
    
    # Calculate metrics for each dimension
    for dim in dimensions[1:]:  # Skip the full dimension (already calculated)
        # Truncate embeddings to current dimension
        query_embedding_trunc = query_embedding_full[:dim]
        corpus_embeddings_trunc = corpus_embeddings_full[:, :dim]
        
        # Calculate similarities
        similarities_trunc = util.dot_score(query_embedding_trunc, corpus_embeddings_trunc)[0].numpy()
        ranks_trunc = np.argsort(-similarities_trunc)
        
        # Kendall Tau rank correlation
        tau, _ = kendalltau(ranks_full, ranks_trunc)
        kendall_tau_scores.append(tau)
        
        # Spearman rank correlation
        rho, _ = spearmanr(similarities_full, similarities_trunc)
        spearman_scores.append(rho)
        
        # NDCG calculation (simplified for top 10)
        k = min(10, len(corpus_texts))
        # Get top k items from full dimension as relevant
        relevant_items = set(ranks_full[:k])
        
        # Calculate DCG for truncated dimension
        dcg = 0
        for i, item in enumerate(ranks_trunc[:k]):
            if item in relevant_items:
                # Using binary relevance (1 if in top-k of full dimension, 0 otherwise)
                dcg += 1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed
        
        # Calculate ideal DCG (just the sum of 1/log2(i+2) for i=0 to k-1)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(k))
        
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
        
        # Precision at k
        precision = len(set(ranks_trunc[:k]) & relevant_items) / k
        precision_at_k_scores.append(precision)
    
    # Add full dimension (perfect correlation with itself)
    kendall_tau_scores = [1.0] + kendall_tau_scores
    spearman_scores = [1.0] + spearman_scores
    ndcg_scores = [1.0] + ndcg_scores
    precision_at_k_scores = [1.0] + precision_at_k_scores
    
    return {
        'kendall_tau': kendall_tau_scores,
        'spearman': spearman_scores,
        'ndcg': ndcg_scores,
        'precision_at_k': precision_at_k_scores
    }

# 2. Semantic Preservation Analysis
def analyze_semantic_preservation(model, texts, dimensions):
    """Analyze how well semantic relationships are preserved across dimensions"""
    # Get embeddings at full dimension
    with torch.no_grad():
        embeddings_full = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    
    # Create similarity matrix at full dimension
    similarity_matrix_full = util.cos_sim(embeddings_full, embeddings_full).numpy()
    
    # Store results
    matrix_correlations = []
    avg_similarity_diffs = []
    cosine_similarity_preservation = []
    
    # Calculate metrics for each dimension
    for dim in dimensions:
        # Truncate embeddings to current dimension
        embeddings_trunc = embeddings_full[:, :dim]
        
        # Create similarity matrix at truncated dimension
        similarity_matrix_trunc = util.cos_sim(embeddings_trunc, embeddings_trunc).numpy()
        
        # Calculate correlation between similarity matrices (flatten the matrices)
        correlation = np.corrcoef(
            similarity_matrix_full.flatten(), 
            similarity_matrix_trunc.flatten()
        )[0, 1]
        matrix_correlations.append(correlation)
        
        # Calculate average absolute difference in similarities
        avg_diff = np.mean(np.abs(similarity_matrix_full - similarity_matrix_trunc))
        avg_similarity_diffs.append(avg_diff)
        
        # Calculate cosine similarity preservation (how similar truncated vectors are to full vectors)
        # Normalize embeddings again for fair comparison
        norm_full = np.linalg.norm(embeddings_full, axis=1, keepdims=True)
        norm_trunc_padded = np.zeros_like(embeddings_full)
        norm_trunc_padded[:, :dim] = embeddings_trunc
        
        # Calculate cosine similarity between full and truncated+padded vectors
        cos_sims = np.sum(embeddings_full * norm_trunc_padded, axis=1) / norm_full.flatten()
        cosine_similarity_preservation.append(np.mean(cos_sims))
    
    return {
        'matrix_correlation': matrix_correlations,
        'avg_similarity_diff': avg_similarity_diffs,
        'cosine_similarity_preservation': cosine_similarity_preservation
    }

# 3. Information Density Analysis
def analyze_information_density(model, texts, dimensions):
    """Analyze information density across dimensions"""
    # Get embeddings at full dimension
    with torch.no_grad():
        embeddings_full = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    
    # Store results
    embedding_norms = []
    
    # Calculate importance of each dimension once
    # Get variance of each dimension across all embeddings
    dim_variance = np.var(embeddings_full, axis=0)
    # Normalize to sum to 1
    dim_importance = dim_variance / np.sum(dim_variance)
    # Calculate cumulative importance
    cum_importance = np.cumsum(dim_importance)
    
    # Calculate metrics for each dimension
    for dim in dimensions:
        # Truncate embeddings to current dimension
        embeddings_trunc = embeddings_full[:, :dim]
        
        # Calculate average L2 norm (magnitude of the vectors)
        avg_norm = np.mean(np.linalg.norm(embeddings_trunc, axis=1))
        embedding_norms.append(avg_norm)
    
    return {
        'embedding_norms': embedding_norms,
        'dimension_importance': cum_importance
    }

# 4. Comparative Analysis with Base Model
def compare_with_base_model(fine_tuned_model, base_model, texts, dimensions):
    """Compare performance between fine-tuned and base models"""
    # Get embeddings from both models
    with torch.no_grad():
        ft_embeddings = fine_tuned_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        base_embeddings = base_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    
    # Store results
    ft_intra_similarities = []
    base_intra_similarities = []
    cosine_similarities = []
    
    # Calculate metrics for each dimension
    for dim in dimensions:
        # Truncate embeddings to current dimension
        ft_embeddings_trunc = ft_embeddings[:, :dim]
        base_embeddings_trunc = base_embeddings[:, :dim]
        
        # Calculate average similarity within each model's embeddings
        ft_sim_matrix = util.cos_sim(ft_embeddings_trunc, ft_embeddings_trunc).numpy()
        base_sim_matrix = util.cos_sim(base_embeddings_trunc, base_embeddings_trunc).numpy()
        
        # Exclude diagonal (self-similarity)
        mask = ~np.eye(len(texts), dtype=bool)
        ft_avg_sim = np.mean(ft_sim_matrix[mask])
        base_avg_sim = np.mean(base_sim_matrix[mask])
        
        ft_intra_similarities.append(ft_avg_sim)
        base_intra_similarities.append(base_avg_sim)
        
        # Calculate cosine similarity between corresponding embeddings from both models
        cos_sim_models = np.mean([
            np.dot(ft_embeddings_trunc[i], base_embeddings_trunc[i]) 
            for i in range(len(texts))
        ])
        cosine_similarities.append(cos_sim_models)
    
    return {
        'ft_intra_similarities': ft_intra_similarities,
        'base_intra_similarities': base_intra_similarities,
        'cosine_similarities': cosine_similarities
    }

# 5. Query Analysis
def analyze_query_performance(model, queries, corpus, dimensions, top_k=10):
    """Analyze query performance across dimensions"""
    # Store results
    retrieval_times = []
    avg_precision_scores = []
    recall_scores = []
    
    # Define ground truth (using full dimension as reference)
    with torch.no_grad():
        query_embeddings_full = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
        corpus_embeddings_full = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
    
    # For each query, find top-k results at full dimension
    ground_truth = {}
    for i, query_emb in enumerate(query_embeddings_full):
        scores = util.dot_score(query_emb, corpus_embeddings_full)[0].numpy()
        top_results = np.argsort(-scores)[:top_k]
        ground_truth[i] = set(top_results)
    
    # Calculate metrics for each dimension
    for dim in dimensions:
        # Truncate embeddings to current dimension
        query_embeddings_trunc = query_embeddings_full[:, :dim]
        corpus_embeddings_trunc = corpus_embeddings_full[:, :dim]
        
        # Measure retrieval time
        start_time = time.time()
        _ = util.dot_score(query_embeddings_trunc, corpus_embeddings_trunc)
        end_time = time.time()
        avg_time = (end_time - start_time) / len(queries)
        retrieval_times.append(avg_time)
        
        # Calculate precision and recall for each query
        precisions = []
        recalls = []
        
        for i, query_emb in enumerate(query_embeddings_trunc):
            scores = util.dot_score(query_emb, corpus_embeddings_trunc)[0].numpy()
            top_results = np.argsort(-scores)[:top_k]
            
            # Calculate precision and recall
            relevant_retrieved = ground_truth[i].intersection(set(top_results))
            precision = len(relevant_retrieved) / len(top_results)
            recall = len(relevant_retrieved) / len(ground_truth[i])
            
            precisions.append(precision)
            recalls.append(recall)
        
        avg_precision_scores.append(np.mean(precisions))
        recall_scores.append(np.mean(recalls))
    
    return {
        'retrieval_times': retrieval_times,
        'avg_precision': avg_precision_scores,
        'avg_recall': recall_scores
    }

# 6. Dimension Importance Analysis
def analyze_dimension_importance(model, texts, dimensions):
    """Analyze dimension importance across the matryoshka representation focusing on model dimensions"""
    # Get embeddings at full dimension
    full_embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    
    # Calculate importance of each dimension
    # Get variance of each dimension across all embeddings
    dim_variance = np.var(full_embeddings, axis=0)
    # Normalize to sum to 1
    dim_importance = dim_variance / np.sum(dim_variance)
    # Calculate cumulative importance
    cum_importance = np.cumsum(dim_importance)
    
    # Find information preserved at each model dimension
    model_thresholds = {}
    for dim in dimensions:
        if dim <= len(cum_importance):
            preserved = cum_importance[dim-1]
            model_thresholds[dim] = preserved
    
    # Plot dimension importance
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(cum_importance)+1), cum_importance)
    
    # Add horizontal and vertical lines for model dimensions
    colors = ['r', 'g', 'b', 'c', 'm']
    for i, dim in enumerate(dimensions):
        if dim <= len(cum_importance):
            preserved = model_thresholds[dim]
            color = colors[i % len(colors)]
            # Add horizontal line
            plt.axhline(y=preserved, color=color, linestyle='--', 
                      label=f"{dim} dims: {preserved:.1%}")
            # Add vertical line
            plt.axvline(x=dim, color=color, linestyle=':')
    
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Cumulative Information Preserved')
    plt.title('Dimension Importance in Matryoshka Representation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dimension_importance.png', dpi=300)
    
    return model_thresholds

# Run all analyses
def run_all_analyses():
    print("Running performance analyses...")
    
    # Prepare data
    anchors = test_dataset["anchor"]
    positives = test_dataset["positive"]
    
    # Combine texts for general analysis
    texts = anchors[:100]  # Use a subset for computational efficiency
    corpus = positives[:100]
    
    # 1. Ranking consistency
    print("Analyzing ranking consistency...")
    ranking_results = analyze_ranking_consistency(fine_tuned_model, texts, dimensions)
    
    # Display ranking consistency results
    print("\n=== Ranking Consistency Metrics ===")
    ranking_table = []
    for i, dim in enumerate(dimensions):
        ranking_table.append([
            dim, 
            f"{ranking_results['kendall_tau'][i]:.4f}",
            f"{ranking_results['spearman'][i]:.4f}",
            f"{ranking_results['ndcg'][i]:.4f}",
            f"{ranking_results['precision_at_k'][i]:.4f}"
        ])
    print(tabulate(ranking_table, 
                  headers=["Dimension", "Kendall Tau", "Spearman", "NDCG@10", "Precision@10"],
                  tablefmt="grid"))
    
    # 2. Semantic preservation
    print("\nAnalyzing semantic preservation...")
    semantic_results = analyze_semantic_preservation(fine_tuned_model, texts, dimensions)
    
    # Display semantic preservation results
    print("\n=== Semantic Preservation Metrics ===")
    semantic_table = []
    for i, dim in enumerate(dimensions):
        semantic_table.append([
            dim, 
            f"{semantic_results['matrix_correlation'][i]:.4f}",
            f"{semantic_results['cosine_similarity_preservation'][i]:.4f}",
            f"{1 - semantic_results['avg_similarity_diff'][i]:.4f}"
        ])
    print(tabulate(semantic_table, 
                  headers=["Dimension", "Matrix Correlation", "Cosine Similarity", "Similarity Preservation"],
                  tablefmt="grid"))
    
    # 3. Information density
    print("\nAnalyzing information density...")
    density_results = analyze_information_density(fine_tuned_model, texts, dimensions)
    
    # Display information density results
    print("\n=== Information Density Metrics ===")
    density_table = []
    for i, dim in enumerate(dimensions):
        density_table.append([
            dim, 
            f"{density_results['embedding_norms'][i]:.4f}"
        ])
    print(tabulate(density_table, 
                  headers=["Dimension", "Avg L2 Norm"],
                  tablefmt="grid"))
    
    # 4. Dimension importance analysis - USING THE REVISED FUNCTION
    print("\nAnalyzing dimension importance...")
    model_thresholds = analyze_dimension_importance(fine_tuned_model, texts, dimensions)
    
    # Display dimension importance results
    print("\n=== Information Preserved at Model Dimensions ===")
    thresholds_table = [[dim, f"{preserved:.2%}"] for dim, preserved in model_thresholds.items()]
    print(tabulate(thresholds_table, 
                  headers=["Dimension", "Information Preserved"],
                  tablefmt="grid"))
    
    # 5. Query performance
    print("\nAnalyzing query performance...")
    query_results = analyze_query_performance(fine_tuned_model, anchors[:20], positives, dimensions)
    
    # Display query performance results
    print("\n=== Query Performance Metrics ===")
    query_table = []
    for i, dim in enumerate(dimensions):
        query_table.append([
            dim, 
            f"{query_results['retrieval_times'][i]*1000:.2f}",
            f"{query_results['avg_precision'][i]:.4f}",
            f"{query_results['avg_recall'][i]:.4f}"
        ])
    print(tabulate(query_table, 
                  headers=["Dimension", "Retrieval Time (ms)", "Precision@10", "Recall@10"],
                  tablefmt="grid"))
    
    # Compare with baseline NDCG@10 from your results
    baseline_ndcg = {
        64: 0.4148,
        128: 0.4401,
        256: 0.4710,
        512: 0.4698,
        768: 0.4654,
    }
    
    finetuned_ndcg = {
        64: 0.4740,
        128: 0.5075,
        256: 0.5158, 
        512: 0.5301,
        768: 0.5295,
    }
    
    # Create improvement percentage
    ndcg_improvement = {
        dim: ((finetuned_ndcg[dim] - baseline_ndcg[dim]) / baseline_ndcg[dim]) * 100 
        for dim in dimensions
    }
    
    # Display NDCG comparison
    print("\n=== NDCG@10 Comparison with Baseline ===")
    ndcg_table = []
    for dim in dimensions:
        ndcg_table.append([
            dim,
            f"{baseline_ndcg[dim]:.4f}",
            f"{finetuned_ndcg[dim]:.4f}",
            f"{ndcg_improvement[dim]:.2f}%"
        ])
    print(tabulate(ndcg_table, 
                  headers=["Dimension", "Baseline NDCG@10", "Fine-tuned NDCG@10", "Improvement (%)"],
                  tablefmt="grid"))
    
    # Plotting
    print("\nGenerating visualizations...")
    
    # Plot ranking consistency metrics
    plt.figure(figsize=(10, 8))
    plt.plot(dimensions, ranking_results['kendall_tau'], 'o-', label='Kendall Tau')
    plt.plot(dimensions, ranking_results['spearman'], 's-', label='Spearman Correlation')
    plt.plot(dimensions, ranking_results['ndcg'], '^-', label='NDCG@10')
    plt.plot(dimensions, ranking_results['precision_at_k'], 'D-', label='Precision@10')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Score')
    plt.title('Ranking Consistency Across Dimensions')
    plt.legend()
    plt.grid(True)
    # Explicitly set x-ticks to the dimensions
    plt.xticks(dimensions)
    # Reverse x-axis to show dimensions from largest to smallest (left to right)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig('ranking_consistency.png', dpi=300)
    
    # Plot semantic preservation metrics
    plt.figure(figsize=(10, 8))
    plt.plot(dimensions, semantic_results['matrix_correlation'], 'o-', label='Similarity Matrix Correlation')
    plt.plot(dimensions, semantic_results['cosine_similarity_preservation'], 's-', label='Cosine Similarity Preservation')
    plt.plot(dimensions, [1 - x for x in semantic_results['avg_similarity_diff']], '^-', label='Similarity Preservation (1-diff)')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Score')
    plt.title('Semantic Preservation Across Dimensions')
    plt.legend()
    plt.grid(True)
    # Explicitly set x-ticks to the dimensions
    plt.xticks(dimensions)
    # Reverse x-axis to show dimensions from largest to smallest (left to right)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig('semantic_preservation.png', dpi=300)
    
    # Plot information density metrics
    plt.figure(figsize=(10, 8))
    plt.plot(dimensions, density_results['embedding_norms'], 'o-')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Average L2 Norm')
    plt.title('Embedding Magnitude Across Dimensions')
    plt.grid(True)
    # Explicitly set x-ticks to the dimensions
    plt.xticks(dimensions)
    # Reverse x-axis to show dimensions from largest to smallest (left to right)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig('embedding_magnitude.png', dpi=300)
    
    # Plot query performance
    plt.figure(figsize=(12, 8))
    
    # Plot retrieval times
    plt.subplot(2, 1, 1)
    plt.plot(dimensions, np.array(query_results['retrieval_times']) * 1000, 'o-')  # Convert to ms
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Average Retrieval Time (ms)')
    plt.title('Query Retrieval Time Across Dimensions')
    plt.grid(True)
    # Explicitly set x-ticks to the dimensions
    plt.xticks(dimensions)
    # Reverse x-axis to show dimensions from largest to smallest (left to right)
    plt.gca().invert_xaxis()
    
    # Plot precision and recall
    plt.subplot(2, 1, 2)
    plt.plot(dimensions, query_results['avg_precision'], 'o-', label='Average Precision@10')
    plt.plot(dimensions, query_results['avg_recall'], 's-', label='Average Recall@10')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Score')
    plt.title('Retrieval Quality Across Dimensions')
    plt.legend()
    plt.grid(True)
    # Explicitly set x-ticks to the dimensions
    plt.xticks(dimensions)
    # Reverse x-axis to show dimensions from largest to smallest (left to right)
    plt.gca().invert_xaxis()
    
    plt.tight_layout()
    plt.savefig('query_performance.png', dpi=300)
    
    # Plot NDCG comparison
    plt.figure(figsize=(10, 8))
    x = np.arange(len(dimensions))
    width = 0.35
    
    plt.bar(x - width/2, [baseline_ndcg[dim] for dim in dimensions], width, label='Baseline')
    plt.bar(x + width/2, [finetuned_ndcg[dim] for dim in dimensions], width, label='Fine-tuned')
    
    plt.xlabel('Embedding Dimension')
    plt.ylabel('NDCG@10 Score')
    plt.title('NDCG@10 Comparison: Baseline vs. Fine-tuned')
    plt.xticks(x, dimensions)
    
    # Add improvement percentages as text
    for i, dim in enumerate(dimensions):
        plt.annotate(f"{ndcg_improvement[dim]:.1f}%", 
                    xy=(i, finetuned_ndcg[dim]), 
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('ndcg_comparison.png', dpi=300)
    
    # Create a comprehensive results dictionary
    results = {
        'ranking_consistency': ranking_results,
        'semantic_preservation': semantic_results,
        'information_density': density_results,
        'query_performance': query_results,
        'model_thresholds': model_thresholds,
        'ndcg_improvement': ndcg_improvement
    }
    
    # Save numerical results to CSV
    results_df = pd.DataFrame({
        'Dimension': dimensions,
        'Kendall_Tau': ranking_results['kendall_tau'],
        'Spearman': ranking_results['spearman'],
        'NDCG@10': ranking_results['ndcg'],
        'Precision@10': ranking_results['precision_at_k'],
        'Matrix_Correlation': semantic_results['matrix_correlation'],
        'Similarity_Preservation': semantic_results['cosine_similarity_preservation'],
        'Retrieval_Time_ms': np.array(query_results['retrieval_times']) * 1000,
        'Retrieval_Precision': query_results['avg_precision'],
        'Retrieval_Recall': query_results['avg_recall'],
        'Baseline_NDCG': [baseline_ndcg[dim] for dim in dimensions],
        'Finetuned_NDCG': [finetuned_ndcg[dim] for dim in dimensions],
        'NDCG_Improvement': [ndcg_improvement[dim] for dim in dimensions]
    })
    
    results_df.to_csv('performance_metrics.csv', index=False)
    print("\nAnalyses complete. Results saved to files.")
    
    return results

# Run the analyses
if __name__ == "__main__":
    results = run_all_analyses()