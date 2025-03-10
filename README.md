# Optimizing Embedding Models for Operational Performance With MRL

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pejman-ebrahimi-4a60151a7/)
[![HuggingFace](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=for-the-badge)](https://huggingface.co/arad1367)
[![Website](https://img.shields.io/badge/Website-008080?style=for-the-badge&logo=About.me&logoColor=white)](https://arad1367.github.io/pejman-ebrahimi/)
[![University](https://img.shields.io/badge/University-00205B?style=for-the-badge&logo=academia&logoColor=white)](https://www.uni.li/pejman.ebrahimi?set_language=en)

This repository contains the code and resources for the research paper "Optimizing Embedding Models for Operational Performance". The project explores the use of Matryoshka Representation Learning (MRL) to fine-tune embedding models for improved efficiency across multiple embedding dimensions.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Synthetic Data Generation](#synthetic-data-generation)
  - [Model Fine-tuning](#model-fine-tuning)
  - [Performance Analysis](#performance-analysis)
  - [Computational Efficiency Analysis](#computational-efficiency-analysis)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Overview

This research investigates a novel approach to optimizing embedding models for information retrieval in domain-specific contexts, particularly focusing on the Technographics marketing domain. By leveraging Matryoshka Representation Learning (MRL), we fine-tune existing embedding models to create flexible, multi-resolution embeddings that maintain semantic integrity across different dimensions (768, 512, 256, 128, and 64).

Our approach allows for dynamic trade-offs between performance and computational efficiency, enabling systems to adapt to different operational constraints without requiring multiple specialized models.

## Repository Structure

```
.
├── Data_Json/                 # JSON formatted datasets
├── main.ipynb                 # Jupyter notebook with complete fine-tuning process
├── synthetic_data.py          # Script for generating synthetic QA data
├── computation_efficiency.py  # Script for analyzing computational efficiency
├── performance_analysis.py    # Script for analyzing model performance
└── README.md                  # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/arad1367/MRL2025.git
cd MRL2025

# Install required packages
pip install torch==2.1.2 tensorboard sentence-transformers>=3 datasets==2.19.1 transformers==4.41.2
```

## Dataset

We use a specialized dataset of question-answer pairs in the Technographics marketing domain. The dataset contains 495 pairs covering various subtopics:

- Technographics Fundamentals
- Data Collection Methods
- Consumer Behavior Patterns
- Predictive Analytics
- Marketing Strategy Integration
- Ethical Considerations
- Case Studies
- Technology Stack Analysis
- Cross-channel Behavior
- Emerging Trends

The dataset is available on Hugging Face: [arad1367/technographics-qa](https://huggingface.co/datasets/arad1367/technographics-qa)

You can also generate similar synthetic data using the `synthetic_data.py` script.

## Usage

### Synthetic Data Generation

To generate synthetic QA pairs for fine-tuning:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Run the script
python synthetic_data.py
```

### Model Fine-tuning

The complete fine-tuning process is detailed in `main.ipynb`. Key steps include:

1. Loading and preprocessing the dataset
2. Evaluating the baseline model performance
3. Implementing Matryoshka Representation Learning (MRL)
4. Training the model with a multi-dimensional loss function
5. Evaluating the fine-tuned model

### Performance Analysis

To analyze the performance of the fine-tuned model across different dimensions:

```bash
python performance_analysis.py
```

This generates comprehensive metrics and visualizations for:
- Ranking consistency
- Semantic preservation
- Information density
- Query performance
- Dimension importance

### Computational Efficiency Analysis

To evaluate the computational efficiency benefits:

```bash
python computation_efficiency.py
```

This script benchmarks the model across different dimensions for:
- Storage requirements
- Inference time
- Memory usage
- Theoretical vs. actual speedup

## Results

Our fine-tuned Matryoshka embedding model demonstrates significant improvements over the baseline across all dimensions (768, 512, 256, 128, and 64). The results show that:

- The fine-tuned model consistently outperforms the baseline model across all dimensions
- Smaller dimensions (128, 64) maintain strong performance while reducing computational requirements
- Performance metrics including NDCG@10, precision, recall, and ranking consistency are enhanced
- The model achieves substantial improvements in information retrieval tasks within the Technographics domain

The fine-tuned model is available on Hugging Face: [arad1367/technographics-marketing-matryoshka](https://huggingface.co/arad1367/technographics-marketing-matryoshka)

## Citation

If you use this code or the models in your research, please cite our work:

```bibtex
@article{ebrahimi2025optimizing,
  title={Optimizing Embedding Models for Operational Performance},
  author={Ebrahimi, Pejman},
  journal={},
  year={2025},
  publisher={}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

**Pejman Ebrahimi**  
Postdoctoral Researcher in Data Science and AI  
Department of Information Systems & Computer Science  
University of Liechtenstein  

📧 Email: pejman.ebrahimi@uni.li  
🔗 Website: [arad1367.github.io/pejman-ebrahimi](https://arad1367.github.io/pejman-ebrahimi/)  
🤗 Hugging Face: [arad1367](https://huggingface.co/arad1367)  
🏛️ University Profile: [uni.li/pejman.ebrahimi](https://www.uni.li/pejman.ebrahimi?set_language=en)