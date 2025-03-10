import os
import time
import random
from openai import OpenAI
from datasets import Dataset, DatasetDict
import huggingface_hub

# Initialize clients
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
hf_token = ""  # Replace with your Hugging Face token

# Configure topics and subtopics for diversity
subtopics = [
    "Technographics Fundamentals",
    "Data Collection Methods",
    "Consumer Behavior Patterns",
    "Predictive Analytics",
    "Marketing Strategy Integration",
    "Ethical Considerations",
    "Case Studies",
    "Technology Stack Analysis",
    "Cross-channel Behavior",
    "Emerging Trends"
]

def generate_qa_pairs(batch_size=10):
    """Generate unique QA pairs using OpenAI API"""
    qa_pairs = []
    attempts = 0
    max_attempts = 500  # Allow for some retries
    
    while len(qa_pairs) < 495 and attempts < max_attempts:
        try:
            # Randomly select subtopics for diversity
            selected_subtopics = random.sample(subtopics, k=3)
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a marketing analytics expert. Generate unique question-answer pairs."},
                    {"role": "user", "content": f"""
                    Generate {batch_size} unique Q&A pairs about Technographics and Predicting Consumer Behavior in Digital Marketing.
                    Focus on: {', '.join(selected_subtopics)}.
                    Requirements:
                    - Questions should be specific and varied
                    - Answers should be 2-3 sentences, technical but clear
                    - Avoid duplicate concepts
                    - Include both strategic and technical aspects
                    - Format: [Q]: question\n[A]: answer\n\n
                    """}
                ],
                temperature=0.9,
                max_tokens=2000
            )
            
            # Process the response
            content = response.choices[0].message.content
            pairs = [p.split("\n")[:2] for p in content.split("\n\n") if p.strip()]
            
            for pair in pairs:
                if len(pair) == 2 and pair[0].startswith("[Q]:") and pair[1].startswith("[A]:"):
                    question = pair[0].split(": ", 1)[1].strip()
                    answer = pair[1].split(": ", 1)[1].strip()
                    qa_pairs.append({"question": question, "answer": answer})
            
            print(f"Generated {len(qa_pairs)} pairs so far...")
            attempts += 1
            time.sleep(1.2)  # Maintain API rate limit compliance
            
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(5)
    
    return qa_pairs[:495]  # Ensure exact count

# Generate dataset
dataset = generate_qa_pairs()

# Convert to Hugging Face dataset
hf_dataset = Dataset.from_list([
    {
        "id": idx,
        "question": item["question"],
        "answer": item["answer"]
    }
    for idx, item in enumerate(dataset)
])

# Create dataset dict
dataset_dict = DatasetDict({
    "train": hf_dataset
})

# Push to Hugging Face Hub
dataset_dict.push_to_hub(
    repo_id="arad1367/technographics-qa",  # Replace with your repo
    token=hf_token,
    commit_message="Add Technographics QA dataset"
)

print("Dataset successfully uploaded to Hugging Face Hub!")