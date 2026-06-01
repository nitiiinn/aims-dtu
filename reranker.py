import pandas as pd
import json

def rank_and_filter_corpus(input_file="eval/corpus_metadata.json", output_file="eval/corpus_metadata_filtered.json", top_n=800):
    # 1. Load the raw metadata
    try:
        df = pd.read_json(input_file)
        print(f"Loaded {len(df)} papers for scoring.")
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return

    # 2. Define weighted keywords (higher number = more important)
    keyword_weights = {
        "llm agent": 5,
        "computer-use": 5,
        "agent benchmark": 5,
        "agent memory": 4,
        "tool learning": 4,
        "tool use": 3,
        "agentic": 3,
        "rag": 1  # Given a low weight because it is a very broad term
    }

    # 3. Define the scoring function
    def calculate_score(row):
        score = 0
        title = str(row.get('title', '')).lower()
        abstract = str(row.get('abstract', '')).lower()

        for kw, weight in keyword_weights.items():
            # Title matches are worth 3x the base weight
            if kw in title:
                score += (weight * 3)
            
            # Abstract matches get the base weight
            # We count how many times it appears to reward dense abstracts
            count_in_abstract = abstract.count(kw)
            if count_in_abstract > 0:
                # Cap the abstract multiplier at 3 so keyword stuffing doesn't skew it
                score += (weight * min(count_in_abstract, 3))
                
        return score

    # 4. Apply scores and sort
    print("Scoring papers based on keyword density and location...")
    df['relevance_score'] = df.apply(calculate_score, axis=1)
    
    # Sort by score descending, then by date descending as a tie-breaker
    df_sorted = df.sort_values(by=['relevance_score', 'published'], ascending=[False, False])

    # 5. Slice the top N papers
    # Filter out papers with a score of 0 just in case
    df_valid = df_sorted[df_sorted['relevance_score'] > 0]
    top_papers = df_valid.head(top_n)

    # 6. Save the final list
    # Drop the temporary score column to keep the schema clean for your agent
    final_output = top_papers.drop(columns=['relevance_score'])
    final_output.to_json(output_file, orient="records", indent=4)
    
    print(f"\nSuccess! Kept the top {len(top_papers)} most relevant papers.")
    print(f"Lowest score in the top {len(top_papers)}: {df_valid['relevance_score'].iloc[len(top_papers)-1] if len(top_papers) == top_n else 'N/A'}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    rank_and_filter_corpus()