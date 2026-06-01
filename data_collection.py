import arxiv
import datetime
import pandas as pd
from tqdm import tqdm

def collect_agent_corpus():
    # 1. Define your exact boundaries
    start_date = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    end_date = datetime.datetime(2026, 4, 30, 23, 59, 59, tzinfo=datetime.timezone.utc)
    
    keywords = [
        "llm agent", "agentic", "tool use", "agent memory", 
        "agent benchmark", "computer-use", "tool learning", "rag"
    ]
    
    # 2. Construct a strict API-level query 
    # Adding the specific date boundaries directly to the arXiv query string
    date_query = 'submittedDate:[202401010000 TO 202604302359]'
    categories = '(cat:cs.CL OR cat:cs.AI OR cat:cs.LG)'
    keyword_query = '(' + ' OR '.join([f'all:"{kw}"' for kw in keywords]) + ')'
    
    full_query = f"{categories} AND {keyword_query} AND {date_query}"
    
    print(f"Querying arXiv API with: {full_query}\n")
    
    # 3. Initialize the arXiv client
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3.0,
        num_retries=5
    )
    
    # FIX: Sort by Relevance to get the strongest keyword matches first
    search = arxiv.Search(
        query=full_query,
        max_results=3000, 
        sort_by=arxiv.SortCriterion.Relevance 
    )
    
    corpus = []
    
    # 4. Fetch and double-check results locally
    for result in tqdm(client.results(search), desc="Fetching papers"):
        pub_date = result.published
        
        # Double check the date locally just in case the API search leaks
        if start_date <= pub_date <= end_date:
            title_lower = result.title.lower()
            summary_lower = result.summary.lower()
            
            # Strict keyword check rule to ensure high relevance
            if any(kw in title_lower or kw in summary_lower for kw in keywords):
                corpus.append({
                    "arxiv_id": result.get_short_id(),
                    "title": result.title,
                    "abstract": result.summary,
                    "published": result.published.strftime("%Y-%m-%d"),
                    "pdf_url": result.pdf_url,
                    "categories": result.categories,
                    "primary_category": result.primary_category
                })

    # 5. Save the metadata corpus
    df = pd.DataFrame(corpus)
    df = df.drop_duplicates(subset=["arxiv_id"])
    
    print(f"\nSuccessfully collected {len(df)} papers matching your criteria.")
    df.to_json("eval/corpus_metadata.json", orient="records", indent=4)
    return df

if __name__ == "__main__":
    corpus_df = collect_agent_corpus()