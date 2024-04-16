from utils import generate_response, get_summary, generate_embeddings, get_web_data
import argparse
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def main():
    parser = argparse.ArgumentParser(description="Compare two URLs")
    parser.add_argument("--urls", nargs="+", type=str, help="URLs to compare")
    parser.add_argument(
        "--model", default="../model/model.pkl", help="Path to the clustering model"
    )
    args = parser.parse_args()

    print("Extracting data from the web...")
    web_data = [get_web_data(url) for url in args.urls]
    if None in web_data:
        print("Error fetching web data")
        return

    print("Generating openai responses...")
    response_data = [generate_response(data) for data in web_data]
    if None in response_data:
        print("Error generating responses")
        return

    summary_data = [get_summary(response) for response in response_data]
    if None in summary_data:
        print("Error generating summaries")
        return

    def calc_dist(args, embeddings):
        _min_similarity = 0.9
        # Load the model and compute the dist
        model = joblib.load(args.model)
        _dist = model.transform(embeddings)
        # Return the normalized cosine similarity
        res = cosine_similarity(_dist)[0][1]
        return (res - _min_similarity) / (1 - _min_similarity)

    print("Generating embeddings and calc the similarity...")
    embeddings = generate_embeddings(summary_data)
    res = calc_dist(args, embeddings)

    print(f"The structural similarity is {res:.3f}")
    return res


if __name__ == "__main__":
    main()
