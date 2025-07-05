"""
SQuAD Loader: Download, parse, and flatten SQuAD v1.1 or v2.0 into QANet-ready examples.
- Downloads SQuAD if not present
- Handles both v1.1 and v2.0 (with or without unanswerable questions)
- Produces flat list: context, question, answer, answer_start, is_impossible
"""
import os
import json
import requests
from typing import List, Dict

SQUAD_URLS = {
    "v1.1": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "v2.0": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
}


def download_squad(version="v1.1", dest_dir="assets"):
    os.makedirs(dest_dir, exist_ok=True)
    url = SQUAD_URLS[version]
    fname = os.path.join(dest_dir, os.path.basename(url))
    if not os.path.exists(fname):
        tf.get_logger().info(f"Downloading SQuAD {version} to {fname} ...")
        r = requests.get(url)
        with open(fname, "w", encoding="utf-8") as f:
            f.write(r.text)
    return fname


def load_and_flatten_squad(json_path: str, version="v1.1") -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        squad = json.load(f)
    examples = []
    for article in squad["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question = qa["question"]
                qid = qa["id"]
                is_impossible = qa.get("is_impossible", False)
                if is_impossible:
                    # SQuAD v2.0 unanswerable
                    examples.append({
                        "context": context,
                        "question": question,
                        "answers": [],
                        "answer_starts": [],
                        "is_impossible": True,
                        "id": qid
                    })
                else:
                    answers = [a["text"] for a in qa["answers"]]
                    answer_starts = [a["answer_start"] for a in qa["answers"]]
                    examples.append({
                        "context": context,
                        "question": question,
                        "answers": answers,
                        "answer_starts": answer_starts,
                        "is_impossible": False,
                        "id": qid
                    })
    return examples

if __name__ == "__main__":
    # Demo: Download and parse both SQuAD versions
    for ver in ["v1.1", "v2.0"]:
        path = download_squad(version=ver)
        exs = load_and_flatten_squad(path, version=ver)
        tf.get_logger().info(f"Loaded {len(exs)} examples from SQuAD {ver}")
        tf.get_logger().info(f"Sample: {exs[0]}")
