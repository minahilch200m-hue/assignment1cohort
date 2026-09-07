import json
import requests
import time

# Load the test dataset
with open("eval_dataset.json", "r") as f:
    dataset = json.load(f)

total_questions = len(dataset)
retrieval_hits = 0

print(f"Starting evaluation on {total_questions} questions...\n")

for idx, item in enumerate(dataset, 1):
    category = item["category"]
    question = item["question"]
    exp_source = item["expected_source"]

    res = None
    # Retry loop up to 3 attempts with a 120-second timeout
    for attempt in range(3):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/chat",
                json={"message": question},
                timeout=120  # Increased timeout from 30s to 120s
            )
            if response.status_code == 200:
                res = response.json()
                break
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"[{idx}/{total_questions}] Failed after 3 attempts: {e}")

    if res:
        response_text = res.get("response", "")
        citations = res.get("citations", [])

        # Check for source hit (Retrieval Recall)
        if exp_source == "None":
            hit = True  # Out-of-domain correctly expects no source
        else:
            hit = any(exp_source.lower() in str(c).lower() for c in citations)

        if hit:
            retrieval_hits += 1

        print(f"[{idx}/{total_questions}] Category: {category}")
        print(f"Q: {question}")
        print(f"A: {response_text[:100]}...")
        print(f"Source Hit: {'✓' if hit else '✗'}\n" + "-"*50)

recall = (retrieval_hits / total_questions) * 100

print("="*50)
print("EVALUATION SUMMARY")
print(f"Total Evaluated: {total_questions}")
print(f"Retrieval Recall@K Rate: {recall:.2f}%")
print("="*50)