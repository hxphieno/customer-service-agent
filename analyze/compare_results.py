#!/usr/bin/env python3
"""
Compare batch_progress_v2_filled.csv and batch_progress_v4_filled.csv
Analyze which answer is better for each question and generate a combined result CSV
"""

import csv
import json
import os
from pathlib import Path
from anthropic import Anthropic

client = Anthropic()

# File paths
v2_file = "/Users/funno/Documents/Competition/AgentCompetition/batch_progress_v2_filled.csv"
v4_file = "/Users/funno/Documents/Competition/AgentCompetition/batch_progress_v4_filled.csv"
output_file = "/Users/funno/Documents/Competition/AgentCompetition/analyze/comparison_results.csv"
analysis_file = "/Users/funno/Documents/Competition/AgentCompetition/analyze/detailed_analysis.jsonl"

def load_csv(filepath):
    """Load CSV and return dict with id as key"""
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['id']] = row['ret']
    return data

def compare_answers(question_id, v2_answer, v4_answer):
    """Use Claude to compare two answers"""
    prompt = f"""You are evaluating customer service responses for a Chinese e-commerce platform.

Question ID: {question_id}

V2 Answer:
{v2_answer}

V4 Answer:
{v4_answer}

Please evaluate which answer is better and provide:
1. Winner: "V2", "V4", or "EQUAL"
2. Key differences
3. Why the winner is better (or why they're equal)

Be concise but specific. Focus on:
- Completeness of answer
- Clarity and structure
- Helpfulness to customer
- Accuracy of information
- Professional tone

Respond in JSON format:
{{
  "winner": "V2" or "V4" or "EQUAL",
  "confidence": 0.0-1.0,
  "v2_strengths": ["..."],
  "v2_weaknesses": ["..."],
  "v4_strengths": ["..."],
  "v4_weaknesses": ["..."],
  "explanation": "..."
}}"""

    message = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response_text = message.content[0].text
    try:
        # Extract JSON from response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        json_str = response_text[start:end]
        result = json.loads(json_str)
        return result
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing JSON for question {question_id}: {e}")
        return {
            "winner": "EQUAL",
            "confidence": 0.5,
            "explanation": f"Error parsing comparison: {str(e)}"
        }

def main():
    print("Loading CSV files...")
    v2_data = load_csv(v2_file)
    v4_data = load_csv(v4_file)

    # Get all unique IDs
    all_ids = sorted(set(v2_data.keys()) | set(v4_data.keys()), key=lambda x: int(x))

    print(f"Found {len(all_ids)} questions")
    print(f"V2 has {len(v2_data)} answers")
    print(f"V4 has {len(v4_data)} answers")

    # Compare each answer
    comparison_results = []
    v2_wins = 0
    v4_wins = 0
    equals = 0

    with open(analysis_file, 'w', encoding='utf-8') as analysis_f:
        for idx, question_id in enumerate(all_ids, 1):
            print(f"\n[{idx}/{len(all_ids)}] Comparing question {question_id}...")

            v2_answer = v2_data.get(question_id, "")
            v4_answer = v4_data.get(question_id, "")

            # Skip if either is missing
            if not v2_answer or not v4_answer:
                print(f"  Skipping: missing data (V2: {bool(v2_answer)}, V4: {bool(v4_answer)})")
                continue

            # Compare using Claude
            comparison = compare_answers(question_id, v2_answer, v4_answer)

            winner = comparison.get("winner", "EQUAL")
            confidence = comparison.get("confidence", 0.5)

            # Choose answer and track winner
            if winner == "V2":
                chosen_answer = v2_answer
                v2_wins += 1
            elif winner == "V4":
                chosen_answer = v4_answer
                v4_wins += 1
            else:
                chosen_answer = v2_answer  # Default to V2 for ties
                equals += 1

            # Store result
            result = {
                "id": question_id,
                "winner": winner,
                "confidence": confidence,
                "chosen_answer": chosen_answer,
                "v2_answer": v2_answer,
                "v4_answer": v4_answer,
                "analysis": comparison
            }
            comparison_results.append(result)

            # Write detailed analysis
            analysis_f.write(json.dumps({
                "id": question_id,
                "winner": winner,
                "confidence": confidence,
                "analysis": comparison
            }, ensure_ascii=False) + '\n')

            print(f"  Winner: {winner} (confidence: {confidence:.2f})")
            print(f"  Summary: {comparison.get('explanation', '')[:100]}...")

    # Write comparison results CSV
    print("\n\nGenerating comparison results CSV...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'winner', 'confidence', 'ret'])
        for result in comparison_results:
            writer.writerow([
                result['id'],
                result['winner'],
                f"{result['confidence']:.2f}",
                result['chosen_answer']
            ])

    # Print summary statistics
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Total questions compared: {len(comparison_results)}")
    print(f"V2 wins: {v2_wins} ({100*v2_wins/len(comparison_results):.1f}%)")
    print(f"V4 wins: {v4_wins} ({100*v4_wins/len(comparison_results):.1f}%)")
    print(f"Ties (V2 chosen): {equals} ({100*equals/len(comparison_results):.1f}%)")
    print(f"\nOutput files:")
    print(f"  - {output_file}")
    print(f"  - {analysis_file}")

if __name__ == "__main__":
    main()
