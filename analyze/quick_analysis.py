#!/usr/bin/env python3
"""
Quick heuristic analysis of v2 vs v4 without using LLM
"""

import csv
from collections import defaultdict

v2_file = "/Users/funno/Documents/Competition/AgentCompetition/batch_progress_v2_filled.csv"
v4_file = "/Users/funno/Documents/Competition/AgentCompetition/batch_progress_v4_filled.csv"

def load_csv(filepath):
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['id']] = row['ret']
    return data

def analyze_answer(answer):
    """Extract features from answer"""
    if not answer:
        return None
    return {
        'length': len(answer),
        'sentences': answer.count('。') + answer.count('！') + answer.count('？'),
        'has_greeting': any(greeting in answer for greeting in ['您好', '为您', '我们']),
        'has_pics': '<PIC>' in answer,
        'num_pics': answer.count('<PIC>'),
        'text': answer
    }

def main():
    print("Loading data...")
    v2_data = load_csv(v2_file)
    v4_data = load_csv(v4_file)

    all_ids = sorted(set(v2_data.keys()) | set(v4_data.keys()), key=lambda x: int(x))

    # Statistics
    length_diff_stats = []
    v2_longer = 0
    v4_longer = 0
    same_length = 0

    print("\nAnalyzing answers...")
    for question_id in all_ids:
        v2_ans = v2_data.get(question_id, "")
        v4_ans = v4_data.get(question_id, "")

        if not v2_ans or not v4_ans:
            continue

        v2_features = analyze_answer(v2_ans)
        v4_features = analyze_answer(v4_ans)

        length_diff = v2_features['length'] - v4_features['length']

        if length_diff > 0:
            v2_longer += 1
        elif length_diff < 0:
            v4_longer += 1
        else:
            same_length += 1

        length_diff_stats.append({
            'id': question_id,
            'v2_length': v2_features['length'],
            'v4_length': v4_features['length'],
            'diff': length_diff,
            'v2_sents': v2_features['sentences'],
            'v4_sents': v4_features['sentences'],
            'v2_greeting': v2_features['has_greeting'],
            'v4_greeting': v4_features['has_greeting']
        })

    print("\n" + "="*70)
    print("QUICK ANALYSIS - STRUCTURAL DIFFERENCES")
    print("="*70)

    # Length analysis
    avg_v2_len = sum(s['v2_length'] for s in length_diff_stats) / len(length_diff_stats)
    avg_v4_len = sum(s['v4_length'] for s in length_diff_stats) / len(length_diff_stats)
    avg_v2_sents = sum(s['v2_sents'] for s in length_diff_stats) / len(length_diff_stats)
    avg_v4_sents = sum(s['v4_sents'] for s in length_diff_stats) / len(length_diff_stats)

    print(f"\nAnswer Length:")
    print(f"  V2 average: {avg_v2_len:.0f} chars, {avg_v2_sents:.1f} sentences")
    print(f"  V4 average: {avg_v4_len:.0f} chars, {avg_v4_sents:.1f} sentences")
    print(f"\nLength Comparison:")
    print(f"  V2 longer: {v2_longer} questions ({100*v2_longer/(v2_longer+v4_longer+same_length):.1f}%)")
    print(f"  V4 longer: {v4_longer} questions ({100*v4_longer/(v2_longer+v4_longer+same_length):.1f}%)")
    print(f"  Same length: {same_length} questions ({100*same_length/(v2_longer+v4_longer+same_length):.1f}%)")

    # Greeting/tone analysis
    v2_greetings = sum(1 for s in length_diff_stats if s['v2_greeting'])
    v4_greetings = sum(1 for s in length_diff_stats if s['v4_greeting'])
    print(f"\nGreeting/Customer-focused tone:")
    print(f"  V2: {v2_greetings} ({100*v2_greetings/len(length_diff_stats):.1f}%)")
    print(f"  V4: {v4_greetings} ({100*v4_greetings/len(length_diff_stats):.1f}%)")

    # Show examples
    print("\n" + "="*70)
    print("SAMPLE COMPARISONS (Length difference)")
    print("="*70)

    # Largest differences
    sorted_by_diff = sorted(length_diff_stats, key=lambda x: abs(x['diff']), reverse=True)[:3]
    for stat in sorted_by_diff:
        v2 = v2_data[stat['id']]
        v4 = v4_data[stat['id']]
        print(f"\nQ{stat['id']}: V2 is {stat['diff']} chars {'longer' if stat['diff'] > 0 else 'shorter'}")
        print(f"  V2 ({stat['v2_length']} chars): {v2[:100]}...")
        print(f"  V4 ({stat['v4_length']} chars): {v4[:100]}...")

if __name__ == "__main__":
    main()
