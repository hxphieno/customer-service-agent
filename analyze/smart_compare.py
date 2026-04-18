#!/usr/bin/env python3
"""
Smart comparison using heuristics based on quality metrics
Combine multiple scoring factors to determine which answer is better
"""

import csv
import re
from pathlib import Path

v2_file = "/Users/funno/Documents/Competition/AgentCompetition/batch_progress_v2_filled.csv"
v4_file = "/Users/funno/Documents/Competition/AgentCompetition/batch_progress_v4_filled.csv"
output_file = "/Users/funno/Documents/Competition/AgentCompetition/analyze/combined_best.csv"
analysis_file = "/Users/funno/Documents/Competition/AgentCompetition/analyze/comparison_analysis.txt"

def load_csv(filepath):
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['id']] = row['ret']
    return data

def score_answer(answer):
    """
    Score answer quality based on multiple heuristics
    Returns a score 0-100 where higher is better
    """
    if not answer:
        return 0

    score = 0

    # 1. Length factor (moderate length is better, too short or too long is bad)
    length = len(answer)
    if 100 <= length <= 600:
        score += 20
    elif 60 <= length <= 800:
        score += 15
    elif length >= 50:
        score += 10

    # 2. Structure: proper sentences with punctuation
    sentence_count = answer.count('。') + answer.count('！') + answer.count('？')
    if sentence_count >= 2:
        score += 15

    # 3. Customer service tone (Chinese markers)
    service_markers = ['您好', '为您', '我们', '建议您', '感谢您', '帮您', '可以为您', '为您解答']
    tone_count = sum(1 for marker in service_markers if marker in answer)
    score += min(tone_count * 3, 15)

    # 4. Content completeness markers
    completeness_markers = [
        ('关于', 1),
        ('具体', 1),
        ('详细', 1),
        ('条件', 1),
        ('要求', 1),
        ('范围', 1),
        ('费用', 1),
        ('时间', 1),
        ('建议', 2),
        ('注意', 2),
        ('说明', 1),
    ]
    completeness_score = sum(count for marker, count in completeness_markers if marker in answer)
    score += min(completeness_score, 20)

    # 5. Image references (should have some if applicable)
    pic_count = answer.count('<PIC>')
    if pic_count > 0:
        score += 10

    # 6. Language quality: avoid repetition
    # Simple repetition check
    words = answer.split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio > 0.7:
            score += 10

    # 7. Penalty for overly verbose or unclear structure
    # Too many commas might indicate run-on sentences
    comma_count = answer.count('、') + answer.count(',')
    if comma_count > sentence_count * 3:
        score -= 5

    # 8. Clarity: proper use of structure markers
    structure_markers = ['首先', '其次', '最后', '一、', '二、', '三、', '（1', '（2', '（3']
    structure_count = sum(1 for marker in structure_markers if marker in answer)
    if structure_count > 0:
        score += 5

    # 9. Mixed language check (if both English and Chinese, could indicate paste errors)
    has_english = bool(re.search(r'[a-zA-Z]{3,}', answer))
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', answer))
    if has_english and has_chinese:
        # Mixed language - check if it's intentional (like part numbers)
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', answer))
        if english_words > len(answer.split()) / 3:
            score -= 10  # Too much English in what should be Chinese answer

    return min(score, 100)

def compare_and_choose(q_id, v2_ans, v4_ans):
    """Compare two answers and return the better one"""
    v2_score = score_answer(v2_ans)
    v4_score = score_answer(v4_ans)

    if v2_score > v4_score:
        return "V2", v2_ans, v2_score, v4_score
    elif v4_score > v2_score:
        return "V4", v4_ans, v4_score, v2_score
    else:
        # Tie: prefer shorter answer (more concise)
        if len(v2_ans) <= len(v4_ans):
            return "EQUAL-V2", v2_ans, v2_score, v4_score
        else:
            return "EQUAL-V4", v4_ans, v4_score, v2_score

def main():
    print("Loading data...")
    v2_data = load_csv(v2_file)
    v4_data = load_csv(v4_file)

    all_ids = sorted(set(v2_data.keys()) | set(v4_data.keys()), key=lambda x: int(x))

    results = []
    v2_wins = 0
    v4_wins = 0
    equal = 0

    print("Comparing answers...")
    analysis_lines = ["="*80, "DETAILED COMPARISON ANALYSIS", "="*80]

    for question_id in all_ids:
        v2_ans = v2_data.get(question_id, "")
        v4_ans = v4_data.get(question_id, "")

        if not v2_ans or not v4_ans:
            continue

        winner, chosen_ans, winner_score, loser_score = compare_and_choose(question_id, v2_ans, v4_ans)

        if winner.startswith("V2"):
            v2_wins += 1
        elif winner.startswith("V4"):
            v4_wins += 1
        else:
            equal += 1

        v2_score = score_answer(v2_ans)
        v4_score = score_answer(v4_ans)

        results.append({
            'id': question_id,
            'winner': winner,
            'v2_score': v2_score,
            'v4_score': v4_score,
            'chosen': chosen_ans
        })

        # Detailed analysis for larger differences
        score_diff = abs(v2_score - v4_score)
        if score_diff > 20:
            analysis_lines.append(f"\nQ{question_id}: {winner} (scores: {v2_score:.1f} vs {v4_score:.1f}, diff={score_diff:.1f})")
            analysis_lines.append(f"  V2 ({len(v2_ans)} chars): {v2_ans[:80]}...")
            analysis_lines.append(f"  V4 ({len(v4_ans)} chars): {v4_ans[:80]}...")

    # Write combined results
    print(f"Writing results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'ret'])
        writer.writeheader()
        for result in results:
            writer.writerow({'id': result['id'], 'ret': result['chosen']})

    # Write analysis
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(analysis_lines))

    # Print summary
    total = v2_wins + v4_wins + equal
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"Total compared: {total}")
    print(f"V2 better: {v2_wins} ({100*v2_wins/total:.1f}%)")
    print(f"V4 better: {v4_wins} ({100*v4_wins/total:.1f}%)")
    print(f"Equal: {equal} ({100*equal/total:.1f}%)")
    print(f"\nFiles generated:")
    print(f"  - {output_file}")
    print(f"  - {analysis_file}")

    # Why V2 is better
    print("\n" + "="*80)
    print("WHY V2 SCORES HIGHER (0.36 vs 0.282)")
    print("="*80)
    print("\nKey factors:")
    print(f"1. Quality over length: V2 wins in {v2_wins} cases ({100*v2_wins/total:.1f}%)")
    print(f"   - V2 answers are more concise and focused")
    print(f"   - Average V2 length: ~357 chars (better precision)")
    print(f"   - Average V4 length: ~460 chars (may dilute focus)")
    print()
    print(f"2. V4 emphasizes quantity: V4 longer in 76.4% of cases")
    print(f"   - Longer doesn't always mean better in customer service")
    print(f"   - Evaluation likely rewards clarity + completeness, not verbosity")
    print()
    print(f"3. V2 maintains better tone consistency:")
    print(f"   - More professional product-focused answers")
    print(f"   - Fewer customer service clichés (2.5% greeting rate)")
    print(f"   - Lets product info speak for itself")
    print()
    print(f"4. V4's approach may backfire:")
    print(f"   - High greeting rate (23%) feels formulaic")
    print(f"   - Longer responses may repeat information")
    print(f"   - Mixed language content in some responses")

if __name__ == "__main__":
    main()
