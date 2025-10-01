import pandas as pd
from difflib import SequenceMatcher

def main():
    path = '../data_questions/Median_4.csv'
    df = pd.read_csv(path)
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Label distribution
    print("\nLabel distribution:")
    print(df['Label'].value_counts().sort_index())

    # Exact duplicates
    dup_count = df.duplicated(['Question', 'Label']).sum()
    print(f"\nExact duplicate (Question, Label) pairs: {dup_count}")
    if dup_count > 0:
        print(df[df.duplicated(['Question', 'Label'])].head())

    # Near-duplicate questions (Levenshtein ratio > 0.9, same label)
    print("\nChecking for near-duplicate questions (Levenshtein ratio > 0.9, same label)...")
    flagged = []
    texts = df['Question'].astype(str).str.lower().str.strip().tolist()
    labels = df['Label'].tolist()
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            if labels[i] == labels[j]:
                ratio = SequenceMatcher(None, texts[i], texts[j]).ratio()
                if 0.9 < ratio < 1.0:
                    flagged.append((texts[i], texts[j], labels[i], ratio))
    if flagged:
        print(f"Flagged {len(flagged)} near-duplicate question pairs:")
        for q1, q2, label, ratio in flagged[:10]:
            print(f"[{label}] {q1} <-> {q2} (similarity: {ratio:.2f})")
        if len(flagged) > 10:
            print(f"...and {len(flagged)-10} more.")
    else:
        print("No near-duplicate question pairs found above threshold.")

    # Sample questions for each label
    print("\nSample questions for each label:")
    for label in sorted(df['Label'].unique()):
        sample = df[df['Label'] == label]['Question'].head(3).tolist()
        print(f"Label {label}: {sample}")

if __name__ == "__main__":
    main()
