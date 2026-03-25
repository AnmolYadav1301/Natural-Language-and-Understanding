from collections import Counter

def print_top_words(tokens):
    freq = Counter(tokens)

    k = int(input("\nEnter how many top frequent words to display: "))
    
    print(f"\n===== TOP {k} FREQUENT WORDS =====")
    for word, count in freq.most_common(k):
        print(f"{word} : {count}")