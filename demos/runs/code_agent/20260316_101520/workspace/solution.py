import re

def extract_emails(text):
    """Extract email addresses from text."""
    emails = []
    for word in text.split():
        if "@" in word and "." in word:
            emails.append(word)
    return emails

def count_words(text):
    """Count words in text, excluding common stop words."""
    stop_words = ["the", "a", "an", "is", "are", "was", "were", "in", "on", "at"]
    words = text.lower().split()
    count = 0
    for w in words:
        if w not in stop_words:
            count = count + 1
    return count

def find_duplicates(items):
    """Find duplicate items in a list."""
    seen = []
    dupes = []
    for item in items:
        if item in seen:
            if item not in dupes:
                dupes.append(item)
        else:
            seen.append(item)
    return dupes
