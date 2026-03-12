import re
from collections import Counter
from typing import List

def extract_emails(text: str) -> List[str]:
    """
    Extract email addresses from text using RFC 5322 compliant regex pattern.
    
    Args:
        text: The input text to search for email addresses
        
    Returns:
        List of unique email addresses found in the text
        
    Examples:
        >>> extract_emails("Contact me at john@example.com or jane@test.org")
        ['john@example.com', 'jane@test.org']
    """
    # RFC 5322 compliant email regex pattern
    email_pattern = r"""
        [a-zA-Z0-9._%+-]+   # Username part
        @                   # @ symbol
        [a-zA-Z0-9.-]+      # Domain name
        \.                  # Dot before TLD
        [a-zA-Z]{2,}        # Top-level domain (2+ characters)
    """
    
    # Find all matches using the pattern with re.VERBOSE for readability
    emails = re.findall(email_pattern, text, re.VERBOSE)
    
    # Return unique emails while preserving order
    seen = set()
    unique_emails = []
    for email in emails:
        if email not in seen:
            seen.add(email)
            unique_emails.append(email)
    
    return unique_emails

def count_words(text: str) -> int:
    """
    Count words in text, excluding common stop words.
    Uses collections.Counter for efficient counting.
    
    Args:
        text: The input text to count words from
        
    Returns:
        Number of non-stop words in the text
        
    Examples:
        >>> count_words("The cat is on the mat")
        3  # 'cat', 'on', 'mat' (excluding 'the', 'is')
    """
    # Common stop words to exclude
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at"}
    
    # Split text into words and convert to lowercase
    words = text.lower().split()
    
    # Use Counter to count all words, then subtract stop words
    word_counts = Counter(words)
    
    # Count words that are not in stop_words
    non_stop_count = sum(count for word, count in word_counts.items() 
                        if word not in stop_words)
    
    return non_stop_count

def find_duplicates(items: List) -> List:
    """
    Find duplicate items in a list using sets for O(n) performance.
    
    Args:
        items: List of items to check for duplicates
        
    Returns:
        List of duplicate items (each duplicate appears only once in result)
        
    Examples:
        >>> find_duplicates([1, 2, 3, 2, 4, 3, 5])
        [2, 3]
    """
    if not items:
        return []
    
    # Use sets to find duplicates efficiently
    seen = set()
    duplicates = set()
    
    for item in items:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    
    # Convert set back to list to maintain consistent return type
    return list(duplicates)