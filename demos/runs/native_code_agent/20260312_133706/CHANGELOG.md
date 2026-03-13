# Changelog

## Improvements to solution.py

### 1. Email Extraction Function (`extract_emails`)

**Changes Made:**
- Replaced simplistic string checking with RFC 5322 compliant regex pattern
- Added proper type hints (`str` -> `List[str]`)
- Enhanced docstring with detailed explanation and examples
- Added deduplication logic to return unique email addresses
- Used `re.VERBOSE` flag for readable regex pattern

**Why:**
- The original implementation was too basic and would incorrectly match non-email strings containing "@" and "."
- RFC 5322 compliant regex ensures proper email validation according to internet standards
- Deduplication prevents returning the same email multiple times
- Type hints improve code clarity and enable better IDE support

### 2. Word Counting Function (`count_words`)

**Changes Made:**
- Replaced manual loop counter with `collections.Counter`
- Changed stop_words from list to set for O(1) lookup time
- Added proper type hints (`str` -> `int`)
- Improved docstring with examples and detailed explanation
- More efficient counting logic using Counter's capabilities

**Why:**
- `collections.Counter` is the Pythonic way to count items and is more efficient
- Sets provide O(1) membership testing vs O(n) for lists
- Counter automatically handles counting and provides useful methods
- Performance improvement for large texts due to better data structures

### 3. Duplicate Finding Function (`find_duplicates`)

**Changes Made:**
- Replaced O(n²) list-based algorithm with O(n) set-based approach
- Changed from nested lists to sets for tracking seen items and duplicates
- Added proper type hints (`List` -> `List`)
- Enhanced docstring with clear examples
- Added edge case handling for empty input

**Why:**
- Original algorithm had O(n²) time complexity due to nested list searches
- Sets provide O(1) membership testing, making the overall algorithm O(n)
- Much better performance for large lists
- Cleaner, more Pythonic code structure

### 4. General Improvements

**Changes Made:**
- Added comprehensive type hints throughout the module
- Improved all docstrings with Args, Returns, and Examples sections
- Added imports for `Counter` and `List` types
- Better variable naming and code organization
- Added edge case handling

**Why:**
- Type hints improve code maintainability and catch potential bugs
- Better documentation makes the code more usable for other developers
- Following Python best practices for imports and code structure
- More robust code that handles edge cases properly

### Performance Improvements

- **Email extraction**: More accurate but similar performance (regex compilation overhead)
- **Word counting**: Significant improvement for large texts due to Counter and set usage
- **Duplicate finding**: Major performance improvement from O(n²) to O(n) time complexity

### Code Quality Improvements

- Better adherence to PEP 8 style guidelines
- More descriptive variable names
- Proper error handling considerations
- Clear separation of concerns
- More maintainable and readable code structure