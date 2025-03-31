import os
import re
from pypdf import PdfReader
from datasketch import MinHash, MinHashLSH
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download stopwords if not already
nltk.download('stopwords')

# Config
path = "/home/abhi/Documents"
threshold = 0.85
num_perm = 256
ngram_size = 3

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def is_pdf_by_extension(file_path):
    return file_path.lower().endswith('.pdf')

def preprocess_text(text):
    """Return a list of cleaned, stemmed, lowercase words without stopwords."""
    words = text.lower().split()
    cleaned = []

    for w in words:
        w = re.sub(r'\W+', '', w)
        if w and w not in stop_words:
            cleaned_word = stemmer.stem(w)
            cleaned.append(cleaned_word)

    return cleaned

def get_ngrams(words, n=3):
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    return ngrams

# Step 1: Extract and clean words from PDFs
files = {}

for file in os.listdir(path):
    if not is_pdf_by_extension(file):
        continue

    file_path = os.path.join(path, file)
    reader = PdfReader(file_path)
    full_word_list = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            cleaned_words = preprocess_text(text)
            for word in cleaned_words:
                full_word_list.append(word)

    ngrams = get_ngrams(full_word_list, ngram_size)
    encoded_shingles = set()

    for ng in ngrams:
        encoded_shingles.add(ng.encode('utf-8'))

    files[file] = encoded_shingles

# Step 2: Create MinHash objects
file_hashes = {}

for file, shingles in files.items():
    m = MinHash(num_perm=num_perm)
    for sh in shingles:
        m.update(sh)
    file_hashes[file] = m

# Step 3: Build LSH index
lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

for file, mh in file_hashes.items():
    lsh.insert(file, mh)

# Step 4: Query for similar documents
res = set()

for file, mh in file_hashes.items():
    matches = lsh.query(mh)
    for match in matches:
        if match != file:
            sim = mh.jaccard(file_hashes[match])
            if sim >= threshold:
                res.add(file)
                res.add(match)

# Step 5: Print result
print(f"\nFound {len(res)} similar or near-duplicate files:\n")
for name in res:
    print(name)
 