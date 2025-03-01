import os
from pypdf import PdfReader
from datasketch import MinHashLSH, MinHash

path = "/home/abhi/Documents" # path to the folder which contains pdf files
dir_list = os.listdir(path)
res = set()
lsh = MinHashLSH(threshold=0.5, num_perm=128)

#iterating over the files
for file in dir_list:
    file_path = os.path.join(path, file)
    reader = PdfReader(file_path)
    file_set = set() # stores all the unique words of a word
    for page in reader.pages: # iterating all the pages in documents
        text = page.extract_text() 
        for word in text.split():
            file_set.add(word.encode('utf-8')) # add encoded data into the set

    # iterating over the text to generate the minhashes
    hash = MinHash(num_perm = 128)
    for text in file_set:
        hash.update(text) 

    # matches the minhash values
    matches = lsh.query(hash)
    if len(matches) != 0:
        for file_names in matches:
            res.add(file_names)
        res.add(file)
    lsh.insert(file, hash)
print(res)
    
    


