import json
import numpy

with open("token_to_vector.json") as f:
    embeddings = json.load(f)

# Export embeddings
with open("embeddings.tsv", "w+") as f:
    for embedding in embeddings.values():
        line = "\t".join(map(str, embedding)) + "\n"
        f.write(line)

# Export meta
with open("meta.tsv", "w+") as f:
    for key in embeddings.keys():
        f.write(key + "\n")
