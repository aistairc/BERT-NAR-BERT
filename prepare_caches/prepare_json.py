import os
import json
import datasets


json_dir = "/scratch/aae15163zd/data/jsons"

wiki = datasets.load_dataset("wikipedia", "20220301.en", split="train")
wiki.remove_columns(["url", "title"])
wiki.to_csv(os.path.join(json_dir, "wikipedia.csv"), num_proc=20)

#books = datasets.load_dataset("bookcorpus", split="train")
#books.to_csv(os.path.join(json_dir, "bookcorpus.csv"))
