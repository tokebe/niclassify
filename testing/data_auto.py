
import pandas as pd
import numpy as np
import csv
from contextlib import closing

import pandas as pd
import requests

geo = "Massachusetts"
taxon = "Insecta"
API = "http://www.boldsystems.org/index.php/API_Public/combined?"
tax_request = "{}taxon={}&geo={}&format=tsv".format(API, taxon, geo)
print(tax_request)

with open("result.tsv", "wb") as file, \
        requests.get(tax_request, stream=True) as response:
    for line in response.iter_lines():

        file.write(line)
        file.write(b"\n")

test = pd.read_csv("result.tsv", delimiter="\t",
                   engine="python", error_bad_lines=False)
test["nucleotides"] = test["nucleotides"].astype(str)

test.shape

test = test[test["marker_codes"].str.contains("COI-5P", na=False)]
test.shape

test = test[
    test.apply(
        (lambda x: True
         if len([i for i in x["nucleotides"] if i.isalpha()]) >= 350
         else False
         ),
        axis=1
    )
]
test.shape


test[["processid", "species_name", "nucleotides"]].to_csv(
    "check.tsv", sep="\t", index=False)

with open("out_fasta.fasta", "w") as file:
    for index, row in test.iterrows():
        file.write(">{}\n".format(row["processid"]))
        file.write(row["nucleotides"])
        file.write("\n")
