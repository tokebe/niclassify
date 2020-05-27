"""
Grab data and format it for use.

The purpose of this file is to test the concepts for automating data retrieval.
Later files will actually implement it in a decent manner.
"""
import csv
from contextlib import closing

import pandas as pd
import requests

# have to implement some form of validation for this.
# any way to get list from the API?
geo = "Massachusetts"
taxon = "Insecta"
API = "http://www.boldsystems.org/index.php/API_Public/combined?"

tax_request = "{}taxon={}&geo={}&format=tsv".format(API, taxon, geo)
print(tax_request)

with open("result.tsv", "wb") as file, \
        requests.get(tax_request, stream=True) as response:
    for line in response.iter_lines():
        file.write(line)

# this would appear to work just as we would like
# TODO add error checking and such
# TODO use the forloop to provide progress feedback in GUI
