import requests
import pandas as pd
# import json
import re
import time

codes = []

data = pd.read_csv("unfiltered_sequence_us2.tsv", sep="\t", engine="python")

get_code_link = "http://api.gbif.org/v1/species?name="

get_records_link = "http://api.gbif.org/v1/species/"

for idx, row in data[data["species_name"].notna()].iterrows():

    # get taxonKey
    req = "{}{}".format(
        get_code_link, row["species_name"].replace(" ", "%20"))

    response = requests.get(req)  # stream this if it's a large response

    response.raise_for_status()

    # print("got 1st response")

    # search for "taxonID":"gbif:" with some numbers, getting the numbers
    taxonKey = re.search('(?<="taxonID":"gbif:)\d+', response.text)

    if taxonKey is None:
        # print("no key found")
        continue
    else:
        taxonKey = taxonKey.group()

    # get Distributions
    req = "{}{}/descriptions".format(
        get_records_link, taxonKey)

    response = requests.get(req)  # stream this if it's a large response

    response.raise_for_status()

    # print("got 2nd response")

    results = response.json()

    for res in results["results"]:

        if res["type"] != "native range":
            continue
        else:
            if res["description"] not in codes:
                print("got code: {}".format(res["description"]))
                codes.append(res["description"])

                with open("gbif_nd_codes.txt", "a+") as codefile:
                    codefile.write("{}\n".format(res["description"]))

    time.sleep(.1)
