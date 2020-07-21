import requests
import pandas as pd
from xml.etree import ElementTree

data = pd.read_csv("res3.tsv", sep="\t", engine="python")

get_tsn_link = "http://www.itis.gov/ITISWebService/services/ITISService/getITISTermsFromScientificName?srchKey="

get_divisions_link = "http://www.itis.gov/ITISWebService/services/ITISService/getGeographicDivisionsFromTSN?tsn="

get_jurisdiction_link = "http://www.itis.gov/ITISWebService/services/ITISService/getJurisdictionalOriginFromTSN?tsn="

divisions = []
jurisdictions = []


for idx, row in data[data["species_name"].notna()].iterrows():

    # get TSN
    req = "{}{}".format(
        get_tsn_link, row["species_name"].replace(" ", "%20"))

    response = requests.get(req)  # stream this if it's a large response

    response.raise_for_status()

    tree = ElementTree.fromstring(response.content)

    vals = [
        i.text
        for i
        in tree.iter('{http://data.itis_service.itis.usgs.gov/xsd}tsn')
    ]

    # replace break with continue and/or add 'unknown'
    if vals is None:  # if there's no tsn to be found
        continue

    elif len(vals) != 1:  # if we get nothing, or too many (ambiguous)
        continue

    tsn = vals[0]

    # get Geographic Divisions
    req = "{}{}".format(get_divisions_link, tsn)

    response = requests.get(req)

    response.raise_for_status()

    tree = ElementTree.fromstring(response.content)

    divs = [
        i.text
        for i
        in tree.iter('{http://data.itis_service.itis.usgs.gov/xsd}geographicValue')
    ]

    if divs is None:  # if there's no tsn to be found
        continue

    elif len(divs) == 0:  # if we get nothing, or too many (ambiguous)
        continue

    # get Geographic Divisions
    req = "{}{}".format(get_jurisdiction_link, tsn)

    response = requests.get(req)

    response.raise_for_status()

    tree = ElementTree.fromstring(response.content)

    juris = [
        i.text
        for i
        in tree.iter('{http://data.itis_service.itis.usgs.gov/xsd}jurisdictionValue')
    ]

    if juris is None:  # if there's no tsn to be found
        continue

    elif len(juris) == 0:  # if we get nothing, or too many (ambiguous)
        continue

    print(" got divs: {}".format(" ".join(divs)))
    print(" got juris: {}".format(" ".join(juris)))

    with open("divs.txt", "a+") as div_txt:
        for d in divs:
            if d not in divisions:
                divisions.append(d)
                div_txt.write("{}\n".format(d))

    with open("juris.txt", "a+") as juris_txt:
        for j in juris:
            if j not in jurisdictions:
                jurisdictions.append(j)
                juris_txt.write("{}\n".format(j))
