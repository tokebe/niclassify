"""
Utilities for data, file, and program interactions relating to retrieving
sequence data and preparing features for classification.

Generally you want to import by importing the directory, utilities, and
accessing by utilities.function (instead of utilities.general_utils.function).
"""

import json
import os
import re
import requests
import shutil
import subprocess
import sys
import xlrd

import pandas as pd

from Bio.Align.Applications import MuscleCommandline
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio import AlignIO
from xml.etree import ElementTree

from .general_utils import MAIN_PATH, USER_PATH, REGIONS, R_LOC, RNotFoundError, RScriptFailedError, PLATFORM
from ..bPTP_interface import bPTP

REQUIRED_COLUMNS = [
    ["processid", "UPID", ""],  # can have one or the other
    "nucleotides",
    ["marker_codes", ""],  # empty means it's optional
    ["species_name", ""]
]

RESERVED_COLUMNS = [
    "species_group",
    "gbif_status",
    "itis_status",
    "final_status",
    "predict",
    "prob. endemic",
    "prob. introduced",
    "ksDist_mean",
    "ksDist_med",
    "ksDist_std",
    "ksDist_min",
    "ksDist_max",
    "ksSim_mean",
    "ksSim_med",
    "ksSim_std",
    "ksSim_min",
    "ksSim_max",
    "kaDist_mean",
    "kaDist_med",
    "kaDist_std",
    "kaDist_min",
    "kaDist_max",
    "kaSim_mean",
    "kaSim_med",
    "kaSim_std",
    "kaSim_min",
    "kaSim_max",
    "aaDist_mean",
    "aaDist_med",
    "aaDist_std",
    "aaDist_min",
    "aaDist_max",
    "aaSim_mean",
    "aaSim_med",
    "aaSim_std",
    "aaSim_min",
    "aaSim_max",
    "dnaDist_mean",
    "dnaDist_med",
    "dnaDist_std",
    "dnaDist_min",
    "dnaDist_max",
    "dnaSim_mean",
    "dnaSim_med",
    "dnaSim_std",
    "dnaSim_min",
    "dnaSim_max",
    "index",
    "level_0"
]


def align_fasta(infname, outfname, debug=False):
    """
    Generate an alignment for the given fasta file.

    Args:
        infname (str): Path to fasta to be aligned.
        outfname (str): Path to output fasta to be
    """
    muscle_exec = {
        "Windows": "niclassify/bin/muscle3.8.31_i86win32.exe",
        "Linux": "niclassify/bin/muscle3.8.31_i86linux64",
        "Darwin": "niclassify/bin/muscle3.8.31_i86darwin64"
    }[PLATFORM]

    alignment_call = MuscleCommandline(
        os.path.realpath(
            os.path.join(MAIN_PATH, muscle_exec)
        ),
        input=os.path.realpath(infname),
        out=os.path.realpath(outfname)
    )

    print(alignment_call.__str__())

    if debug:
        subprocess.run(
            alignment_call.__str__(),
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            shell=True
        )
    else:
        subprocess.run(alignment_call.__str__(), shell=True)

    r_script = os.path.realpath(
        os.path.join(
            MAIN_PATH, "niclassify/core/scripts/trim_alignment.R")
    )

    trim_call = [
        R_LOC,
        r_script,
        outfname,
        outfname
    ]

    if debug:
        proc = subprocess.run(
            trim_call,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            env=os.environ.copy()
        )
    else:
        proc = subprocess.run(trim_call, env=os.environ.copy())

    if os.stat(outfname).st_size == 0:
        raise ChildProcessError("Sequence Alignment Failed")

    if proc.returncode != 0:
        raise RScriptFailedError("R TrimAlignment failed")


def delimit_species_bPTP(infname, outtreefname, outfname, debug=False):
    """
    Delimit species by nucleotide sequence using bPTP method.

    Args:
        infname (str): Input file path.
        outtreefname (str): Output tree file path.
        outfname (str): Output file path.
        debug (bool, optional): Save script output to file.
    """
    r_script = os.path.realpath(
        os.path.join(
            MAIN_PATH, "niclassify/core/scripts/delim_tree.R")
    )
    python_path = sys.executable
    # bPTP = os.path.realpath(
    #     os.path.join(
    #         MAIN_PATH, "niclassify/bin/PTP-master/bin/bPTP.py")
    # )

    # assign log number, guarantee that both logs have same number
    fs = 0
    lpath = os.path.join(
        USER_PATH,
        "logs/delim"
    )
    while os.path.isfile(os.path.join(lpath, "delim/log{}.txt".format(fs))):
        fs += 1
    delimlogfile = open(
        os.path.join(lpath, "delim/log{}.txt".format(fs)), "w"
    )
    treelogfile = open(
        os.path.join(lpath, "tree/log{}.txt".format(fs)), "w"
    )

    # make tree
    if debug:
        proc = subprocess.run(
            [
                R_LOC,
                r_script,
                infname,
                outtreefname
            ],
            stdout=treelogfile,
            stderr=treelogfile,
            env=os.environ.copy()
        )
    else:
        proc = subprocess.run(
            [
                R_LOC,
                r_script,
                infname,
                outtreefname
            ],
            stdout=treelogfile,
            stderr=treelogfile,
            env=os.environ.copy(),
            creationflags=(
                0 if PLATFORM != 'Windows' else subprocess.CREATE_NO_WINDOW)
        )

    if os.stat(outtreefname).st_size == 0:
        raise ChildProcessError("bPTP Delimitation: Tree gen failed.")

    if proc.returncode != 0:
        raise RScriptFailedError("R TrimAlignment failed")

    # delimit species
    bPTP.main_routine(
        trees=outtreefname,
        output=outfname,
        seed=123,
        reroot=False,
        delete=False,
        method="H1",
        nmcmc=10000,
        imcmc=100,
        burnin=0.1,
        num_trees=0,
        nmi=False,
        scale=500
    )
    treelogfile.close()
    delimlogfile.close()

    # read delimitation file and convert to .tsv
    with open(
        outfname + ".PTPMLPartition.txt",
        "r"
    ) as dfile:
        # read lines of file
        delim = dfile.readlines()
        # first line is useless for data capture
        del delim[0]

        # grab species and samples
        species = []
        for d in delim[::3]:
            species.extend(re.findall("(?<=Species) [0-9]*", d))
        # species = [
        #     re.search("(?<=Species) [0-9]*", d).group()
        #     for d in delim[::3]
        # ]
        samples = [d.strip().split(",") for d in delim[1::3]]

        # make two equal-length lists of data in long format
        species_expanded = []
        samples_expanded = []

        for sp, sa in {sp: sa for sp, sa in zip(species, samples)}.items():
            for sample in sa:
                species_expanded.append(sp)
                samples_expanded.append(sample)

        # convert to dataframe and save to file
        pd.DataFrame({
            "Delim_spec": species_expanded,
            "sample_name": samples_expanded
        }).to_csv(outfname, index=False)


def delimit_species_GMYC(infname, outtreefname, outfname, debug=False):
    """
    Delimit species by nucleotide sequence using GMYC method.

    Args:
        infname (str): Input file path.
        outtreefname (str): Output tree file path.
        outfname (str): Output file path.
        debug (bool, optional): Save script output to file.
    """
    r_script = os.path.realpath(
        os.path.join(
            MAIN_PATH, "niclassify/core/scripts/delim_tree.R")
    )

    fs = 0
    lpath = os.path.join(
        USER_PATH,
        "logs/delim/delim"
    )
    while os.path.isfile(os.path.join(lpath, "log{}.txt".format(fs))):
        fs += 1

    if debug:
        with open(
                os.path.join(lpath, "log{}.txt".format(fs)), "w"
        ) as logfile:
            proc = subprocess.run(
                [
                    R_LOC,
                    r_script,
                    infname,
                    outtreefname,
                    outfname
                ],
                stdout=logfile,
                stderr=logfile,
                env=os.environ.copy()
            )
    else:
        proc = subprocess.run(
            [
                R_LOC,
                r_script,
                infname,
                outtreefname,
                outfname
            ],
            env=os.environ.copy()
        )

    if proc.returncode != 0:
        raise RScriptFailedError("R TrimAlignment failed")


def generate_measures(fastafname, delimfname, outfname, debug=False):
    r_script = os.path.realpath(
        os.path.join(
            MAIN_PATH,
            "niclassify/core/scripts/create_measures.R"
        )
    )
    ftgen_call = [
        R_LOC,
        r_script,
        fastafname,
        delimfname,
        outfname
    ]
    # assign log number
    fs = 0
    lpath = os.path.join(
        USER_PATH,
        "logs/ftgen"
    )
    while os.path.isfile(os.path.join(lpath, "log{}.txt".format(fs))):
        fs += 1
    logfile = open(
        os.path.join(lpath, "log{}.txt".format(fs)), "w"
    )

    # run script
    if debug:
        proc = subprocess.run(
            ftgen_call,
            stdout=logfile,
            stderr=logfile,
            env=os.environ.copy()
        )
    else:
        proc = subprocess.run(
            ftgen_call,
            stdout=logfile,
            stderr=logfile,
            env=os.environ.copy(),
            creationflags=(
                0 if PLATFORM != 'Windows' else subprocess.CREATE_NO_WINDOW)
        )

    if proc.returncode != 0:
        raise RScriptFailedError("R TrimAlignment failed")


def geo_contains(ref_geo, geo):
    """
    Check if a given reference geography contains another geography.

    Args:
        ref_geo (str): The reference geography.
        geo (str): The geography expected to be contained within the reference.

    Raises:
        TypeError: If the reference geography does not exist in the configs.

    Returns:
        bool: True if geo in ref_geo else False.

    """
    # get the actual hierarchy
    hierarchy = get_ref_hierarchy(ref_geo)
    if hierarchy is None:
        # raise TypeError(
        print(
            "reference geography <{}> does not exist!".format(ref_geo))
        return False
    if hierarchy["Contains"] is None:
        return False

    def match_geo(level, ref):
        if level is None:
            return False
        result = False

        for name, sub in level.items():
            if name == ref:
                result = True
                break
            elif not result and sub["Contains"] is not None:
                result = match_geo(sub["Contains"], ref)

        return result

    return match_geo(hierarchy["Contains"], geo)


def get_geo_taxon(filename, geo=None, taxon=None, api=None):
    """
    Save a request result from the api.

    Args:
        filename (str): Path to file to be created.
        geo (str): Geography descriptor
        taxon (str): Taxonomy descriptor
        api (str, optional): Base API URL. Defaults to None.

    Raises:
        OSError: If file creation fails.
        request.RequestException: If request otherwise fails.

    """
    if api is None:
        api = "http://www.boldsystems.org/index.php/API_Public/combined?"

    if not os.path.isabs(filename):
        filename = os.path.join(USER_PATH, "data/unprepared/" + filename)

    # create request from options
    request = []

    if taxon is not None:
        request.append("taxon={}".format(taxon))
    if geo is not None:
        request.append("geo={}".format(geo))
    request.append("format=tsv")

    request = api + "&".join(request)

    try:
        attempts = 3
        while True:
            if attempts == 0:
                raise request.exceptions.RequestException(
                    "site keeps timing out")
                break
            print("making request...")
            try:
                with open(filename, "wb") as file, \
                        requests.get(request, stream=True) as response:

                    # error if response isn't success
                    response.raise_for_status()
                    shutil.copyfileobj(response.raw, file)

                return
            except requests.exceptions.Timeout:
                attempts -= 1
                print(
                    "    request timed out, trying again ({} of 3)...".format(
                        3 - attempts))
                pass
            except requests.exceptions.RequestException as e:
                raise e
                break

    except UnicodeDecodeError as e:
        raise e

    # except (OSError, IOError, KeyError, TypeError, ValueError):
    #     raise OSError("File could not be created.")

    # except requests.RequestException:
    #     raise request.RequestException("Request Failed.")


def get_geographies():
    """
    Return a list of all geographies in regions config file.

    Returns:
        list: All geography names as str.

    """
    def getlist(section):
        items = []
        for name, sub in section.items():
            items.append(name)
            if sub["Contains"] is not None:
                items.extend(getlist(sub["Contains"]))
        return items

    return getlist(REGIONS)


def get_jurisdictions(species_name):
    """
    Get ITIS jurisdictions for a given species.

    Args:
        species_name (str): A binomial species name.

    Returns:
        dict: A dictionary of jurisdictions and status.

    """
    tsn_link = (
        "http://www.itis.gov/ITISWebService/services/ITISService/\
getITISTermsFromScientificName?srchKey="
    )
    jurisdiction_link = (
        "http://www.itis.gov/ITISWebService/services/ITISService/\
getJurisdictionalOriginFromTSN?tsn="
    )

    print("making request...")
    # get TSN
    req = "{}{}".format(
        tsn_link, species_name.replace(" ", "%20"))

    # query website and check for error
    response = requests.get(req)  # stream this if it's a large response
    print("got request, parsing...")
    response.raise_for_status()
    # get xml tree from response
    tree = ElementTree.fromstring(response.content)
    # get any TSN's
    vals = [
        i.text
        for i
        in tree.iter('{http://data.itis_service.itis.usgs.gov/xsd}tsn')
    ]

    if vals is None:  # skip if there's no tsn to be found
        return None

    elif len(vals) != 1:  # skip if tsn is empty or there are more than one
        return None

    tsn = vals[0]  # tsn captured

    # get jurisdiction
    req = "{}{}".format(jurisdiction_link, tsn)

    response = requests.get(req)

    response.raise_for_status()

    try:
        tree = ElementTree.fromstring(response.content)
    except UnicodeDecodeError:
        return None

    juris = {
        j.text: n.text
        for j, n
        in zip(
            tree.iter(
                '{http://data.itis_service.itis.usgs.gov/xsd}jurisdictionValue'
            ),
            tree.iter('{http://data.itis_service.itis.usgs.gov/xsd}origin')
        )
    }

    if len(juris) == 0:  # or if it's somehow returned empty
        return None

    return juris


def get_native_ranges(species_name):
    """
    Get native ranges from GBIF for a given species.

    Args:
        species_name (str): A binomial species name.

    Returns:
        list: A list of native ranges as str.

    """
    code_link = "http://api.gbif.org/v1/species?name="
    records_link = "http://api.gbif.org/v1/species/"

    # get taxonKey
    req = "{}{}".format(
        code_link, species_name.replace(" ", "%20"))

    response = requests.get(req)  # stream this if it's a large response

    response.raise_for_status()

    # print("got 1st response")

    # search for "taxonID":"gbif:" with some numbers, getting the numbers
    taxonKey = re.search('(?<="taxonID":"gbif:)\\d+', response.text)

    if taxonKey is None:
        # print("no key found")
        return None
    else:
        taxonKey = taxonKey.group()

    # get native range
    req = "{}{}/descriptions".format(
        records_link, taxonKey)

    response = requests.get(req)  # stream this if it's a large response

    response.raise_for_status()

    try:
        results = response.json()
    except UnicodeDecodeError:
        return None

    lookup = [
        res["description"]
        for res in results["results"]
        if ("type" in res and res["type"] == "native range" and "description" in res)
    ]

    if len(lookup) == 0:
        return None
    else:
        return lookup


def get_ref_hierarchy(ref_geo):
    """
    Get a hierarchy contained in a given reference geography.

    Args:
        ref_geo (str): A geography name.

    Returns:
        dict: The hierarchy contained in the reference geography.

    """
    def find_geo(level, ref):
        if level is None:
            return None
        result = None

        for name, sub in level.items():
            if name == ref:
                result = sub
                break
            elif result is None:
                result = find_geo(sub["Contains"], ref)

        return result

    return find_geo(REGIONS, ref_geo)


def make_genetic_measures(infname, outfname=None):
    """
    Make the genetic measures to be used for classification.

    Args:
        infname (str): Input file path.
        outfname (str, optional): Output file path. Defaults to None.
    """
    print("reading data...")
    aln = AlignIO.read(open(infname), 'fasta')
    print("calculating distances...")
    calculator = DistanceCalculator('identity')
    dm = calculator.get_distance(aln)

    print(dm)


def filter_sequence_data(data, bp=350):
    """
    Prepare sequence data previously saved from API.

    Args:
        data (DataFrame): DataFrame of sequence data.

    Returns:
        DataFrame: The data after cleaning.

    Raises:
        pandas.errors.ParserError: If data could not be parsed. Likely caused
            by request returning extraneous error code.

    """
    if data.shape[0] == 0:
        raise ValueError("Datafile contains no observations.")

    # change to str in case it's not
    data["nucleotides"] = data["nucleotides"].astype(str)

    # remove rows missing COI-5P in marker_codes (if column provided)
    if "marker_codes" in data.columns:
        # avoid exceptions by casting to str
        data["marker_codes"] = data["marker_codes"].astype(str)
        data = data[data["marker_codes"].str.contains("COI-5P", na=False)]

    # remove rows with less than bp (350 default) base pairs
    data = data[
        data.apply(
            (lambda x: True
             if len([i for i in x["nucleotides"] if i.isalpha()]) >= bp
             else False),
            axis=1
        )
    ]

    # drop legitimate duplicate rows
    data.drop_duplicates(inplace=True)

    # in case user already had UPID
    if "UPID" in data.columns:
        # make new UPID if UPID is not unique
        if not data["UPID"].is_unique:
            bad_UPID = data["UPID"]
            data.drop("UPID", axis=1, inplace=True)
            data.insert(
                0, "UPID",
                bad_UPID.astype(str)
                + "_"
                + bad_UPID.groupby(bad_UPID).cumcount().add(1).astype(str)
            )

    # make new ID column (fix any duplicate processid's)
    elif "processid" in data.columns:
        if data["processid"].is_unique:
            data.insert(0, "UPID", data["processid"])
        else:
            data.insert(0, "UPID", (
                data["processid"].astype(str)
                + "_"
                + data.groupby("processid").cumcount().add(1).astype(str)
            ))


    # neither provided, make new
    else:
        data.insert(0, "UPID", (
            "SN" + data.reset_index()["index"].astype(str)
        ))  # SN stands for Sample Number

    # drop columns which may interfere with operation
    # TODO add a warning about reserved columns in manual
    data.drop(
        columns=RESERVED_COLUMNS,
        inplace=True,
        errors='ignore'
    )

    return data.reset_index(drop=True)


def write_fasta(data, filename):
    """
    Write a fasta file from a dataframe of sequence data.

    Args:
        data (DataFrame): sequence data, preferably filtered.
        filename (str): path to save fasta file to.
    """
    with open(filename, "w") as file:
        for index, row in data.iterrows():
            file.write(">{}\n".format(row["UPID"]))
            file.write(row["nucleotides"])
            file.write("\n")
