#! /usr/bin/env python
import enum
import sys
import math
import random
import argparse
import os
from pathlib import Path

from ete3 import Tree
from nexus import NexusReader
from bptp.summary import partitionparser
from bptp.ptpllh import (
    lh_ratio_test,
    exp_distribution,
    species_setting,
    exponential_mixture,
)

import matplotlib.pyplot as plt


class ptpmcmc:
    """MCMC on a single tree using PTP model"""

    def __init__(
        self,
        tree,
        start_config=None,
        reroot=False,
        startmethod="H0",
        min_br=0.0001,
        seed=1234,
        thinning=100,
        sampling=10000,
        burning=0.1,
        taxa_order=[],
    ):
        if start_config == None:
            me = exponential_mixture(tree=tree)
            me.search(strategy=startmethod, reroot=reroot)
            me.count_species(print_log=False, pv=0.0)
            self.tree = me.tree
            self.current_setting = me.max_setting
        else:
            self.current_setting = start_config
            self.tree = Tree(tree, format=1)
        self.burning = burning
        self.last_setting = self.current_setting
        self.current_logl = self.current_setting.get_log_l()
        self.last_logl = self.last_setting.get_log_l()
        self.min_br = min_br
        self.rand_nr = random.Random()
        self.rand_nr.seed(seed)
        self.thinning = thinning
        self.sampling = sampling
        if taxa_order == []:
            self.taxaorder = self.tree.get_leaf_names()
        else:
            self.taxaorder = taxa_order
        self.numtaxa = len(self.taxaorder)
        self.partitions = []
        self.llhs = []
        self.nsplit = 0
        self.nmerge = 0
        """remember the ML partition"""
        self.maxllh = self.current_logl
        to, spe = self.current_setting.output_species(taxa_order=self.taxaorder)
        self.maxpar = spe
        self.max_setting = self.current_setting
        """record all delimitation settings for plotting, this could consume a lot of MEM"""
        self.settings = []

    def split(self, chosen_anode):
        self.nsplit = self.nsplit + 1
        newspenodes = []
        for node in self.current_setting.spe_nodes:
            newspenodes.append(node)
        newspenodes.extend(chosen_anode.get_children())
        self.current_setting = species_setting(
            spe_nodes=newspenodes,
            root=self.tree,
            sp_rate=0,
            fix_sp_rate=False,
            minbr=self.min_br,
        )
        self.current_logl = self.current_setting.get_log_l()

    def merge(self, chosen_anode):
        self.nmerge = self.nmerge + 1
        mnodes = chosen_anode.get_children()
        newspenodes = []
        for node in self.current_setting.spe_nodes:
            if not node in mnodes:
                newspenodes.append(node)
        self.current_setting = species_setting(
            spe_nodes=newspenodes,
            root=self.tree,
            sp_rate=0,
            fix_sp_rate=False,
            minbr=self.min_br,
        )
        self.current_logl = self.current_setting.get_log_l()

    def mcmc(self):
        cnt = 0
        accepted = 0
        sample_start = int(self.sampling * self.burning)
        printinterval = self.thinning * 100
        while cnt < self.sampling:
            cnt = cnt + 1
            if cnt % printinterval == 0:
                print("MCMC generation: " + repr(cnt))
            self.last_setting = self.current_setting
            self.last_logl = self.current_logl
            acceptance = 0.0
            """proposal"""
            """First chose to split or merge"""
            rdchoice = self.rand_nr.uniform(0.0, 1.0)
            if rdchoice <= 0.5:
                """split"""
                xinverse = self.current_setting.get_nodes_can_split()
                if xinverse > 0:
                    rdidx = self.rand_nr.randint(0, xinverse - 1)
                    chosen_anode = self.current_setting.node_can_split[rdidx]
                    self.split(chosen_anode)
                    xpinverse = self.current_setting.get_nodes_can_merge()
                    if xpinverse > 0:
                        newlogl = self.current_logl
                        oldlogl = self.last_logl
                        acceptance = (
                            math.exp(newlogl - oldlogl)
                            * float(xinverse)
                            / float(xpinverse)
                        )
                        if newlogl > self.maxllh:
                            self.maxllh = newlogl
                            to, spe = self.current_setting.output_species(
                                taxa_order=self.taxaorder
                            )
                            self.maxpar = spe
                            self.max_setting = self.current_setting
            else:
                """merge"""
                xinverse = self.current_setting.get_nodes_can_merge()
                if xinverse > 0:
                    rdidx = self.rand_nr.randint(0, xinverse - 1)
                    chosen_anode = self.current_setting.node_can_merge[rdidx]
                    self.merge(chosen_anode)
                    xpinverse = self.current_setting.get_nodes_can_split()
                    if xpinverse > 0:
                        newlogl = self.current_logl
                        oldlogl = self.last_logl
                        acceptance = (
                            math.exp(newlogl - oldlogl)
                            * float(xinverse)
                            / float(xpinverse)
                        )
                        if newlogl > self.maxllh:
                            self.maxllh = newlogl
                            to, spe = self.current_setting.output_species(
                                taxa_order=self.taxaorder
                            )
                            self.maxpar = spe
                            self.max_setting = self.current_setting

            if acceptance > 1.0:
                if cnt % self.thinning == 0 and cnt >= sample_start:
                    to, spe = self.current_setting.output_species(
                        taxa_order=self.taxaorder
                    )
                    self.partitions.append(spe)
                    self.llhs.append(newlogl)
                    self.settings.append(self.current_setting)
                accepted = accepted + 1
            else:
                u = self.rand_nr.uniform(0.0, 1.0)
                if u < acceptance:
                    if cnt % self.thinning == 0 and cnt >= sample_start:
                        to, spe = self.current_setting.output_species(
                            taxa_order=self.taxaorder
                        )
                        self.partitions.append(spe)
                        self.llhs.append(newlogl)
                        self.settings.append(self.current_setting)
                    accepted = accepted + 1
                else:
                    self.current_setting = self.last_setting
                    self.current_logl = self.last_logl
                    if cnt % self.thinning == 0 and cnt >= sample_start:
                        to, spe = self.current_setting.output_species(
                            taxa_order=self.taxaorder
                        )
                        self.partitions.append(spe)
                        self.llhs.append(self.current_logl)
                        self.settings.append(self.current_setting)

        print("Accptance rate: " + repr(float(accepted) / float(cnt)))
        print("Merge: " + repr(self.nmerge))
        print("Split: " + repr(self.nsplit))
        return self.partitions, self.llhs, self.settings


class FType(enum.Enum):
    nexus = "nexus"
    newick = "newick"


class Method(enum.Enum):
    H0 = "H0"
    H1 = "H1"
    H2 = "H2"
    H3 = "H3"


class BayesianPTP:
    """Run MCMC on multiple trees"""

    def __init__(
        self,
        filename: str,
        ftype: FType = FType.newick,
        reroot=False,
        method: Method = Method.H1,
        seed=1234,
        thinning=100,
        sampling=10000,
        burnin=0.1,
        firstktrees=0,
        taxa_order: list[str] = [],
    ):
        self.method = method
        self.seed = seed
        self.thinning = thinning
        self.sampling = sampling
        self.burnin = burnin
        self.firstktrees = firstktrees
        if ftype == "nexus":
            self.nexus = NexusReader(filename)
            self.nexus.blocks["trees"].detranslate()
            self.trees = self.nexus.trees.trees
        else:
            self.trees = self.raxmlTreeParser(filename)

        if self.firstktrees > 0 and self.firstktrees <= len(self.trees):
            self.trees = self.trees[: self.firstktrees]

        self.taxa_order = taxa_order
        if len(self.taxa_order) == 0:
            self.taxa_order = Tree(self.trees[0], format=1).get_leaf_names()
        self.numtaxa = len(self.taxa_order)
        self.numtrees = len(self.trees)
        self.reroot = reroot

    def remove_outgroups(self, ognames: list[str], remove=False, output=""):
        """reroot using outgroups and remove them"""
        self.reroot = False
        try:
            if remove:
                for og in ognames:
                    self.taxa_order.remove(og)
                self.numtaxa = len(self.taxa_order)
            for i in range(len(self.trees)):
                t = Tree(self.trees[i])
                if len(ognames) < 2:
                    t.set_outgroup(ognames[0])
                else:
                    ancestor = t.get_common_ancestor(ognames)
                    if not t == ancestor:
                        t.set_outgroup(ancestor)
                if remove:
                    t.prune(self.taxa_order, preserve_branch_length=True)
                self.trees[i] = t.write()
            if remove and output != "":
                with open(output, "w") as fout:
                    for t in self.trees:
                        fout.write(t + "\n")
        except ValueError as e:
            print(e)
            print("")
            print("")
            print("Somthing is wrong with the input outgroup names")
            print("")
            print("Quiting .....")
            sys.exit()

    def delimit(self):
        self.partitions = []
        self.llhs = []
        self.settings = []
        cnt = 1
        for tree in self.trees:
            print("Running MCMC sampling on tree " + repr(cnt) + ":")
            cnt = cnt + 1
            mcptp = ptpmcmc(
                tree=tree,
                reroot=self.reroot,
                startmethod=self.method,
                min_br=0.0001,
                seed=self.seed,
                thinning=self.thinning,
                sampling=self.sampling,
                burning=self.burnin,
                taxa_order=self.taxa_order,
            )
            pars, lhs, settings = mcptp.mcmc()
            self.maxhhlpar = mcptp.maxpar
            self.maxhhlsetting = mcptp.max_setting
            self.partitions.extend(pars)
            self.llhs.extend(lhs)
            self.settings.extend(settings)
            print("")
        return self.partitions, self.llhs, self.settings

    def raxmlTreeParser(self, fin):
        f = open(fin)
        lines = f.readlines()
        f.close()
        trees = []
        for line in lines:
            line = line.strip()
            if not line == "":
                trees.append(line[line.index("(") :])
        return trees

    def get_maxhhl_partition(self):
        return self.maxhhlpar


def print_run_info(
    tree_path: str,
    output_path: str,
    seed: int,
    thinning: int,
    sampling: int,
    burnin: float,
    nmi: bool,
    num_tree: int,
):
    print("bPTP finished running with the following parameters:")
    print(f" Input tree:.....................{tree_path}")
    print(f" MCMC iterations:................{thinning}")
    print(f" MCMC sampling interval:.........{sampling}")
    print(f" MCMC burn-in:...................{burnin:.2f}")
    print(f" MCMC seed:......................{seed}")
    print("")
    print(" MCMC samples written to:")
    print("  " + output_path + ".PTPPartitions.txt")
    print("")
    print(" Posterial LLH written to:")
    print("  " + output_path + ".PTPllh.txt")
    print("")
    print(" Posterial LLH plot:")
    print("  " + output_path + ".llh.pdf")
    print("")
    print(" Posterial Prob. of partitions written to:")
    print("  " + output_path + ".PTPPartitonSummary.txt")
    print("")
    print(" Highest posterial Prob. supported partition written to:")
    print("  " + output_path + ".PTPhSupportPartition.txt")
    print("  Tree plot written to:")
    print("  " + output_path + ".PTPhSupportPartition.txt.png")
    print("  " + output_path + ".PTPhSupportPartition.txt.svg")
    if nmi:
        print("")
        print(" MAX NMI partition written to:")
        print("  " + output_path + ".PTPhNMIPartition.txt")
    if num_tree == 1:
        print("")
        print(" Max LLH partition written to:")
        print("  " + output_path + ".PTPMLPartition.txt")
        print("  Tree plot written to:")
        print("  " + output_path + ".PTPMLPartition.txt.png")
        print("  " + output_path + ".PTPMLPartition.txt.svg")


def run_bptp(
    tree_path: Path,
    output_path: Path,
    file_type: FType = FType.newick,
    seed: int = 123,
    reroot: bool = False,
    delete: bool = False,
    method: Method = Method.H1,
    thinning: int = 10000,
    sampling: int = 100,
    burnin: float = 0.1,
    first_k_trees: int = 0,
    nmi: bool = False,
    scale: int = 500,
):

    bbptp = BayesianPTP(
        filename=str(tree_path),
        ftype=file_type,
        reroot=reroot,
        method=method,
        seed=seed,
        thinning=thinning,
        sampling=sampling,
        burnin=burnin,
        firstktrees=first_k_trees,
    )

    partitions, llhs, settings = bbptp.delimit()

    partition_parser = partitionparser(
        taxa_order=bbptp.taxa_order, partitions=partitions, llhs=llhs, scale=scale
    )

    if bbptp.numtrees == 1:
        partition_parser.summary(
            fout=str(output_path),
            bnmi=nmi,
            ML_par=bbptp.get_maxhhl_partition(),
            ml_spe_setting=bbptp.maxhhlsetting,
            sp_setting=settings,
        )
    else:
        partition_parser.summary(fout=str(output_path), bnmi=nmi, sp_setting=settings)

    min_no_p, max_no_p, mean_no_p = partition_parser.hpd_numpartitions()
    print(
        "Estimated number of species is between "
        + repr(min_no_p)
        + " and "
        + repr(max_no_p)
    )
    print("Mean: " + "{0:.2f}".format(mean_no_p))
    print("")
    print_run_info(
        str(tree_path),
        str(output_path),
        seed,
        thinning,
        sampling,
        burnin,
        nmi,
        bbptp.numtrees,
    )
