import sys
import numpy as np
from scipy.spatial import cKDTree
import argparse

from .handlers import ANGLE, QUNIT

from . import constants
from . import utils



def parse_args():
    parser = utils.ErrorHandlingParser(prog = "trajan", formatter_class = utils.NoMetavarHelpFormatter)

    parser.add_argument("file", help = "LAMMPS trajectory file to be analized.", type = str)

    parser.add_argument("-v", "--verbose", help = f"Screen log verbosity (1 - {constants.MAX_VERBOSITY}). Default: {constants.DEFAULT_VERBOSITY}", type = utils.verbosity_type(), default = constants.DEFAULT_VERBOSITY)

    parser.add_argument("-o", "--outfile", help = f"Name of the ouput data file", type = str, default = constants.DEFAULT_OUTFILE)

    #parser.add_argument("-s", "--steps", help = "Pattern for specifying which steps to use", type = str)

    subparsers = parser.add_subparsers(dest="command", required=True, metavar = "analyzer", action = utils.StrictSubParsersAction)

    bond_angle = subparsers.add_parser("angle", help = "Argument parser for extracting bond angle distributions from LAMMPS-generated trajectory files.", formatter_class = utils.NoMetavarHelpFormatter)

    bond_angle.add_argument("types", nargs = 3, type = int, default = [1, 1, 1], help = "Integer values representing atomic types of a bonded triplet with the atom of second type in the middle.")

    #bond_angle.add_argument("-m", "--maxcut", nargs ="+", type = float, default = [], help = "Maximum distance allowed for nearest species of a triplet to be considered bonded.")

    bond_angle.add_argument("-b", "--bincount", type = int, default = constants.DEFAULT_HIST_BINCOUNT, help = f"Number of bins for the bond angle histogram. Default: {constants.DEFAULT_HIST_BINCOUNT}.")

    bond_angle.set_defaults(handler_class = ANGLE)


    qunit = subparsers.add_parser("qunit", help = "Argument parser for calculating Q-unit distributions from LAMMPS-generated trajectory files.", formatter_class = utils.NoMetavarHelpFormatter)
    qunit.add_argument("types", nargs = "+", type = int, default = [], help = "Integer values representing atomic types of network formers and connectors separated by a zero. Any connector bonding two network formers will contribute to the Qunit count of both of the network formers. Exmaple of input: 1 2 3 0 4 5; here atoms of types 1, 2, and 3 anre considered network formers and atoms of types 4 and 5 are considered network connectors.")
    qunit.add_argument("-c", "--cutoffs", nargs = "+", type = float, default = [], help = "Maximum distances between each former and connector (in Angstroms) for them to be considered bonded. Default: inf. Example: 1.1 1.2 1.3 1.4. If types supplied are (1 2 0 3 4) then C(1, 3) = 1.1, C(1, 4) = 1.2, C(2, 3) = 1.3, and C(2, 4) = 1.4.")

    qunit.set_defaults(handler_class = QUNIT)

    return parser.parse_args()


def main():
    args = parse_args()

    handler = args.handler_class(args)

    handler.analyze()

    handler.statistics()

    handler.write()


if __name__ == "__main__":
    main()
