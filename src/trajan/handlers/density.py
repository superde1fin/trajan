from .base_handler import BASE
from trajan import constants
import numpy as np
import sys

class DENSITY(BASE):
    def __init__(self, args):
        super().__init__(args.file, args.verbose)

        self.masses = list()
        for el in args.elements:
            try:
                self.masses.append(float(el))
            except ValueError:
                self.masses.append(constants.ATOMIC_MASSES[el])


        self.outfile = args.outfile
        if args.units in constants.AVAILABLE_UNITS:
            self.units = args.units
        else:
            print(f"ERROR: Unit set \"{args.units}\" is not supported. Please choose out of available unit sets: \n\n{"\n".join(constants.AVAILABLE_UNITS)}\n")
            sys.exit(1)

        self.parse_file()

        num_masses = len(self.masses)
        max_ntype = np.max(self.ntypes)
        if num_masses != max_ntype:
            print(f"ERROR: Incorrect number of masses given ({num_masses}) while at least one of the timesteps has ({max_ntype}) types.")
            sys.exit(1)

        self.masses = np.concatenate(([0], self.masses))

        self.wrap_positions()

        self.densities = list()

    def analyze(self):
        for i in range (self.Nframes):
            all_types = self.atomic_data[i][:, self.columns["type"]].astype(int)
            all_masses = self.masses[all_types]
            density = np.sum(all_masses) / (np.prod(self.lengths[i])) #amu/A^3
            density *= constants.MASS_CONVERSIONS[self.units] / np.power(constants.DISTANCE_CONVERSIONS[self.units], 3)
            self.densities.append(density)

            self.verbose_print(f"{i + 1} analysis of TS {self.timesteps[i]}", verbosity = 2)

        self.densities = np.array(self.densities)
        print("Analysis complete")

    def write(self):

        super().write(data = np.column_stack((np.arange(1, self.timesteps.shape[0] + 1), self.timesteps, self.densities)),
                      header = "frame, time step, density",
                      outfile = self.outfile,
                      )

    def statistics(self):
        #Name : (value, verbosity)
        stats_dict = {"Mean density" : (np.mean(self.densities), 1),
                      "Standard deviation" : (np.std(self.densities), 1),
        }

        super().statistics(stats_dict = stats_dict)
