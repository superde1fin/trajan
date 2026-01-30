from .base_handler import BASE
from trajan import constants

import numpy as np
import sys
from collections import deque

class VDOS(BASE):
    def __init__(self, args):
        super().__init__(args.file, args.verbose, args.steps, args.buffer)


        self.timestep = args.timestep * self.get_frame_step()
        self.max_timelag = args.max_timelag / self.timestep
        if args.lag_step is None:
            self.lag_step = constants.DEFAULT_LAG_STEP
        else:
            self.lag_step = int(args.lag_step / self.timestep)

        if self.lag_step < self.timestep:
            print(f"ERROR: The velocity autocorrelation function resolution ({args.lag_step}) cannot be smaller than the time between frames ({args.timestep}).")
            sys.exit(1)
        self.units = args.units
        self.timestep *= constants.TIMESTEPS[self.units]

        self.outfile = args.outfile
        if self.outfile == constants.DEFAULT_OUTFILE:
            self.outfile = "vDOS_" + self.outfile

        self.taper_fraction = args.taper
        self.padding_fraction = args.padding

        self.parse_file()

        self.check_required_columns("vx", "vy", "vz")

        self.autocorrel = list()
        self.autocount = list()

    def analyze(self):
        velocities = list()
        finite_cap = self.max_timelag != np.inf
        if finite_cap:
            max_storage = int(self.max_timelag // self.lag_step) + 1
            self.autocorrel = np.zeros(shape = (max_storage, ))
            self.autocount = np.zeros(shape = (max_storage, ))
        else:
            max_storage = None

        self.history = deque(maxlen = max_storage)

        for frame_idx in self.trajectory_reader():
            if (not finite_cap) and (frame_idx < max_storage):
                self.autocorrel.append(0)
                self.autocount.append(0)

            current_velocities = self.extract_columns("vx", "vy", "vz")
            self.history.append(current_velocities)

            for tau, past_velocities in enumerate(reversed(self.history)):
                if tau % self.lag_step:
                    continue

                storage_idx = tau // self.lag_step

                self.autocount[storage_idx] += 1
                self.autocorrel[storage_idx] += np.einsum("ij,ij->", current_velocities, past_velocities) / current_velocities.shape[0]

            self.verbose_print(f"{frame_idx} analysis of TS {self.get_timestep()}", verbosity = 2)

        self.autocorrel = np.array(self.autocorrel) /  np.array(self.autocount)
        self.autocorrel /= self.autocorrel[0]

        num_taper = len(self.autocorrel)

        window = np.ones(num_taper)
        num_taper = int(num_taper * self.taper_fraction)
        padding =  int(num_taper * self.padding_fraction)
        if num_taper > 0:
            x = np.linspace(0, np.pi, num_taper)
            decay = 0.5 * (1 + np.cos(x))
            window[-num_taper:] = decay

        self.tapered_autocorrel = self.autocorrel * window

        vdos_complex = np.fft.rfft(self.tapered_autocorrel, n = padding)
        self.vDOS = vdos_complex.real

        #Conversion from Hz to cm^-1
        self.frequencies = np.fft.rfftfreq(padding, d = self.timestep * self.lag_step) * 33.35641e-12

    def write(self):
        super().write(data = np.column_stack((self.frequencies, self.vDOS)),
                      header = "Frequency (cm^-1), vDOS",
                      outfile = self.outfile,
                      )
