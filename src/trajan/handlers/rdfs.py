from .base_handler import BASE
from trajan import constants
import numpy as np
import sys
import scipy as sp
import scipy.signal
import scipy.integrate

class RDFS(BASE):
    def __init__(self, args):
        super().__init__(args.file, args.verbose)

        self.cutoff = args.cutoff
        self.nbins = args.bincount

        self.outfile = args.outfile
        if self.outfile == constants.DEFAULT_OUTFILE:
            self.outfile = "rdfs_" + self.outfile

        self.parse_file()
        self.check_required_columns("type", "x", "y", "z")

        if args.total:
            self.calc_total = True
            self.mappings = np.array(args.total)
            present_types = self.get_types()
            if self.mappings.size != present_types.size:
                print(f"ERROR: Number of provided mappings ({self.mappings.size}) is not equal to the number of present types ({present_types.size}).")
                sys.exit(1)
            else:
                self.scat_lengths = np.zeros(shape = (np.max(present_types) + 1, ))
                for i in range(present_types.size):
                    try:
                        self.scat_lengths[present_types[i]] = float(self.mappings[i])
                    except:
                        if self.mappings[i] in constants.NEUTRON_SCATTERING_LENGTHS.keys():
                            self.scat_lengths[present_types[i]] = constants.NEUTRON_SCATTERING_LENGTHS[self.mappings[i]]
                        else:
                            print(f"Element ({self.mappings[i]}) is not currently supported)")
                            input_accepted = False
                            while not input_accepted:
                                scat_input = input("Please proivide custom neutron scattering length: ")
                                try:
                                    self.scat_lengths[present_types[i]] = float(scat_input)
                                    input_accepted = True
                                except:
                                    pass
        else:
            self.calc_total = False

        if args.pair and not self.calc_total:
            self.pairs = np.array(args.pair)
        else:
            if self.calc_total:
                print("WARNING: You have invoked the total correlation function calculations. For this each pairwise radial distribution function has to be computed. -p/--pairs argument is disregarded.\n")
            types = self.get_types()
            self.pairs = types[np.column_stack(np.triu_indices(types.size))]

        self.batch_size = args.batch_size

        self.broaden = False
        if hasattr(args, "broaden"):
            if not self.calc_total:
                print("WARNING: Broadening of the total correlation function requested, without the request for T(r) itself. This will be ignored.\n")
            else:
                self.broaden = True
                self.Qmax = args.broaden




    def analyze(self):
        timesteps = self.get_timesteps()
        boxes = self.get_boxes()
        sides = np.diff(boxes, axis = -1).reshape((-1, 3))
        volumes = np.prod(sides, axis = -1)
        self.bins = np.linspace(0, self.cutoff, self.nbins)
        self.hist_counts = np.zeros(shape = (self.pairs.shape[0], self.bins.size - 1), dtype = np.int64)
        r_inner = self.bins[:-1]
        r_outer = self.bins[1:]
        self.edges = (r_outer + r_inner)/2
        shell_volumes = (4/3) * np.pi * (r_outer**3 - r_inner**3)
        self.g_r = np.zeros(shape = (self.pairs.shape[0], self.bins.size - 1), dtype = np.float64)
        Nframes = self.get_Nframes()
        #For T(r) calculations
        type_counts = np.zeros(shape = (Nframes, np.max(self.pairs + 1)))

        for pair_idx, pair in enumerate(self.pairs):
            n1_rho2 = 0
            self.verbose_print(f"Calculating distribution for pair: {pair[0]} {pair[1]}", verbosity = 2)
            for i in range (Nframes):
                atoms1 = self.extract_positions(self.select_type(type = pair[0], frame = i))
                atoms2 = self.extract_positions(self.select_type(type = pair[1], frame = i))
                n1 = atoms1.shape[0]
                n2 = atoms2.shape[0]

                type_counts[i][pair[0]] = n1
                type_counts[i][pair[1]] = n2

                for btch_idx in range(0, n1, self.batch_size):
                    batch_atoms1 = atoms1[btch_idx : btch_idx + self.batch_size]
                    delta = batch_atoms1[:, np.newaxis, :] - atoms2[np.newaxis, :, :]
                    delta -= sides[i] * np.around(delta / sides[i])
                    square_dist = np.sum(delta**2, axis = 2)
                    square_dist = square_dist[square_dist <= self.cutoff**2]
                    if pair[0] == pair[1]:
                        square_dist = square_dist[square_dist > 0]
                    dists = np.sqrt(square_dist)
                    counts, _ = np.histogram(dists, bins = self.bins)
                    self.hist_counts[pair_idx] += counts

                n1_rho2 += n1 * n2/volumes[i]

                self.verbose_print(f"{i + 1} analysis of TS {timesteps[i]}", verbosity = 3)

            self.g_r[pair_idx] += self.hist_counts[pair_idx] / (shell_volumes * n1_rho2)

        if self.calc_total:
            total_atoms = self.get_atom_counts()
            fractions = np.mean(type_counts/total_atoms[:, np.newaxis], axis = 0)
            number_density = np.mean(total_atoms/volumes)

            average_scat = np.sum(fractions * self.scat_lengths)
            g_weighted_sum = np.zeros_like(self.edges)

            for pid, pair in enumerate(self.pairs):
                weight = np.prod(fractions[pair]) * np.prod(self.scat_lengths[pair])
                if pair[0] != pair[1]:
                    weight *= 2
                g_weighted_sum += weight * self.g_r[pid]

            self.T_r = 4 * np.pi * self.edges * number_density * g_weighted_sum * 0.01

            if self.broaden:

                T_0 = 4 * np.pi * number_density * self.edges * 0.01 * (average_scat**2)
                oscillations = self.T_r - T_0

                dr = self.edges[1] - self.edges[0]
                npoints = int(np.ceil(100 * 2 * np.pi / self.Qmax / dr))
                kernel_r = np.arange(-npoints, npoints + 1) * dr
                kernel = np.zeros_like(kernel_r)

                q_grid = np.linspace(0, self.Qmax, constants.FT_GRID_DENSITY)

                lorch_window = np.sinc(q_grid / self.Qmax)

                for i, r_val in enumerate(kernel_r):
                    integrand = np.cos(q_grid * r_val) * lorch_window

                    kernel[i] = (1 / np.pi) * sp.integrate.simpson(integrand, x=q_grid)

                broadened = sp.signal.convolve(oscillations, kernel, mode='same') * dr

                self.broad_T_r = T_0 + broadened


        print("Analysis complete")

    def write(self):
        data = np.column_stack((self.edges, self.g_r.T))
        header = "r," + ",".join([f"{pair[0]}-{pair[1]}" for pair in self.pairs])
        if self.calc_total:
            data = np.column_stack((data, self.T_r))
            header += ", T(r)"
        if self.broaden:
            data = np.column_stack((data, self.broad_T_r))
            header += ", broad T(r)"
        super().write(data = data,
                      header = header,
                      outfile = self.outfile,
                      )

    def statistics(self):
        verbosity = self.get_verbosity()
        #Name : (value, verbosity)
        stats_dict = dict()
        for pid, pair in enumerate(self.pairs):
            if verbosity > 2:
                peaks, _ = sp.signal.find_peaks(self.g_r[pid], height = 1.0, prominence = 1)
                stats_dict[f"Peaks ({pair[0]}-{pair[1]})"] = (" ".join(self.edges[peaks].astype(str)), 3)
            else:
                stats_dict[f"Max Peak ({pair[0]}-{pair[1]})"] = (self.edges[np.argmax(self.g_r[pid])], 2)


        super().statistics(stats_dict = stats_dict)

"""
import numpy as np

def calculate_neutron_total_T_r(partials, r_axis, density_rho0):
    Combines partial g(r)s into the Total Correlation Function T(r)
    as seen in Neutron Diffraction.
    
    Parameters:
    -----------
    partials : dict
        A dictionary containing your calculated g(r) arrays.
        Keys should be tuples like ('Si', 'O'), ('O', 'O').
        Values are the numpy arrays of g(r).
    r_axis : np.array
        The r (distance) array corresponding to the g(r)s.
    density_rho0 : float
        Total number density (Total Atoms / Box Volume) in atoms/Angstrom^3.
        
    Returns:
    --------
    T_r : np.array
        The Total Correlation Function.
    G_total : np.array
        The weighted Total Radial Distribution Function.
    
    # 1. Define Neutron Scattering Lengths (in femtometers or arbitrary units)
    # These are physical constants.
    b = {
        'Si': 4.149,
        'O':  5.803,
        'Na': 3.63
    }
    
    # 2. Define Concentrations (c_i = N_i / N_total)
    # You need to adjust these based on your specific glass composition
    # Example for Na2O-2SiO2 (NS2)
    total_atoms = 2 + 1 + 2 + 4 # 2Na, 1O, 2Si, 4O -> Na2Si2O5
    c = {
        'Si': 2 / 9.0,
        'Na': 2 / 9.0,
        'O':  5 / 9.0
    }
    
    # Initialize the sum
    g_weighted_sum = np.zeros_like(r_axis)
    normalization_factor = 0.0
    
    # 3. Sum the weighted partials
    # Formula: G_tot(r) = [ sum(ci * cj * bi * bj * g_ij(r)) ] / [ sum(ci * bi) ]^2
    
    # First, calculate the denominator (average scattering length squared)
    avg_b = sum(c[atom] * b[atom] for atom in c)
    denominator = avg_b**2
    
    print("Combining partials...")
    
    # Iterate through all unique pairs expected in the partials dict
    # Note: Ensure your partials dict has both ('Si', 'O') and ('O', 'Si') 
    # or handle the symmetry here. usually g_ij = g_ji.
    pairs = [('Si', 'Si'), ('Si', 'O'), ('Si', 'Na'),
             ('O', 'O'),   ('O', 'Na'), ('Na', 'Na')]
             
    for atom1, atom2 in pairs:
        key = (atom1, atom2)
        
        if key in partials:
            g_partial = partials[key]
        elif (atom2, atom1) in partials:
            g_partial = partials[(atom2, atom1)]
        else:
            print(f"Warning: Missing partial for {key}. Assuming 0.")
            g_partial = np.zeros_like(r_axis)

        # Weighting factor w_ij
        # If i == j: weight = c_i * c_i * b_i * b_i
        # If i != j: weight = 2 * c_i * c_j * b_i * b_j (The 2 accounts for i-j and j-i symmetry)
        
        weight = c[atom1] * c[atom2] * b[atom1] * b[atom2]
        if atom1 != atom2:
            weight *= 2
            
        g_weighted_sum += weight * g_partial
        
    # 4. Calculate G_total (The Neutron Weighted g(r))
    G_total = g_weighted_sum / denominator
    
    # 5. Calculate T(r)
    # T(r) = 4 * pi * r * rho0 * G_total(r)
    T_r = 4 * np.pi * r_axis * density_rho0 * G_total
    
    # Optional: Calculate the baseline T^0(r) shown in your image
    # This is the straight line slope
    T_zero = 4 * np.pi * r_axis * density_rho0
    
    return T_r, T_zero, G_total
"""
