from .base_handler import BASE
from trajan import constants
import numpy as np
import sys
import collections

class RINGS(BASE):
    def __init__(self, args):
        super().__init__(args.file, args.verbose, args.steps)

        self.outfile = args.outfile
        if self.outfile == constants.DEFAULT_OUTFILE:
            self.outfile = "rings_" + self.outfile

        if 0 in args.types:
            zero_index = args.types.index(0)
            self.base = np.array(args.types[:zero_index])
            self.connectors = np.array(args.types[zero_index + 1:])
        else:
            print("ERROR: There must be a separating 0 in atom types. Please refer to the help message for the ring size analyzer.")
            sys.exit(1)


        nbase = len(self.base)
        nconnectors = len(self.connectors)
        ncuts = len(args.cutoffs)

        if args.cutoffs and ncuts != nbase * nconnectors:
            print(f"ERROR: number of cutoffs ({ncuts}) is not equal to the number of base ({nbase}) atoms times the number of connector atoms ({nconnectors}).")
            sys.exit(1)

        self.parse_file()

        self.check_required_columns("type", "x", "y", "z")

        types = self.get_types()
        max_type = np.max(types)
        self.interaction_matrix = np.full((max_type + 1, max_type + 1), np.inf)

        if args.cutoffs:
            cutoff_iter = iter(args.cutoffs)
            for b in self.base:
                if not b in types:
                    print(f"ERROR: Incorrect base atom type ({b}). Types present in trajectories: {' '.join(types.astype(str))}.")
                    sys.exit(1)
                for c in self.connectors:
                    if not c in types:
                        print(f"ERROR: Incorrect connector type ({c}). Types present in trajectories: {' '.join(types.astype(str))}.")
                        sys.exit(1)
                    cut = next(cutoff_iter)
                    self.interaction_matrix[b, c] = cut
                    self.interaction_matrix[c, b] = cut

        self.conbonds = args.connector_bonds
        nconbonds = len(self.conbonds)
        if nconbonds < nconnectors:
            if self.conbonds:
                print(f"WARNING: Number of connectors supplied ({nconnectors}) is not the same as the number of connector bonds arguments ({nconbonds}). Only the first {nconbonds} onnectors will have non default values.")
            for i in range(nconnectors - nconbonds):
                self.conbonds.append(constants.DEFAULT_CONNECTOR_BONDS)
        elif nconbonds > nconnectors:
            print(f"WARNING: Too many connector bond arguments provided. Only the first ({nconnectors}) arguments will be used.")
            self.conbonds = self.conbonds[:nconnectors]

        self.conbonds = np.array(self.conbonds)

        self.max_depth = args.max_size

        self.all_rings = list()
        self.ring_values = np.arange(self.max_depth)


        self.wrap_positions()


    def analyze(self):
        columns = self.get_columns()
        for frame_idx in self.trajectory_reader():
            num_each_type = self.get_num_each_type()
            num_neighs = int(2 * np.sum(num_each_type[self.connectors] * self.conbonds) / np.sum(num_each_type[self.base]))
            base_atoms = self.select_types(types = self.base)
            base_atom_positions = self.extract_positions(base_atoms)

            num_base_atoms = base_atoms.shape[0]

            frame_graph = np.full(shape = (num_base_atoms, num_neighs), fill_value = -1)
            neighs_recorder = np.zeros(shape = (num_base_atoms, ), dtype = np.int32)
            ring_participation_counts = np.zeros(shape = (num_base_atoms, self.max_depth))

            for cid, connector in enumerate(self.connectors):
                target_connectors = self.select_types(types = self.connectors)
                target_connector_positions = self.extract_positions(target_connectors)
                connector_types = target_connectors[:, columns["type"]].astype(int)

                numbonds = self.conbonds[cid]
                norms, idx = self.get_nclosest(
                    central = target_connector_positions,
                    neighs = base_atom_positions,
                    N = numbonds,
                )

                base_types = base_atoms[idx][..., columns["type"]].astype(int)
                cutoff_grid = self.interaction_matrix[connector_types[:, None], base_types]

                distance_mask = norms <= cutoff_grid
                idx = idx[distance_mask.all(axis = 1)]


                #Implement frame graph expansion if truly needed

                sources = list()
                targets = list()

                #Create flat lists of base atoms connected through a connector
                for i in range(numbonds):
                    for j in range(numbonds):
                        if i == j: continue
                        sources.append(idx[:, i])
                        targets.append(idx[:, j])

                all_sources = np.concatenate(sources)
                all_targets = np.concatenate(targets)

                sort_order = np.argsort(all_sources)
                all_sources = all_sources[sort_order]
                all_targets = all_targets[sort_order]

                #Find at which index repetition starts
                unique_atoms, start_indices = np.unique(all_sources, return_index = True)
                #Calculate how many source atoms repeated (num of neighs for each base)
                neigh2add = np.diff(np.append(start_indices, len(all_sources)))

                #Get relative position of each target atom in original graph
                group_offsets = np.repeat(start_indices, neigh2add)
                relcol = np.arange(len(all_sources)) - group_offsets

                #Fetch how many neighbors each base atom has already
                shift = neighs_recorder[all_sources]
                #Shift relative column positions for new neighbors
                relcol = relcol + shift
                if np.max(relcol) >= num_neighs:
                    new_frame_graph = np.zeroes((num_base_atoms, num_neighs * 2), dtype = np.int32)
                    new_frame_graph[:, num_neighs] = frame_graph
                    num_neighs *= 2
                    frame_graph = new_frame_graph

                frame_graph[all_sources, relcol] = all_targets

                neighs_recorder[unique_atoms] += neigh2add



            visited_token = np.full(num_base_atoms, fill_value = -1, dtype = np.int32)
            dist = np.zeros(num_base_atoms, dtype = np.int32)
            parent = np.full(num_base_atoms, fill_value = -1, dtype = np.int32)
            branch_id = np.full(num_base_atoms, -1, dtype = np.int32)

            queue = collections.deque()

            for start_node in range(num_base_atoms):

                if frame_graph[start_node, 0] == -1:
                    continue

                current_token = start_node
                queue.clear()

                queue.append(start_node)
                visited_token[start_node] = current_token
                dist[start_node] = 0
                parent[start_node] = -1
                branch_id[start_node] = -1

                found_ring = False

                while queue:
                    node = queue.popleft()
                    #print(node)
                    if dist[node] >= (self.max_depth / 2 ) + 1:
                        continue

                    for neigh in frame_graph[node]:
                        if neigh == -1: break

                        if neigh == parent[node]: continue

                        if visited_token[neigh] == current_token:
                            if branch_id[node] != branch_id[neigh]:
                                current_size = dist[node] + dist[neigh] + 1

                                bt_atom = node
                                while parent[bt_atom] != -1:
                                    ring_participation_counts[bt_atom, current_size] += 1
                                    bt_atom = parent[bt_atom]

                                bt_atom = neigh
                                while bt_atom != -1:
                                    ring_participation_counts[bt_atom, current_size] += 1
                                    bt_atom = parent[bt_atom]



                                found_ring = True
                                break

                        else:
                            visited_token[neigh] = current_token
                            dist[neigh] = dist[node] + 1
                            parent[neigh] = node
                            if node == start_node:
                                branch_id[neigh] = neigh
                            else:
                                branch_id[neigh] = branch_id[node]
                            queue.append(neigh)

                    if found_ring: break

            self.all_rings.append(np.mean(ring_participation_counts, axis = 0))
            self.verbose_print(f"{frame_idx} analysis of TS {self.get_timestep()}", verbosity = 2)

        self.all_rings = np.array(self.all_rings)


        print("Analysis complete")

    def write(self):

        super().write(data = np.column_stack((self.ring_values, self.mean_rings)),
                      header = "Ring size, ",
                      outfile = self.outfile,
                      )

    def statistics(self):
        self.mean_rings = np.mean(self.all_rings, axis = 0)
        self.deviations = np.std(self.all_rings, axis = 0)
        #Name : (value, verbosity)
        stats_dict = {"Average ring size" : (np.sum(self.ring_values * self.mean_rings / np.sum(self.mean_rings)), 1),
        }


        super().statistics(stats_dict = stats_dict)
