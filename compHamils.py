
import sys
import numpy as np
import scipy.io as sio
from scipy.io import loadmat

def main():
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_file.h5> <output_file.mat>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        # 1. Load the initial data
        data = loadmat(input_file)
        ham = data['entries']
        shapes = data['chunk_shapes']
        bounds = data['chunk_boundaries'].flatten().tolist()
        pairs = data['atom_pairs']
        ijcell_orig = pairs[0:3, :]

        # Process unique cells and re-index pairs
        ijcell, indcell = np.unique(ijcell_orig.T, axis=0, return_inverse=True)
        pairs = np.vstack([indcell, pairs[3:],])

        # 2. Extract and Reshape segments into chunks
        chunks = []
        for i in range(shapes.shape[1]):
            start, stop = bounds[i], bounds[i+1]
            segment = ham[0, start:stop]
            # Reshape based on matrix shape
            reshaped_chunk = segment.reshape(shapes[1, i], shapes[0, i]).T
            chunks.append(reshaped_chunk)

        # 3. Setup Assembly Parameters
        block_heights = np.array([19, 19, 14, 14, 14, 14])
        total_dim = sum(block_heights)
        row_offsets = np.insert(np.cumsum(block_heights), 0, 0)

        # 4. Assemble Hamiltonians
        hchunks = []
        n_values = np.unique(pairs[0, :])

        for n in n_values:
            ij = np.where(pairs[0, :] == n)[0]
            D0n = np.zeros((total_dim, total_dim), dtype=complex)

            for idx in ij:
                p = int(pairs[1, idx])
                q = int(pairs[2, idx])
                h1 = chunks[idx]

                r_start, r_end = row_offsets[p], row_offsets[p + 1]
                c_start, c_end = row_offsets[q], row_offsets[q + 1]

                D0n[r_start:r_end, c_start:c_end] = h1

            hchunks.append(D0n)

        # 5. Save the final result
        hchunks_array = np.array(hchunks)
        hchunks_matlab = np.moveaxis(hchunks_array, 0, -1)

        sio.savemat(output_file, {'chunks': hchunks_matlab, 'ijcell': ijcell})
        print(f"Successfully processed {input_file} and saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
