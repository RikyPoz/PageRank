import numpy as np
from scipy import sparse

def build_sparse_link_matrix(filename):
    """
    Constructs the Link Matrix A in Sparse Format (CSR).
    
    CRITICAL OPTIMIZATION:
    This function does NOT physically apply the 'dangling node patch' (filling 
    columns with 1/N). Doing so would destroy sparsity and consume O(N^2) memory.
    Instead, it identifies dangling nodes so we can handle them 'virtually' 
    during the calculation phase.
    
    Args:
        filename (str): Path to the .dat file.
        
    Returns:
        tuple: (A_sparse, N, dangling_indices)
    """
    links = []
    out_degree = {}
    
    print(f"Reading file: {filename}...")
    try:
        with open(filename, 'r') as file:
            # 1. Read Header (N = Nodes, M = Edges)
            header = file.readline().strip().split()
            if not header: raise ValueError("File is empty or header is missing.")
            N = int(header[0])
            
            # 2. Skip URL mapping lines (we only need the graph structure)
            # The first N lines are strings/URLs which we don't need for the math.
            for _ in range(N): file.readline()
            
            # 3. Read Edges
            # Format: "Source_ID Target_ID"
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    src, tgt = int(parts[0]), int(parts[1])
                    
                    out_degree[src] = out_degree.get(src, 0) + 1
                    links.append((src, tgt))
                    
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, 0, None

    # --- SPARSE MATRIX CONSTRUCTION ---
    # We use the COO (Coordinate) format logic initially: separate lists for Data, Rows, Cols
    data = []
    rows = [] # Target Node (i)
    cols = [] # Source Node (j)
    
    # Identify Dangling Nodes (Nodes with 0 out-degree)
    # Initialize a boolean mask: assume ALL are dangling initially.
    is_dangling = np.ones(N, dtype=bool) 
    
    for src, tgt in links:
        # Adjust 1-based IDs from file to 0-based Indices for Python
        src_idx = src - 1
        tgt_idx = tgt - 1
        
        # If a node appears as a source, it has outgoing links -> Not Dangling
        is_dangling[src_idx] = False
        
        # Calculate transition probability: 1 / (Number of outgoing links)
        # This makes valid columns sum to 1.
        val = 1.0 / out_degree[src]
        
        rows.append(tgt_idx)
        cols.append(src_idx)
        data.append(val)
        
    # Convert to CSR (Compressed Sparse Row) format.
    # CSR is extremely efficient for Matrix-Vector multiplication (A @ x).
    A_sparse = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    
    # Extract indices of dangling nodes for later use
    dangling_indices = np.where(is_dangling)[0]
    
    print(f"Sparse Matrix Constructed: {N} nodes.")
    print(f"Valid Non-Zero Links: {len(data)}.")
    print(f"Dangling Nodes Identified: {len(dangling_indices)}.")
    
    return A_sparse, N, dangling_indices

def calculate_pagerank_sparse(A_sparse, N, dangling_indices, m, max_iter=200, tol=1e-7):
    """
    Computes PageRank using the Power Method with Sparse Optimizations.
    
    Mathematical Logic:
    x_new = (1-m)*Ax + (1-m)*[Dangling_Correction] + m/N
    
    Args:
        A_sparse: The sparse link matrix (missing dangling connections).
        N: Total nodes.
        dangling_indices: List of nodes that have no outgoing links.
        m: Teleportation probability (1-m is the damping factor).
        
    Returns:
        tuple: (PageRank Vector, Iterations)
    """
    
    # 1. Initialization
    # Start with uniform probability distribution (1/N for everyone)
    x = np.full(N, 1.0/N)
    
    # 2. Teleportation Constant
    # This is the "m * s" part of the formula.
    # Since s is [1/N, 1/N...], this term is just the scalar m/N added to every node.
    teleport_contribution = m / N
    
    # Ensure dangling_indices is a numpy array for fast indexing
    if not isinstance(dangling_indices, np.ndarray):
        dangling_indices = np.array(dangling_indices)
        
    iterations = 0
    
    # --- POWER METHOD LOOP ---
    for k in range(max_iter):
        x_prev = x.copy()
        
        # Step A: Standard Matrix Multiplication
        # This calculates flow only from nodes that have existing links.
        # Since A_sparse is sparse, this is O(Links) complexity, not O(N^2).
        Ax = A_sparse.dot(x_prev)
        
        # Step B: Implicit Dangling Node Handling
        # Since dangling columns are 0 in A_sparse, we "lost" the probability mass 
        # that was sitting on those nodes. We calculate how much was lost.
        dangling_mass_sum = np.sum(x_prev[dangling_indices])
        
        # We redistribute this lost mass evenly to ALL nodes (1/N), 
        # applied with the damping factor (1-m).
        dangling_correction = (1 - m) * (dangling_mass_sum / N)
        
        # Step C: Combine Everything
        # 1. Flow from Links (damped)
        # 2. Flow from Dangling Patch (damped)
        # 3. Flow from Random Teleport (m)
        x = (1 - m) * Ax + dangling_correction + teleport_contribution
        
        # Step D: Convergence Check (L1 Norm)
        diff = np.sum(np.abs(x - x_prev))
        iterations = k + 1
        
        if diff < tol:
            break
            
    return x, iterations

