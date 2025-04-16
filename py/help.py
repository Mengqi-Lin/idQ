def brute_check_id1(Q_basis):
    """
    Perform candidate generation on the basis matrix Q_basis to check the identifiability.
    
    Q_basis is the matrix obtained after removing all-zero, duplicate, and generated rows.
    
    The function generates candidate Q_basis_bar matrices by modifying the replaceable rows of Q_basis.
    It returns a tuple (status, Q_basis_bar) where:
      - status == 0 indicates that Q_basis is not identifiable and Q_basis_bar is a candidate,
      - status == 1 indicates that Q_basis is identifiable (and no candidate is found).
    
    Note: Lifting Q_basis_bar back to the full Q will be handled in the main function.
    """
    J_basis, K = Q_basis.shape
    distances = distances2U(Q_basis)
    irreplaceable_rows = np.where(np.array(distances) == K - 1)[0]
    replaceable_rows = set(range(J_basis)) - set(irreplaceable_rows)
    
    replacement_indices = list(replaceable_rows)
    subQ_bars = []
    for i in range(len(replacement_indices)):
        index = replacement_indices[i]
        q_bars = []
        for p in range(1, K - distances[index] + 1):
            q_bars.extend(generate_binary_vectors(K, p))
        valid_q_bars = []
        Q_temp = Q_basis.copy()
        for q_bar in q_bars:
            Q_temp[index, :] = q_bar
            if preserve_partial_order(Q_basis, Q_temp, set(irreplaceable_rows), [index]):
                valid_q_bars.append(q_bar)
        subQ_bars.append(valid_q_bars)
    
    # Generate Cartesian product of candidate replacements.
    subQ_bars = itertools.product(*subQ_bars)
    Q_bar_gen = generate_unique_Q_bars(subQ_bars, Q_basis, replacement_indices)


    cols_phiQ = unique_response_columns(Q_basis)
    for Q_basis_bar in Q_bar_gen:
        candidate_valid = True  
        for S in cols_phiQ:
            S_arr = np.array(S, dtype=int)
            idx_active = [j for j in range(J_basis) if S_arr[j] == 1]

            # Compute h_a: the logical OR of the rows of Q_basis_bar indexed by idx_active.
            if idx_active:
                h_a = Q_basis_bar[idx_active[0]].copy()
                for j in idx_active[1:]:
                    h_a = np.logical_or(h_a, Q_basis_bar[j]).astype(int)
            else:
                h_a = np.zeros(K, dtype=int)

            # Now check if there is any row j not in idx_active such that Q_basis_bar[j] <= h_a.
            # If so, then the response computed from h_a would have an extra 1, differing from S.
            for j in range(J_basis):
                if j in idx_active:
                    continue  # These rows are by construction "covered"
                if np.all(Q_basis_bar[j] <= h_a):
                    # Found an extra match not corresponding to S.
                    candidate_valid = False
                    break  # No need to check further S for this candidate.

            if not candidate_valid:
                # One S failed, so break out immediately to try the next Q_basis_bar.
                break

        if candidate_valid:
            return 0, Q_basis_bar
    return 1, None 

### This function checks if Q is identifiable, if not, it returns one possible Q_bar.
def identifiability1(Q):
    Q = Q.copy()
    
    # Step 1: get Q_basis
    (Q_basis, basis_to_original, orig_indices_for_basis,
         Q_unique, unique_to_original, basis_to_unique) = get_basis(Q)
    
    J_basis, K = Q_basis.shape
    
    if J_basis < K:
        Q_basis_bar = np.eye(K, dtype=int)[:J_basis]
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        print("Q_basis with dimensions J < K, thus not identifiable.")
        return 0, Q_bar
    # All subsequent candidate generation is performed on Q_basis.
    # Step 3: Check for trivial non-identifiability on Q_basis.
    for k in range(K):
        if np.all(Q_basis[:, k] == 0):
            Q_basis_bar = Q_basis.copy()
            Q_basis_bar[:, k] = 1
            print("Q is trivially not identifiable (all zero column).")
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            return 0, Q_bar
        if np.all(Q_basis[:, k] == 1):
            Q_basis_bar = Q_basis.copy()
            Q_basis_bar[:, k] = 0
            print("Q is trivially not identifiable (all one column).")
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            return 0, Q_bar

    candidate = check_two_column_submatrices(Q_basis)
    if candidate is not None:
        print("Q is not identifiable (two-column submatrix not identifiable).")
        Q_basis_bar = candidate
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        return 0, Q_bar

    candidate = check_three_column_submatrices(Q_basis)
    if candidate is not None:
        print("Q is not identifiable (three-column submatrix not identifiable).")
        Q_basis_bar = candidate
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        return 0, Q_bar

    # Step 4: Determine identifiability of Q_basis.
    if direct_check_id(Q_basis):
        print("Q is identifiable (direct_check).")
        return 1, None
    else:
        status, Q_basis_bar = brute_check_id1(Q_basis)
        if status == 0:
            print("Q is not identifiable (brute_check_id).")
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            return 0, Q_bar
        else:
            print("Q is identifiable (brute_check).")
            return 1, None
def brute_check_id2(Q_basis):
    J_basis, K = Q_basis.shape
    distances = distances2U(Q_basis)
    irreplaceable_rows = np.where(np.array(distances) == K - 1)[0]
    replaceable_rows = set(range(J_basis)) - set(irreplaceable_rows)
    
    replacement_indices = list(replaceable_rows)
    subQ_bars = []
    for i in range(len(replacement_indices)):
        index = replacement_indices[i]
        q_bars = []
        for p in range(1, K - distances[index] + 1):
            q_bars.extend(generate_binary_vectors(K, p))
        valid_q_bars = []
        Q_temp = Q_basis.copy()
        for q_bar in q_bars:
            Q_temp[index, :] = q_bar
            if preserve_partial_order(Q_basis, Q_temp, set(irreplaceable_rows), [index]):
                valid_q_bars.append(q_bar)
        subQ_bars.append(valid_q_bars)
    
    # Generate Cartesian product of candidate replacements.
    subQ_bars = itertools.product(*subQ_bars)
    Q_bar_gen = generate_unique_Q_bars(subQ_bars, Q_basis, replacement_indices)


    cols_phiQ = unique_response_columns(Q_basis)
    for Q_basis_bar in Q_bar_gen:
        cols_phiQbar = unique_response_columns(Q_basis_bar)
        if cols_phiQ.issubset(cols_phiQbar):
            return 0, Q_basis_bar
    return 1, None

### This function checks if Q is identifiable, if not, it returns one possible Q_bar.
def identifiability2(Q):
    Q = Q.copy()
    
    # Step 1: get Q_basis
    (Q_basis, basis_to_original, orig_indices_for_basis,
         Q_unique, unique_to_original, basis_to_unique) = get_basis(Q)
    
    J_basis, K = Q_basis.shape
    
    if J_basis < K:
        Q_basis_bar = np.eye(K, dtype=int)[:J_basis]
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        print("Q_basis with dimensions J < K, thus not identifiable.")
        return 0, Q_bar
    # All subsequent candidate generation is performed on Q_basis.
    # Step 3: Check for trivial non-identifiability on Q_basis.
    for k in range(K):
        if np.all(Q_basis[:, k] == 0):
            Q_basis_bar = Q_basis.copy()
            Q_basis_bar[:, k] = 1
            print("Q is trivially not identifiable (all zero column).")
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            return 0, Q_bar
        if np.all(Q_basis[:, k] == 1):
            Q_basis_bar = Q_basis.copy()
            Q_basis_bar[:, k] = 0
            print("Q is trivially not identifiable (all one column).")
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            return 0, Q_bar

    candidate = check_two_column_submatrices(Q_basis)
    if candidate is not None:
        print("Q is not identifiable (two-column submatrix not identifiable).")
        Q_basis_bar = candidate
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        return 0, Q_bar

    candidate = check_three_column_submatrices(Q_basis)
    if candidate is not None:
        print("Q is not identifiable (three-column submatrix not identifiable).")
        Q_basis_bar = candidate
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        return 0, Q_bar

    # Step 4: Determine identifiability of Q_basis.
    if direct_check_id(Q_basis):
        print("Q is identifiable (direct_check).")
        return 1, None
    else:
        status, Q_basis_bar = brute_check_id2(Q_basis)
        if status == 0:
            print("Q is not identifiable (brute_check_id).")
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            return 0, Q_bar
        else:
            print("Q is identifiable (brute_check).")
            return 1, None
    