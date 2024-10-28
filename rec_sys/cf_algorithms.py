from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import data_util 
def complete_code(message):
    raise Exception(f"Please complete the code: {message}")
    return None

def center_and_nan_to_zero(matrix, axis=0):
    """ Center the matrix and replace nan values with zeros"""
    means = np.nanmean(matrix, axis=axis)
    matrix_centered = matrix - means
    return np.nan_to_num(matrix_centered)

def cosine_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def fast_cosine_sim(utility_matrix, vector, axis=0):
    """ Compute the cosine similarity between the matrix and the vector"""
    norms = np.linalg.norm(utility_matrix, axis=axis)
    um_normalized = utility_matrix / norms
    dot = np.dot(um_normalized.T, vector)  # Completed
    scaled = dot / np.linalg.norm(vector)
    return scaled

# New Function: Centered Cosine Similarity for Sparse Vectors
def centered_cosine_sim(vector1, vector2):
    """Compute centered cosine similarity between two sparse vectors."""
    vector1_float = vector1.astype(float)
    vector2_float = vector2.astype(float)

    x_nonzero_mean = np.mean(vector1_float.data) if vector1_float.nnz > 0 else 0
    y_nonzero_mean = np.mean(vector2_float.data) if vector2_float.nnz > 0 else 0
    
    x_centered = vector1_float.copy()
    y_centered = vector2_float.copy()
    
    x_centered.data -= x_nonzero_mean
    y_centered.data -= y_nonzero_mean
    
    numerator = x_centered.dot(y_centered.T).data[0] if x_centered.nnz > 0 and y_centered.nnz > 0 else 0
    denominator = np.linalg.norm(x_centered.data) * np.linalg.norm(y_centered.data)
    
    return numerator / denominator if denominator != 0 else 0

# New Function: Fast Centered Cosine Similarity for Sparse Matrix and Vector
def fast_centered_cosine_sim(matrix, vector):
    """Compute centered cosine similarity between each row of a sparse matrix and a sparse vector."""
    matrix_float = matrix.astype(float)
    vector_float = vector.astype(float)
    
    vector_mean = np.mean(vector_float.data) if vector_float.nnz > 0 else 0
    
    vector_centered = vector_float.copy()
    vector_centered.data -= vector_mean

    matrix_centered = lil_matrix(matrix.shape)
    for i in range(matrix.shape[0]):
        row = matrix_float.getrow(i)
        row_mean = np.mean(row.data) if row.nnz > 0 else 0
        row_centered = row.copy()
        row_centered.data -= row_mean
        matrix_centered[i] = row_centered
    
    matrix_centered = matrix_centered.tocsr()
    
    dot_products = matrix_centered.dot(vector_centered.T).toarray().flatten()
    matrix_norms = np.sqrt(matrix_centered.multiply(matrix_centered).sum(axis=1)).A1
    vector_norm = np.linalg.norm(vector_centered.data)
    norms = matrix_norms * vector_norm
    
    return np.divide(dot_products, norms, out=np.zeros_like(dot_products), where=norms != 0)

def rate_all_items(orig_utility_matrix, user_index, neighborhood_size):
    print(f"\n>>> CF computation for UM w/ shape: "
          + f"{orig_utility_matrix.shape}, user_index: {user_index}, neighborhood_size: {neighborhood_size}\n")

    clean_utility_matrix = center_and_nan_to_zero(orig_utility_matrix)
    user_col = clean_utility_matrix[:, user_index]
    similarities = fast_cosine_sim(clean_utility_matrix, user_col)

    def rate_one_item(item_index):
        if not np.isnan(orig_utility_matrix[item_index, user_index]):
            return orig_utility_matrix[item_index, user_index]

        users_who_rated = np.where(np.isnan(orig_utility_matrix[item_index, :]) == False)[0]
        best_among_who_rated = np.argsort(similarities[users_who_rated])[-neighborhood_size:]
        best_among_who_rated = users_who_rated[best_among_who_rated]
        best_among_who_rated = best_among_who_rated[np.isnan(similarities[best_among_who_rated]) == False]
        
        if best_among_who_rated.size > 0:
            rating_of_item = np.sum(similarities[best_among_who_rated] * orig_utility_matrix[item_index, best_among_who_rated]) / np.sum(similarities[best_among_who_rated])
        else:
            rating_of_item = np.nan
        print(f"item_idx: {item_index}, neighbors: {best_among_who_rated}, rating: {rating_of_item}")
        return rating_of_item

    num_items = orig_utility_matrix.shape[0]
    ratings = list(map(rate_one_item, range(num_items)))
    return ratings

#Task 4
def process_movielens_data():
    rated_by = {}
    user_col = []
    user_indices = {}
    
    dataset = data_util.load_movielens_tf('25M')
    user_count = 0
    
    for entry in dataset:
        user_id = entry['user_id']
        item_id = entry['item_id']
        rating = entry['rating']

        if item_id not in rated_by:
            rated_by[item_id] = []
        rated_by[item_id].append(user_id)

        if user_id not in user_indices:
            user_indices[user_id] = user_count
            user_count += 1
            user_col.append(csr_matrix((1, len(dataset)))) 
        
        user_index = user_indices[user_id]
        user_col[user_index][0, item_id] = rating 

    user_col = csr_matrix(user_col)

    return rated_by, user_col

# Run the processing function
rated_by, user_col = process_movielens_data()
# log results 
print("Rated by structure:", rated_by)
print("User column structure:", user_col)

# Unit Tests
# Task 2
def test_centered_cosine_sim():
    k = 100
    vector_x = csr_matrix(([i + 1 for i in range(k)], ([0] * k, range(k))), shape=(1, k))
    vector_y = csr_matrix(([k - i for i in range(k)], ([0] * k, range(k))), shape=(1, k))
    similarity = centered_cosine_sim(vector_x, vector_y)
    print("Test 1 - Centered Cosine Similarity:", similarity)

def test_centered_cosine_sim_with_nan():
    k = 100
    vector_x_data = [i + 1 if i % 10 != 0 else 0 for i in range(k)]
    vector_x = csr_matrix((vector_x_data, ([0] * k, range(k))), shape=(1, k))
    vector_y = csr_matrix(([k - i for i in range(k)], ([0] * k, range(k))), shape=(1, k))
    similarity = centered_cosine_sim(vector_x, vector_y)
    print("Test 2 - Centered Cosine Similarity with NaNs:", similarity)

def test_fast_centered_cosine_sim():
    k = 100
    utility_matrix_data = [[(i + 1) if j % 10 != 0 else 0 for j in range(k)] for i in range(k)]
    utility_matrix = csr_matrix(utility_matrix_data)
    vector = csr_matrix(([k - i for i in range(k)], ([0] * k, range(k))), shape=(1, k))
    similarities = fast_centered_cosine_sim(utility_matrix, vector)
    print("Test 3 - Fast Centered Cosine Similarity:", similarities)

# Run tests
test_centered_cosine_sim()
test_centered_cosine_sim_with_nan()
test_fast_centered_cosine_sim()
