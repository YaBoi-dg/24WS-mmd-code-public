# Artur Andrzejak, October 2024
# Algorithms for latent factor models

# Limit size of GPU memory pre-allocated by jax
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import dataclasses
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds


def init_latent_factors(num_users, num_items, num_factors, rng_key):
    """ Initialize latent factors for users and items
    """
    key_u, key_v = jax.random.split(rng_key)
    matrix_u = jax.random.normal(key_u, (num_items, num_factors))
    matrix_v = jax.random.normal(key_v, (num_factors, num_users))
    return matrix_u, matrix_v


def load_data_and_init_factors(config):
    # load the dataset using TensorFlow Datasets
    import data_util as data

    ratings_tf, user_ids_voc, movie_ids_voc = data.load_movielens_tf(config)
    num_users = len(user_ids_voc.get_vocabulary())
    num_items = len(movie_ids_voc.get_vocabulary())
    rng_key_factors, rng_key_r = jax.random.split(jax.random.PRNGKey(config.rng_seed))
    matrix_u, matrix_v = init_latent_factors(num_users, num_items, config.num_factors, rng_key_factors)
    return ratings_tf, matrix_u, matrix_v, num_users, num_items


def predict_and_compare(mat_u, mat_v, train_ds, config):
    """ Predict ratings for the test dataset, compare to target ratings
        Returns a list of tuples with (predicted, target) ratings"""

    predictions_and_targets = []
    # Only batch_size=1 is supported for now

    for idx, record in enumerate(tfds.as_numpy(train_ds.batch(1))):
        # Batch sizes > 1 compute too many predictions (all pairs of users and items)
        # i, j, rating = record["user_id"], record["movie_id"], record["user_rating"]
        # rating_pred = jnp.dot(mat_u[i, :], mat_v[:, j])
        i, j, rating = record["movie_id"][0], record["user_id"][0], float(record["user_rating"][0])
        rating_pred = float(jnp.dot(mat_u[i, :], mat_v[:, j]))
        predictions_and_targets.append((rating_pred, rating))
        if idx >= config.num_records_predict_and_compare:
            break
    return predictions_and_targets


def mse_loss_all_batches(mat_u, mat_v, dataset, batch_size):
    """ Compute mse per batch using vectorized operations
        Returns a list of mse values for all batches as floats
    """
    mse_all_batches = []
    for record in tfds.as_numpy(dataset.batch(batch_size)):
        mse = mse_loss_one_batch(mat_u, mat_v, record)
        mse_all_batches.append(mse)
    # convert list of arrays to list of floats
    mse_all_batches = list(map(float, mse_all_batches))
    return mse_all_batches


@jax.jit  # Comment out for single-step debugging
def mse_loss_one_batch(mat_u, mat_v, record):
    """This colab experiment motivates the implementation:
    https://colab.research.google.com/drive/1c0LpSndbTJaHVoLTatQCbGhlsWbpgvYh?usp&#x3D;sharing=
    """
    rows, columns, ratings = record["movie_id"], record["user_id"], record["user_rating"]
    estimator = -(mat_u @ mat_v)[(rows, columns)]
    square_err = jnp.square(estimator + ratings)
    mse = jnp.mean(square_err)
    return mse


def uv_factorization_dense_um(mat_u, mat_v, mat_um, num_epochs=1, learning_rate=0.01, reg_param=0.1):
    """ Matrix factorization using stochastic gradient descent (SGD) for a dense utility matrix
        Terribly slow implementation, just for illustration purposes
    Args:
        mat_u: user factor matrix
        mat_v: item factor matrix
        mat_um: utility matrix
        num_epochs: number of iterations
        learning_rate: step size for gradient descent
        reg_param: regularization parameter
    Returns:
        U: learned user matrix
        V: learned item matrix
    """
    for epoch in range(num_epochs):
        # Will be printed during tracing, not execution!
        print(f"In uv_factorization_dense_um, starting epoch {epoch}")
        for i in range(mat_um.shape[0]):
            for j in range(mat_um.shape[1]):
                eij = mat_um[i, j] - jnp.dot(mat_u[i, :], mat_v[:, j])
                # Will be printed during tracing, not execution!
                # print (f"Current i,j, eij are: {i}, {j}, {eij}")
                for k in range(mat_u.shape[1]):
                    mat_u = mat_u.at[i, k].add(learning_rate * (eij * mat_v[k, j] - reg_param * mat_u[i, k]))
                    mat_v = mat_v.at[k, j].add(learning_rate * (eij * mat_u[i, k] - reg_param * mat_v[k, j]))
    return mat_u, mat_v


# jit-compile the function, pointing out the static arguments
uv_factorization_dense_um = jax.jit(uv_factorization_dense_um,
                                    static_argnames=("num_epochs", "learning_rate", "reg_param"))


def uv_factorization_tf_slow(mat_u, mat_v, train_ds, config):
    """ Matrix factorization using stochastic gradient descent (SGD) for sparse (and batched) utility matrix
        A terribly slow implementation, just for illustration purposes
    """

    def update_uv_for_record(config, mat_u, mat_v, record):
        lr, rp = 0.1, config.reg_param
        i, j, rating = record["movie_id"], record["user_id"], record["user_rating"]
        # Convert np arrays to scalars - only ok for batch size 1!
        i, j, rating = i[0], j[0], rating[0]
        abs_error_ij = rating - jnp.dot(mat_u[i, :], mat_v[:, j])
        for k in range(mat_u.shape[1]):
            mat_u = mat_u.at[i, k].add(lr * (abs_error_ij * mat_v[k, j] - rp * mat_u[i, k]))
            mat_v = mat_v.at[k, j].add(lr * (abs_error_ij * mat_u[i, k] - rp * mat_v[k, j]))
        return mat_u, mat_v

    for epoch in range(config.num_epochs):
        print(f"In uv_factorization_tf_slow, starting epoch {epoch}")
        for record in tfds.as_numpy(train_ds.batch(config.batch_size_training).prefetch(config.batch_size_training)):
            mat_u, mat_v = update_uv_for_record(config, mat_u, mat_v, record)
    return mat_u, mat_v


def uv_factorization_vec_no_reg(mat_u, mat_v, train_ds, valid_ds, config):
    """ Matrix factorization using SGD without regularization
        Fast vectorized implementation using JAX
    """

    @jax.jit  # Comment out for single-step debugging
    def update_uv(mat_u, mat_v, record, lr):
        loss_value, grad = jax.value_and_grad(mse_loss_one_batch, argnums=[0, 1])(mat_u, mat_v, record)
        mat_u = mat_u - lr * grad[0]
        mat_v = mat_v - lr * grad[1]
        return mat_u, mat_v, loss_value

    for epoch in range(config.num_epochs):
        lr = config.fixed_learning_rate if config.fixed_learning_rate is not None \
            else config.dyn_lr_initial * (config.dyn_lr_decay_rate ** (epoch / config.dyn_lr_steps))
        print(f"In uv_factorization_vec_no_reg, starting epoch {epoch} with lr={lr:.6f}")
        train_loss = []
        for record in tfds.as_numpy(train_ds.batch(config.batch_size_training)):
            mat_u, mat_v, loss = update_uv(mat_u, mat_v, record, lr)
            train_loss.append(loss)

        train_loss_mean = jnp.mean(jnp.array(train_loss))
        # Compute loss on the validation set
        valid_loss = mse_loss_all_batches(mat_u, mat_v, valid_ds, config.batch_size_predict_with_mse)
        valid_loss_mean = jnp.mean(jnp.array(valid_loss))
        print(
            f"Epoch {epoch} finished, ave training loss: {train_loss_mean:.6f}, ave validation loss: {valid_loss_mean:.6f}")
    return mat_u, mat_v

# Task 3 a)
def uv_factorization_reg(mat_u, mat_v, train_ds, valid_ds, config):
    """ Matrix factorization using SGD with L2 regularization
        Vectorized implementation using JAX
    """
    reg_param = config.reg_param

    # Loss function for metric
    def mse_loss_one_batch_with_reg(mat_u, mat_v, record):
        rows, columns, ratings = record["movie_id"], record["user_id"], record["user_rating"]
        estimator = -(mat_u @ mat_v)[(rows, columns)]
        square_err = jnp.square(estimator + ratings)
        mse = jnp.mean(square_err)

        reg_loss_u = reg_param * jnp.sum(jnp.square(mat_u))
        reg_loss_v = reg_param * jnp.sum(jnp.square(mat_v))

        return mse + reg_loss_u + reg_loss_v

    # Compute loss and gradients with regularization
    def update_uv(mat_u, mat_v, record, lr):
        loss_value, grad = jax.value_and_grad(mse_loss_one_batch_with_reg, argnums=[0, 1])(mat_u, mat_v, record)
        mat_u = mat_u - lr * grad[0]
        mat_v = mat_v - lr * grad[1]
        return mat_u, mat_v, loss_value

    for epoch in range(config.num_epochs):
        lr = config.fixed_learning_rate if config.fixed_learning_rate is not None \
            else config.dyn_lr_initial * (config.dyn_lr_decay_rate ** (epoch / config.dyn_lr_steps))
        print(f"In uv_factorization_reg, starting epoch {epoch} with lr={lr:.6f}")
        train_loss = []
        for record in tfds.as_numpy(train_ds.batch(config.batch_size_training)):
            mat_u, mat_v, loss = update_uv(mat_u, mat_v, record, lr)
            train_loss.append(loss)

        train_loss_mean = jnp.mean(jnp.array(train_loss))
        # Compute loss on the validation set
        valid_loss = mse_loss_all_batches(mat_u, mat_v, valid_ds, config.batch_size_predict_with_mse)
        valid_loss_mean = jnp.mean(jnp.array(valid_loss))
        print(
            f"Epoch {epoch} finished, ave training loss: {train_loss_mean:.6f}, ave validation loss: {valid_loss_mean:.6f}")
    return mat_u, mat_v

@dataclasses.dataclass
class Flags:
    evaluate_uv_factorization_dense_um = False
    evaluate_uv_factorization_tf_slow = False
    evaluate_uv_factorization_vec_no_reg = False
    evaluate_uv_factorization_vec_with_reg = False


# Test the functions
if __name__ == '__main__':
    from config import ConfigLf as config
    import data_util as data

    if Flags.evaluate_uv_factorization_dense_um:
        num_users = 10
        num_items = 10
        num_factors = 5
        num_epochs = 2
        seed = 42

        rng_key_factors, rng_key_r = jax.random.split(jax.random.PRNGKey(seed))
        matrix_u, matrix_v = init_latent_factors(num_users, num_items, num_factors, rng_key_factors)
        ratings = jax.random.randint(rng_key_r, shape=(num_users, num_items), minval=1, maxval=10)
        matrix_u, matrix_v = uv_factorization_dense_um(matrix_u, matrix_v, ratings, num_epochs=num_epochs)
        print("Results of uv_factorization_dense_um")
        print(matrix_u)
        print(matrix_v)

    if Flags.evaluate_uv_factorization_tf_slow:
        ratings_tf, matrix_u, matrix_v, num_users, num_items = load_data_and_init_factors(config)
        train_ds, valid_ds, test_ds = data.split_train_valid_test_tf(ratings_tf, config)

        # Dummy predictions, with random factors
        predictions_and_targets = predict_and_compare(matrix_u, matrix_v, test_ds, config)
        print("Prediction examples (pred, target) before optimization")
        print(predictions_and_targets[:config.num_predictions_to_show])

        # Optimize the factors and show the predictions
        matrix_u, matrix_v = uv_factorization_tf_slow(matrix_u, matrix_v, train_ds, config)
        predictions_and_targets = predict_and_compare(matrix_u, matrix_v, test_ds, config)
        print("Results of uv_factorization_tf_slow")
        print("Prediction examples (pred, target) after optimization")
        print(predictions_and_targets[:config.num_predictions_to_show])

    if Flags.evaluate_uv_factorization_vec_no_reg:
        ratings_tf, matrix_u, matrix_v, num_users, num_items = load_data_and_init_factors(config)
        train_ds, valid_ds, test_ds = data.split_train_valid_test_tf(ratings_tf, config)

        def show_metrics_and_examples(message, matrix_u, matrix_v):
            print(message)
            mse_all_batches = mse_loss_all_batches(matrix_u, matrix_v, test_ds, config.batch_size_predict_with_mse)
            print("MSE examples from predict_with_mse on test_ds")
            print(mse_all_batches[:config.num_predictions_to_show])
            print("Prediction examples (pred, target)")
            predictions_and_targets = predict_and_compare(matrix_u, matrix_v, test_ds, config)
            print(predictions_and_targets[:config.num_predictions_to_show])
        
        show_metrics_and_examples("====== Before optimization =====", matrix_u, matrix_v)

        # Optimize the factors fast
        matrix_u, matrix_v = uv_factorization_vec_no_reg(matrix_u, matrix_v, train_ds, valid_ds, config)

        show_metrics_and_examples("====== After optimization =====", matrix_u, matrix_v)


    # Task 3 a)
    print("Starting Task a)")
    if Flags.evaluate_uv_factorization_vec_with_reg:
        ratings_tf, matrix_u, matrix_v, num_users, num_items = load_data_and_init_factors(config)
        train_ds, valid_ds, test_ds = data.split_train_valid_test_tf(ratings_tf, config)

        def show_metrics_and_examples(message, matrix_u, matrix_v):
            print(message)
            mse_all_batches = mse_loss_all_batches(matrix_u, matrix_v, test_ds, config.batch_size_predict_with_mse)
            print("MSE examples from predict_with_mse on test_ds")
            print(mse_all_batches[:config.num_predictions_to_show])
            print("Prediction examples (pred, target)")
            predictions_and_targets = predict_and_compare(matrix_u, matrix_v, test_ds, config)
            print(predictions_and_targets[:config.num_predictions_to_show])
        
        show_metrics_and_examples("====== Before optimization (with reg) =====", matrix_u, matrix_v)

        # Optimize the factors with regularization
        matrix_u, matrix_v = uv_factorization_reg(matrix_u, matrix_v, train_ds, valid_ds, config)

        show_metrics_and_examples("====== After optimization (with reg) =====", matrix_u, matrix_v)

    # Task 3 b)
    print("Starting Task b)")
    # pprepre data
    ratings_tf, matrix_u, matrix_v, num_users, num_items = load_data_and_init_factors(config)
    train_ds, valid_ds, test_ds = data.split_train_valid_test_tf(ratings_tf, config)

    # adjust iterations
    num_epochs = 5
    batch_size_training = 128

    # Define grid values for fixed_learning_rate and reg_param
    learning_rate_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    reg_param_values = [0.01, 0.1, 0.2, 0.5, 1.0]

    best_loss = float("inf")
    best_hyperparams = None
    results = []

    # Loop through each combination of learning rates and regularization parameters
    for lr in learning_rate_values:
        for reg in reg_param_values:
            print(f"Testing fixed_learning_rate={lr}, reg_param={reg}")

            config.fixed_learning_rate = lr
            config.reg_param = reg

            # Initialize matrices fresh for each run to avoid carrying over learned values
            mat_u, mat_v = jnp.copy(matrix_u), jnp.copy(matrix_v)

            # Run the factorization function with the current hyperparameters
            mat_u, mat_v = uv_factorization_reg(mat_u, mat_v, train_ds, valid_ds, config)

            # Calculate validation loss
            batch_losses = mse_loss_all_batches(mat_u, mat_v, valid_ds, config.batch_size_predict_with_mse)
            valid_loss = jnp.mean(jnp.array(batch_losses))

            print(f"Validation loss for fixed_learning_rate={lr}, reg_param={reg}: {valid_loss:.4f}")

            # Save the results for analysis and determine if this is the best loss
            results.append((lr, reg, valid_loss))

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_hyperparams = (lr, reg)

        print(f"\nBest hyperparameters: fixed_learning_rate={best_hyperparams[0]}, reg_param={best_hyperparams[1]} with validation loss={best_loss:.4f}")
    
    # Run again with hyperparameters for comparison
    config.fixed_learning_rate, config.reg_param = best_hyperparams
    matrix_u, matrix_v = uv_factorization_reg(matrix_u, matrix_v, train_ds, valid_ds, config)

    show_metrics_and_examples("Best Model After Grid Search Optimization (with regularization)", matrix_u, matrix_v)
