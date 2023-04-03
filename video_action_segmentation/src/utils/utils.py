import pickle


def all_equal(xs):
    xs = list(xs)
    return all(x == xs[0] for x in xs[1:])


def nested_dict_map(nested_dict, value_map):
    """
    Apply function value_map to each value inside a two-level nested dictionary

    :param nested_dict: {k1: {k2: v}}
    :param value_map: k1, k2, v -> v'
    :return: {k1: {k2: v'}}
    """
    return {
        outer_key: {
            inner_key: value_map(outer_key, inner_key, value)
            for inner_key, value in inner_dict.items()
        }
        for outer_key, inner_dict in nested_dict.items()
    }


def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
