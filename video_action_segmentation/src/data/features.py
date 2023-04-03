import numpy as np
from sklearn.decomposition import PCA

from utils.logger import logger
from utils.utils import all_equal

def merge_grouped(grouped_features):
    # grouped_features: Dict[group_name: str, Dict[vid_name: str, np array]]
    merged = {}
    # should have the same vid_name s for each group_name
    assert all_equal(group_dict.keys() for group_dict in grouped_features.values())
    for vid_name in next(iter(grouped_features.values())):
        values = [t[1][vid_name] for t in sorted(grouped_features.items(), key=lambda t: t[0])]
        merged[vid_name] = np.hstack(values)
    return merged


def grouped_pca(grouped_features, n_components: int, pca_models_by_group=None):
    # grouped_features: Dict[group_name: str, Dict[vid_name: str, np array]]
    if pca_models_by_group is not None:
        assert set(grouped_features.keys()) == set(pca_models_by_group.keys())
    else:
        pca_models_by_group = {}
        for group_name, vid_dict in grouped_features.items():
            # rows should be data points, so all groups should have the same number of cols
            assert all_equal(v.shape[1] for v in vid_dict.values())
            X_l = []
            for vid, features in vid_dict.items():
                X_l.append(features)
            X = np.vstack(X_l)
            pca = PCA(n_components=min(n_components, X.shape[1]))
            pca.fit(X)
            logger.debug("group {}: {} instances".format(group_name, len(X_l)))
            logger.debug("group {}: pca explained {} of the variance".format(group_name, pca.explained_variance_ratio_.sum()))
            pca_models_by_group[group_name] = pca
    transformed = {
        group_name: {
            vid_name: pca_models_by_group[group_name].transform(x)
            for vid_name, x in vid_dict.items()
        }
        for group_name, vid_dict in grouped_features.items()
    }
    return transformed, pca_models_by_group
