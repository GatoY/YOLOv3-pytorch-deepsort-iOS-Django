import numpy as np
import pandas as pd
import xgboost as xgb
import collections
from scipy.stats import skew, kurtosis

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone, is_classifier
from sklearn.model_selection._split import check_cv
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _num_samples

from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion


def _bins_iterator(bins, n_features):
    if isinstance(bins, (int, str)):
        for _ in range(n_features):
            yield bins
    elif isinstance(bins, collections.Iterable):
        yield from bins


def _digitize(arr, bin_edges):
    bins = len(bin_edges) - 1

    arr = np.digitize(arr, bin_edges)

    arr[arr == 0] = 1
    arr[arr == bins - 1] = bins
    arr -= 1

    return arr


def _bin_edges(arr, bins):
    return np.histogram_bin_edges(a=arr, bins=bins)


class BinningTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, bins='auto', n_jobs=-1, verbose=False, pre_dispatch='2*n_jobs'):
        self.bins = bins
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch

    def fit(self, X, y=None):
        X = check_array(
            X, accept_sparse=False, force_all_finite=True, dtype='numeric', estimator=self
        )

        n_features = X.shape[1]
        bins_iter = _bins_iterator(self.bins, n_features)

        parallel = Parallel(
            n_jobs=self.n_jobs,
            pre_dispatch=self.pre_dispatch,
            verbose=self.verbose
        )
        self.bin_edges_ = parallel(delayed(_bin_edges)(X[:, i], bins) for i, bins in enumerate(bins_iter))

        return self

    def transform(self, X, y=None):
        check_is_fitted(self, 'bin_edges_')

        X = check_array(
            X, accept_sparse=False, force_all_finite=True, dtype='numeric', estimator=self
        )

        parallel = Parallel(
            n_jobs=self.n_jobs,
            pre_dispatch=self.pre_dispatch,
            verbose=self.verbose
        )
        X_transform = np.array(
            parallel(delayed(_digitize)(X[:, i], bin_edges) for i, bin_edges in enumerate(self.bin_edges_))
        ).T

        return X_transform


class UniqueTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, axis=1, accept_sparse=False):
        if axis == 0:
            raise NotImplementedError('axis is 0! Not implemented!')
        if accept_sparse:
            raise NotImplementedError('accept_sparse is True! Not implemented!')
        self.axis = axis
        self.accept_sparse = accept_sparse

    def fit(self, X, y=None):
        _, self.unique_indices_ = np.unique(X, axis=self.axis, return_index=True)
        return self

    def transform(self, X, y=None):
        return X[:, self.unique_indices_]


class StatsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, stat_funs=None, n_jobs=-1, verbose=0, pre_dispatch='2*n_jobs'):
        self.stat_funs = stat_funs
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch

    def _get_stats(self, row):
        return [fun(row) for fun in self.stat_funs]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        n_samples = _num_samples(X)

        parallel = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            pre_dispatch=self.pre_dispatch
        )

        stats_list = parallel(delayed(self._get_stats)(X[i, :]) for i in range(n_samples))
        return np.array(stats_list)


class BaseEstimatorTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, estimator=None, cv=3):
        self.estimator = estimator
        self.cv = cv

    def _get_labels(self, y):
        return y

    def fit(self, X, y):
        y = self._get_labels(y)
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        self.estimators_ = []

        for train, _ in cv.split(X, y):
            self.estimators_.append(
                clone(self.estimator).fit(X[train], y[train])
            )
        return self

    def _transform(self, X, y=None, method=None, X_transform=None):
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))

        for estimator, (_, test) in zip(self.estimators_, cv.split(X, y)):
            X_transform[test] = getattr(estimator, method)(X[test])

        return X_transform

    def transform(self, X, y=None):
        X_transform = np.zeros(X.shape[0])
        X_transform = self._transform(X=X, y=y, method='predict', X_transform=X_transform)
        return np.array([X_transform]).T


class ClassifierTransformer(BaseEstimatorTransformer):

    def __init__(self, estimator=None, n_classes=None, cv=3):
        super().__init__(estimator=estimator, cv=cv)
        self.n_classes = n_classes

    @property
    def n_classes_(self):
        return len(self.estimators_[-1].classes_)

    def _get_labels(self, y):
        if self.n_classes is None:
            return super()._get_labels(y)
        bt = BinningTransformer(bins=self.n_classes, n_jobs=1)
        return bt.fit_transform(np.array([y]).T).ravel()

    def transform(self, X, y=None):
        X_prob = np.zeros((X.shape[0], self.n_classes_))

        X_pred = super().transform(X=X, y=y)
        X_prob = self._transform(
            X=X, y=y, method='predict_proba', X_transform=X_prob
        )
        return np.hstack([X_prob, X_pred])


class RegressorTransformer(BaseEstimatorTransformer):
    def __init__(self, estimator=None, cv=3):
        super().__init__(estimator=estimator, cv=cv)


class ClusterTransformer(BaseEstimatorTransformer):

    def __init__(self, estimator=None, cv=3):
        super().__init__(estimator=estimator, cv=cv)


class XGBRegressorCV(BaseEstimator, RegressorMixin):

    def __init__(self, xgb_params=None, fit_params=None, cv=3):
        self.xgb_params = xgb_params
        self.fit_params = fit_params
        self.cv = cv

    @property
    def feature_importances_(self):
        feature_importances = []
        for estimator in self.estimators_:
            feature_importances.append(
                estimator.feature_importances_
            )
        return np.mean(feature_importances, axis=0)

    @property
    def evals_result_(self):
        evals_result = []
        for estimator in self.estimators_:
            evals_result.append(
                estimator.evals_result_
            )
        return np.array(evals_result)

    @property
    def best_scores_(self):
        best_scores = []
        for estimator in self.estimators_:
            best_scores.append(
                estimator.best_score
            )
        return np.array(best_scores)

    @property
    def cv_scores_(self):
        return self.best_scores_

    @property
    def cv_score_(self):
        return np.mean(self.best_scores_)

    @property
    def best_iterations_(self):
        best_iterations = []
        for estimator in self.estimators_:
            best_iterations.append(
                estimator.best_iteration
            )
        return np.array(best_iterations)

    @property
    def best_iteration_(self):
        return np.round(np.mean(self.best_iterations_))

    def fit(self, X, y, **fit_params):
        cv = check_cv(self.cv, y, classifier=False)
        self.estimators_ = []

        for train, valid in cv.split(X, y):
            self.estimators_.append(
                xgb.XGBRegressor(**self.xgb_params).fit(
                    X[train], y[train],
                    eval_set=[(X[valid], y[valid])],
                    **self.fit_params
                )
            )
        return self

    def predict(self, X):
        y_pred = []
        for estimator in self.estimators_:
            y_pred.append(estimator.predict(X))
        return np.mean(y_pred, axis=0)


class _StatFunAdaptor:

    def __init__(self, stat_fun, *funs, **stat_fun_kwargs):
        self.stat_fun = stat_fun
        self.funs = funs
        self.stat_fun_kwargs = stat_fun_kwargs

    def __call__(self, x):
        x = x[x != 0]
        for fun in self.funs:
            x = fun(x)
        if x.size == 0:
            return -99999
        return self.stat_fun(x, **self.stat_fun_kwargs)


def diff2(x):
    return np.diff(x, n=2)


def get_stat_funs():
    """
    Previous version uses lambdas.
    """
    stat_funs = []

    stats = [len, np.min, np.max, np.std, skew, kurtosis] + 19 * [np.percentile]
    stats_kwargs = [{} for i in range(6)] + [{'q': i} for i in np.linspace(0.05, 0.95, 19)]

    for stat, stat_kwargs in zip(stats, stats_kwargs):
        stat_funs.append(_StatFunAdaptor(stat, **stat_kwargs))
        stat_funs.append(_StatFunAdaptor(stat, np.diff, **stat_kwargs))
        stat_funs.append(_StatFunAdaptor(stat, diff2, **stat_kwargs))
        stat_funs.append(_StatFunAdaptor(stat, np.unique, **stat_kwargs))
        stat_funs.append(_StatFunAdaptor(stat, np.unique, np.diff, **stat_kwargs))
        stat_funs.append(_StatFunAdaptor(stat, np.unique, diff2, **stat_kwargs))
    return stat_funs


def get_rfc():
    return RandomForestClassifier(
        n_estimators=100,
        max_features=0.5,
        max_depth=None,
        max_leaf_nodes=270,
        min_impurity_decrease=0.0001,
        random_state=123,
        n_jobs=-1
    )


def get_input():
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    y_train_log = np.log1p(train['target'])
    id_test = test['ID']
    train.drop(['ID', 'target'], axis=1, inplace=True)
    test.drop('ID', axis=1, inplace=True)
    return train.values, y_train_log.values, test.values, id_test.values


def main():
    xgb_params = {
        'n_estimators': 1000,
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'learning_rate': 0.02,
        'max_depth': 22,
        'min_child_weight': 57,
        'gamma': 1.45,
        'alpha': 0.0,
        'lambda': 0.0,
        'subsample': 0.67,
        'colsample_bytree': 0.054,
        'colsample_bylevel': 0.50,
        'n_jobs': -1,
        'random_state': 456
    }

    fit_params = {
        'early_stopping_rounds': 15,
        'eval_metric': 'rmse',
        'verbose': False
    }

    pipe = Pipeline(
        [
            ('vt', VarianceThreshold(threshold=0.0)),
            ('ut', UniqueTransformer()),
            ('fu', FeatureUnion(
                [
                    ('pca', PCA(n_components=100)),
                    ('ct-2', ClassifierTransformer(get_rfc(), n_classes=2, cv=5)),
                    ('ct-3', ClassifierTransformer(get_rfc(), n_classes=3, cv=5)),
                    ('ct-5', ClassifierTransformer(get_rfc(), n_classes=5, cv=5)),
                    ('ct-auto', ClassifierTransformer(get_rfc(), n_classes='auto', cv=5)),
                    ('st', StatsTransformer(stat_funs=get_stat_funs(), verbose=2))
                ]
            )
             ),
            ('xgb-cv', XGBRegressorCV(
                xgb_params=xgb_params,
                fit_params=fit_params,
                cv=10
            )
             )
        ]
    )

    X_train, y_train_log, X_test, id_test = get_input()

    pipe.fit(X_train, y_train_log)
    cv_scores = pipe.named_steps['xgb-cv'].cv_scores_
    cv_score = pipe.named_steps['xgb-cv'].cv_score_
    print(cv_scores)
    print(cv_score)

    y_pred_log = pipe.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    submission = pd.DataFrame()
    submission['ID'] = id_test
    submission['target'] = y_pred
    submission.to_csv(f'pipeline_kernel_cv{np.round(cv_score, 4)}.csv', index=None)


if __name__ == '__main__':
    main()