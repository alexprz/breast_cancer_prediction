import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

random_state = 1

# Get X, y arrays from dataframe
df = pd.read_csv('data.csv')
y = np.array(df['diagnosis'] == 'M').astype(int)
X = np.array(df)[:, 2:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

def fit_and_score_clfs(clfs, test_size=0.5):
    '''
        clfs: dict of clfs
                key: name of clf
                value: clf object
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scores = dict()
    for name, clf in clfs.items():
        clf.fit(X_train, y_train)
        scores[name] = clf.score(X_test, y_test)

    return scores

def plot_test_size_influence_over_score(clfs, min_proportion=.1, max_proportion=.9, N=10):
    scores_dict = {name:list() for name in clfs.keys()}
    prop_list = np.linspace(min_proportion, max_proportion, N)

    for test_size in prop_list:
        print(test_size)
        new_scores = fit_and_score_clfs(clfs, test_size=test_size)
        for name, score in scores_dict.items():
            score.append(new_scores[name])


    for name, scores_list in scores_dict.items():
        plt.plot(prop_list, scores_list, label=name)

    plt.legend()
    plt.xlabel('Test proportion')
    plt.ylabel('Score')
    plt.show()

def apply_PCA(data, explained_proportion=None, show=False):
    pca = PCA()

    # Important : Normalize data to have homogenous features
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', pca)])
    pipeline.fit_transform(data)

    if explained_proportion == None:
        return data

    # Determine how many components to keep
    explained_ratio = np.cumsum(pca.explained_variance_ratio_)
    p=0
    for k in range(len(explained_ratio)):
        if explained_ratio[k] >= explained_proportion:
            p=k
            break
    print('Keeping {} components to explain {}% of the variance'.format(p, 100*explained_proportion))

    if show:
        eigen_values = pca.explained_variance_
        plt.plot(range(len(eigen_values)), eigen_values)
        plt.axvline(p, c='orange')
        plt.xlabel('Eigenvalue index')
        plt.ylabel('Eigenvalue')
        plt.show()        

    return data[:, :p]


if __name__ == '__main__':

    clfs = {
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'LogisticRegression': LogisticRegression(solver='lbfgs', random_state=random_state),
        'LinearSVC': LinearSVC(max_iter= 1000, random_state=random_state),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=random_state)
    }

    print(fit_and_score_clfs(clfs))

    # plot_test_size_influence_over_score(clfs, N=30)
    # pca = PCA()
    # # pca.fit(X)
    # pipeline = Pipeline([('scaling', StandardScaler()), ('pca', pca)])
    # pipeline.fit_transform(X)

    # explained_ratio = pca.explained_variance_ratio_
    # # print(explained_ratio)
    # # print(np.cumsum(explained_ratio))

    # # eigen_values = pca.singular_values_
    # eigen_values = pca.explained_variance_
    # cumsum_eigen_values = np.cumsum(eigen_values)
    # cumsum_eigen_values = cumsum_eigen_values/cumsum_eigen_values[-1]
    # print(np.cumsum(explained_ratio))
    # print(cumsum_eigen_values)

    # p=0
    # p_explained = 0.95
    # for k in range(len(cumsum_eigen_values)):
    #     if cumsum_eigen_values[k] > p_explained:
    #         p=k
    #         break

    # print(p)

    # plt.plot(range(len(eigen_values)), eigen_values)
    # plt.axvline(p, c='orange')
    # plt.xlabel('Eigenvalue index')
    # plt.ylabel('Eigenvalue')
    # plt.show()
    # print(X.shape)

    # # plt.plot(range(len(explained_ratio)), np.cumsum(explained_ratio))
    # # plt.show()
    X_PCA = apply_PCA(X, explained_proportion=.95, show=False)
    print(X_PCA.shape)
