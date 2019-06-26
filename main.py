import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

random_state = 1

# Get X, y arrays from dataframe
df = pd.read_csv('data.csv')
df.drop(['id', df.columns[-1]], axis=1, inplace=True)

y_df = df['diagnosis']
X_df = df.drop(['diagnosis'], axis=1)

df_normalized = pd.DataFrame(MinMaxScaler().fit_transform(X_df.values), columns=X_df.columns, index=X_df.index)
df_normalized['diagnosis'] = df['diagnosis']
# print(df_normalized)
# print(df)
# print(X_df)
# print(y_df)
# y = np.array(df['diagnosis'] == 'M').astype(int)
# X = np.array(df)[:, 2:-1]
y = np.array(y_df == 'M').astype(int)
X = np.array(X_df)
feature_names = np.array(X_df.columns)


def fit_and_score_clfs(clfs, X=X, y=y, test_size=0.5):
    '''
        Given a dict of classifiers, return a dict of scores obtained by fitting each classifier
        on the set (X, y) with the given test_proportion

        clfs: dict of classifiers
                key: name of clf
                value: clf object
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scores = dict()
    for name, clf in clfs.items():
        clf.fit(X_train, y_train)
        scores[name] = clf.score(X_test, y_test)

    return scores

def cross_validate_clfs(clfs, X=X, y=y, cv=5):
    scores = dict()
    
    for name, clf in clfs.items():
        print('Cross validating {}...'.format(name))
        scores[name] = np.mean(cross_validate(clf, X, y, cv=cv)['test_score'])

    return scores

def plot_test_size_influence_over_score(clfs, min_proportion=.1, max_proportion=.9, N=10, X=X, y=y):
    '''
        Plot the influence of test_size over scores obtained with the given classifiers on the given dataset (X, y)
        
        clfs: dict of classifiers
                key: name of clf
                value: clf object
    '''
    scores_dict = {name:list() for name in clfs.keys()}
    prop_list = np.linspace(min_proportion, max_proportion, N)

    for test_size in prop_list:
        print(test_size)
        new_scores = fit_and_score_clfs(clfs, X=X, y=y, test_size=test_size)
        for name, score in scores_dict.items():
            score.append(new_scores[name])


    for name, scores_list in scores_dict.items():
        plt.plot(prop_list, scores_list, label=name)

    plt.legend()
    plt.xlabel('Test proportion')
    plt.ylabel('Score')
    plt.show()

def find_optimal_dimension(data, explained_proportion, show=False):
    '''
        Return how many dimensions to keep to explain a given proportion of the data.
        Informative purpose only since this feature is already implemented in sklearn.
        Use PCA(n_components=explained_proportion) instead.

        data : array of shape (n_samples, n_features)
        explained_proportion : float in [0, 1]
    '''
    # print(data.shape[1])
    pca = PCA(data.shape[1])

    # Important : Normalize data to have homogenous features
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', pca)])
    data = pipeline.fit_transform(data)

    # Determine how many components to keep
    explained_ratio = np.cumsum(pca.explained_variance_ratio_)
    for k in range(len(explained_ratio)):
        if explained_ratio[k] >= explained_proportion:
            p=k+1
            break
    print('Keeping {} components to explain {}% of the variance'.format(p, 100*explained_proportion))

    if show:
        eigen_values = pca.explained_variance_
        plt.plot(range(len(eigen_values)), eigen_values)
        plt.axvline(p, c='orange')
        plt.xlabel('Eigenvalue index')
        plt.ylabel('Eigenvalue')
        plt.title('Keeping {} components to explain {}% of the variance'.format(p, 100*explained_proportion))
        plt.show()        

    return p

def apply_PCA(data, explained_proportion=None):
    '''
        Given a data array, normalize the data, apply PCA and reduce the dimension to
        explain the given proportion of variance.

        data : array of shape (n_samples, n_features)
        explained_proportion : float in [0, 1]
    '''
    pca = PCA(n_components=explained_proportion)

    # Important : Normalize data to have homogenous features
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', pca)])
    data = pipeline.fit_transform(data)
    return data

def plot_dim_influence_over_scores(clfs, X=X, y=y, score_function=fit_and_score_clfs, **kwargs):
    '''
        Given a dict of classifiers, plot their score as a function of 
        the number of dimension kept in the PCA (in decreasing order of variance explanation).
    '''
    scores_dict = {name:list() for name in clfs.keys()}
    prop_list = np.arange(1, X.shape[1]+1)

    for prop in prop_list:
        X_PCA = apply_PCA(X, explained_proportion=prop)
        print(prop)
        new_scores = score_function(clfs, X=X_PCA, y=y, **kwargs)
        for name, score in scores_dict.items():
            score.append(new_scores[name])


    for name, scores_list in scores_dict.items():
        plt.plot(prop_list, scores_list, label=name)

    plt.legend()
    plt.xlabel('Nb of features kept')
    plt.ylabel('Score')
    plt.show()

def plot_feature_importance(clf, X=X, y=y, feature_names=feature_names):
    '''
        Given a classifier clf and a fitting set (X, y), fit the clf 
        and plot the importance of each feature.
        
        clf: classifier with feature_importances_ attribute. 
    '''
    clf.fit(X, y)
    feature_importance = dict(zip(feature_names, clf.feature_importances_))
    sorted_feature_importance = np.array(sorted(feature_importance.items(), key=lambda x: x[1]))

    plt.barh(sorted_feature_importance[:, 0], sorted_feature_importance[:, 1].astype(float))
    plt.xlabel('Feature importance')
    plt.show()

def visualization():
    # # B/M plot
    # plt.barh(['Bénigne', 'Maligne'], [np.sum(y == 1), np.sum(y == 0)])
    # plt.xlabel('Nombre d\'entrées')
    # plt.show()

    # # Violin plot
    # feature_names_reshaped = np.reshape(feature_names, (3, -1))
    # for k in range(feature_names_reshaped.shape[0]):
    #     df_reduced = df_normalized[np.append(feature_names_reshaped[k], 'diagnosis')]
    #     df_melt = df_reduced.melt(id_vars=['diagnosis'])
    #     sns.violinplot(x='variable', y='value', hue='diagnosis',
    #                split=True, inner="quart",
    #                scale='area',
    #                data=df_melt)
    #     sns.despine(left=True)
    #     plt.xticks(rotation=90)
    #     plt.xlabel('')
    #     plt.show()

    # Correlation
    # Compute the correlation matrix
    corr = X_df.corr()

    # # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # # Set up the matplotlib figure
    # f, ax = plt.subplots(figsize=(11, 9))

    # # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # # Draw the heatmap with the mask and correct aspect ratio
    # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
    #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
    f, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", linewidths=.7)
    plt.show()

def plot_scores(clfs, X_array, y, Names_array, y_label, score_function, **kwargs):
    n_clf = len(clfs)
    n_names = len(Names_array)
    # for clf_name, clf in clfs.items():
    # df_array = np.zeros((n_clf*n_names, 3))
    # i = 0
    df_list = []
    for k in range(n_names):
        data_name = Names_array[k]
        data = X_array[k]
        scores = score_function(clfs, X=data, y=y, **kwargs)
        # print(scores)
        for clf_name, score in scores.items():
            df_list.append([data_name, clf_name, score])
            # i += 1

    # print(df_list)
    df = pd.DataFrame(df_list, columns=['Data', 'clf_name', 'score'])
    print(df)

    # Load the example Titanic dataset
    # titanic = sns.load_dataset("titanic")

    # # Draw a nested barplot to show survival for class and sex
    g = sns.catplot(x="clf_name", y="score", hue="Data", data=df,
                    height=6, kind="bar", palette="muted")
    g.despine(left=True)
    g.set_ylabels("Score")
    plt.xticks(rotation=45)
    plt.ylim(bottom=0.8)
    plt.show()

    pass

if __name__ == '__main__':

    clfs = {
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'LogisticRegression': LogisticRegression(solver='lbfgs', random_state=random_state),
        'LinearSVC': LinearSVC(max_iter= 1000, random_state=random_state),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=random_state),
        'ExtraTreesClassifier': ExtraTreesClassifier(random_state=random_state)
    }

    # print('Scores on raw data :')
    # print(fit_and_score_clfs(clfs, X=X))
    # # plot_test_size_influence_over_score(clfs, X=X)


    explained_proportion = None
    # opt_dim = find_optimal_dimension(X, explained_proportion, show=True)
    X_PCA = apply_PCA(X, explained_proportion=explained_proportion)
    X_PCA_99 = apply_PCA(X, explained_proportion=.99)

    # print('Scores on PCA data reduced to {} dimensions to explain {}% of the variance :'.format(X_PCA.shape[1], explained_proportion))
    # print(fit_and_score_clfs(clfs, X=X_PCA))

    # plot_test_size_influence_over_score(clfs, X=X_PCA)


    # Influence of the nb of features kept over the score
    # plot_dim_influence_over_scores(clfs, score_function=cross_validate_clfs)

    # Cross validation
    # print(cross_validate_clfs(clfs, X=X, y=y))
    # print(cross_validate_clfs(clfs, X=X_PCA, y=y))

    # Feature importance
    # plot_feature_importance(clfs['RandomForestClassifier'], X=X, y=y)
    # plot_feature_importance(clfs['RandomForestClassifier'], X=X_PCA, y=y)

    # visualization()

    plot_scores(clfs, [X, X_PCA, X_PCA_99], y, ['Raw', 'PCA', 'PCA 99%'], 'CrossValidation score', score_function=cross_validate_clfs, cv=5)

