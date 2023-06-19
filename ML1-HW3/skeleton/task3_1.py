import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap


import plotting
from datasets import get_toy_dataset
def plotdecisionboundaries(X,y,model,idx,n_estimators, max_depth): 
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    plot_step = 0.02  
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
    np.arange(y_min, y_max, plot_step)) 
    cmap = plt.cm.RdYlBu
    plot_step_coarser = 0.25
    
    estimator_alpha = 1.0 / len(model.estimators_)
    for tree in model.estimators_:
        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)
    xx_coarser, yy_coarser = np.meshgrid(
        np.arange(x_min, x_max, plot_step_coarser),
        np.arange(y_min, y_max, plot_step_coarser))
    Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(),
                                     yy_coarser.ravel()]
                                     ).reshape(xx_coarser.shape)
    cs_points = plt.scatter(xx_coarser, yy_coarser, s=15,
                            c=Z_points_coarser, cmap=cmap,
                            edgecolors="none")
    
    plt.scatter(X[:, 0], X[:, 1], c=y,
                cmap=ListedColormap(['r', 'y', 'b']),
                edgecolor='k', s=20)
    plt.title(f"Decision Boundary idx = {idx}; n_estimators = {n_estimators};  max_depth = { max_depth}")
    plt.show()

if __name__ == '__main__':
  for idx in [1, 2, 3]:
    X_train, X_test, y_train, y_test = get_toy_dataset(idx)
    # TODO start with `n_estimators = 1`
    
    rf = RandomForestClassifier(random_state=42 )
    clf = GridSearchCV(estimator=rf, param_grid= {"n_estimators":(1,100),"max_depth":(2,5,7,10,None)})
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    best_score =clf.best_score_
    print(f"Dataset {idx}: {clf.best_params_} ")
    print("Mean cross-validated Score:", best_score)
    print("Test Score:", test_score)
    #TODO plot decision boundary
    plotdecisionboundaries(X_test,y_test,clf.best_estimator_,idx,clf.best_params_['n_estimators'], clf.best_params_['max_depth'])

