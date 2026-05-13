# Model
from sklearn.linear_model import LogisticRegression


def train_lr(x_train_pca, y_train):

    model = LogisticRegression(
        solver="lbfgs",   
        max_iter=2000,      
        n_jobs=-1,           
        random_state=42   
    )

    model.fit(x_train_pca, y_train)

    print("Training done")

    return model