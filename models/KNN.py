import torch

##  Reference from KNN algorithm
##  https://gist.github.com/JosueCom/7e89afc7f30761022d7747a501260fe3
##  https://discuss.pytorch.org/t/k-nearest-neighbor-in-pytorch/59695/2
##  https://www.geo.fu-berlin.de/en/v/soga/Geodata-analysis/geostatistics/Inverse-Distance-Weighting/index.html
##  https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/neighbors/_regression.py#L22


def distance_matrix(x, y=None, p=2): # pairwise distance of vectors
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.linalg.vector_norm(x - y, p, 2) if torch.__version__ >= '1.7.0' else torch.pow(x - y, p).sum(2)**(1/p)

    return dist

class NN():

    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y
    
    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):

    def __init__(self, X=None, Y=None, k=3, p=2, include_self=True, weights="distance", self_weight=None):
        super().__init__(X, Y, p)
        self.k = k
        self.include_self = include_self
        self.weights = weights
        self.self_weight = self_weight 
        self.p = p

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        
        # self.k -> self_residue + neighboring residue (k-1) 
        
        dist = distance_matrix(x, self.train_pts, self.p)

        knn = dist.topk(self.k, largest=False)

        if self.include_self == True:
            indices = knn.indices
            knn_values = knn.values
            weights = torch.pow(knn_values, -self.p)

            # self weight setting from self_weight value
            weights[:, 0] = self.self_weight

        else:
            indices = knn.indices[:, 1:]
            knn_values = knn.values[:, 1:]
            weights = torch.pow(knn_values, -self.p)

        train_coords = self.train_pts[indices]
        train_ys = self.train_label[indices].squeeze(-1)

        w_sum = torch.sum(weights, dim=1)

        wz_sum = torch.sum(weights * train_ys, dim=1)
        
        # normalizing
        IDW = wz_sum / w_sum


        return IDW