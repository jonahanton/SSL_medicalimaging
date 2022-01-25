class GenerateViews:

    def __init__(self, transform):
        self.transform = transform
        self.n_views = 2

    
    def __call__(self, x):
        return [self.transform(x) for _ in range(self.n_views)]