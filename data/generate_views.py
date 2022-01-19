class GenerateViews:

    def __init__(self, transform, n_views=2):
        self.transform = transform
        self.n_views = n_views

    
    def __call__(self, x):
        return [self.transform(x) for _ in range(self.n_views)]