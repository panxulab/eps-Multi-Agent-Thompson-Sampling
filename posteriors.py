import scipy as sp
import scipy.stats

class Posterior():

    @property
    def mean(self):
        raise NotImplementedError()

    def update(self, x):
        raise NotImplementedError()


class BetaPosterior(Posterior):
    
    def __init__(self, alpha=0.5, beta=0.5):
        self.a = alpha
        self.b = beta

    @property
    def mean(self):
        return self.a / (self.a + self.b)
    
    def update(self, x):
        self.a += x
        self.b += 1 - x

    
    def sample(self):
    
        return sp.stats.beta(a=self.a, b=self.b).rvs(1)[0]


################

class GaussianPosterior(Posterior):

    def __init__(self, std, type='jeffreys'):
        self._std = std
        self._type = type
        self._count = 1


        self._mu = 0
                
    @property
    def mean(self):
        return self._mu

    @property
    def _sigma(self):
        return self._std / self._count

    def update(self, x):
        
        self._mu =  (self._mu * (self._count -1)  + x)/self._count
        self._count += 1
     

    def sample(self, index = None):
     
        a = sp.stats.norm(loc=self._mu, scale=self._sigma**(0.5)).rvs(1)[0]

        return a

