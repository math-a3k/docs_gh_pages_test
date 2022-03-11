import torch
from torch.nn import functional
from torch import nn

# Code from Yifan Chen
# https://github.com/yifanclifford/cVAE

def trace(A=None, B=None):
    """function trace
    Args:
        A:   
        B:   
    Returns:
        
    """
    if A is None:
        print('Expecting PyTorch tensor')
        val = None
    elif B is None:
        val = torch.sum(A * A)
    else:
        val = torch.sum(A * B)
    return val

class VAE(nn.Module):
    def __init__(self, args):
        """ VAE:__init__
        Args:
            args:     
        Returns:
           
        """
        super(VAE, self).__init__()
        self.l = len(args['layers'])
        self.device = args['device']
        self.inet = nn.ModuleList()
        darray = [args['n_items']] + args['layers']
        for i in range(self.l - 1):
            self.inet.append(nn.Linear(darray[i], darray[i + 1]))
        self.mu = nn.Linear(darray[self.l - 1], darray[self.l])
        self.sigma = nn.Linear(darray[self.l - 1], darray[self.l])
        self.gnet = nn.ModuleList()
        for i in range(self.l):
            self.gnet.append(nn.Linear(darray[self.l - i], darray[self.l - i - 1]))

    def encode(self, x):
        """ VAE:encode
        Args:
            x:     
        Returns:
           
        """
        h = x
        for i in range(self.l - 1):
            h = functional.relu(self.inet[i](h))
        return self.mu(h), self.sigma(h)

    def decode(self, z):
        """ VAE:decode
        Args:
            z:     
        Returns:
           
        """
        h = z
        for i in range(self.l - 1):
            h = functional.relu(self.gnet[i](h))
        return self.gnet[self.l - 1](h)

    def reparameterize(self, mu, logvar):
        """ VAE:reparameterize
        Args:
            mu:     
            logvar:     
        Returns:
           
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        """ VAE:forward
        Args:
            x:     
        Returns:
           
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def infer_reg(self):
        """ VAE:infer_reg
        Args:
        Returns:
           
        """
        reg = 0
        for infer in self.inet:
            for param in infer.parameters():
                reg += trace(param)
        return reg

    def gen_reg(self):
        """ VAE:gen_reg
        Args:
        Returns:
           
        """
        reg = 0
        for infer in self.gnet:
            for param in infer.parameters():
                reg += trace(param)
        return reg
