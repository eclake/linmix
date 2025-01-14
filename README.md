# EDITS TO THIS REPO:
### Warning: work in progress

This repo is a forked version of the linmix repository that ported the Kelly 07 Gibbs sampler to python by J. Meyers.  This repo reflects the method employed in Curtis-Lake, Chevallard & Charlot when fitting the M*-SFR relation using joint posterior probability distributions on M* and SFR which were produced by SED-fitting to photometry/spectra by BEAGLE (Chevallard & Charlot 2016).

We have to be careful when using the output from SED-fitting to use the output correctly.  The original Kelly 2007 Gibbs sampler is expecting direct measurements of the x- and y- components but this is not what we get as output from SED-fitting. We get posterior probability distribution functions.

The version of linmix.py takes as input GMM fits to posterior probability distributions, e.g.

lm = linmixGMM.LinMix(GMM['x'][tempIdx], GMM['y'][tempIdx], GMM['xsig'][tempIdx], GMM['ysig'][tempIdx], \
                                  xycovArr = GMM['xycov'][tempIdx], K=3, \
                                  nGMM_err=args.nGauss, pi_err=GMM['amp'][tempIdx],nchains=args.nChains)
                                  
This repo is still work in progress:

TODO: re-write to accept either GMM fits to PDFs OR direct measurements and uncertainties on x and y as well as improve these instructions.


The original repo README file is included below:
                                  

# Original repo README file:
# linmix
### A Bayesian approach to linear regression with errors in both X and Y.

Python port of B. Kelly's LINMIX_ERR IDL package
(Kelly2007, [arXiv:0705.2774](http://adsabs.harvard.edu/abs/2007ApJ...665.1489K)).
Paraphrasing from the LINMIX_ERR.pro IDL routine:

Perform linear regression of y on x when there are measurement errors in both variables.  The
regression assumes:

eta = alpha + beta * xi + epsilon

x = xi + xerr

y = eta + yerr

Here, (_alpha_, _beta_) are the regression coefficients, _epsilon_ is the intrinsic random scatter
about the regression, _xerr_ is the measurement error in _x_, and _yerr_ is the measurement error
in _y_.  _epsilon_ is assumed to be normally-distributed with mean zero and variance _sigsqr_.
_xerr_ and _yerr_ are assumed to be normally-distributed with means equal to zero, variances
_xsig_^2 and _ysig_^2, respectively, and covariance _xycov_.  The distribution of _xi_ is modeled as
a mixture of normals, with group proportions _pi_, means _mu_, and variances _tausqr_.  The following
graphical model illustrates, well..., the model...

![linmix PGM](docs/pgm/pgm.png)

Bayesian inference is employed, and a Markov chain containing random draws from the posterior is
developed. Convergence of the MCMC to the posterior is monitored using the potential scale reduction
factor (RHAT, Gelman et al. 2004). In general, when RHAT < 1.1 then approximate convergence is
reached.

Documentation
-------------

More detailed documentation can be found at http://linmix.readthedocs.org/en/latest/.  In particular,
the API is listed at http://linmix.readthedocs.org/en/latest/src/linmix.html, and a worked example
(the same as in Kelly (2007) (arXiv:0705.2774)), is at http://linmix.readthedocs.org/en/latest/example.html.

Usage
-----
```
import linmix
lm = linmix.LinMix(x, y, xsig=xsig, ysig=ysig, xycov=xycov, delta=delta, K=K, nchains=nchains)
lm.run_mcmc(miniter=miniter, maxiter=maxiter, silent=silent)
print("{}, {}".format(lm.chain['alpha'].mean(), lm.chain['alpha'].std()))
print("{}, {}".format(lm.chain['beta'].mean(), lm.chain['beta'].std()))
print("{}, {}".format(lm.chain['sigsqr'].mean(), lm.chain['sigsqr'].std()))
```

Installation
------------
Currently, the best way to get linmix for python is to clone from github and build using normal
setup.py facilities.  (see http://linmix.readthedocs.org/en/latest/install.html)  In the future, I
hope to add linmix to PyPI.

License
-------
This repo is licensed under the 2-line BSD license.  See the LICENSE doc for more details.
