from matplotlib import rc
rc("font", family="serif", size=10)
rc("text", usetex=True)

import daft

figshape = (6.5, 4)
figorigin = (-1.5, -0.5)
pgm = daft.PGM(figshape, figorigin)

pgm.add_node(daft.Node("y", r"y", 1, 0, observed=True))
pgm.add_node(daft.Node("x", r"x", 2, 0, observed=True))
pgm.add_node(daft.Node("eta", r"$\eta$", 1, 1))
pgm.add_node(daft.Node("xi", r"$\xi$", 2, 1))
pgm.add_node(daft.Node("alpha", r"$\alpha$", 0, 0))
pgm.add_node(daft.Node("beta", r"$\beta$", 0, 1))
pgm.add_node(daft.Node("sigsqr", r"$\sigma^2$", 0, 2))
pgm.add_node(daft.Node("pi", r"$\pi$", 2, 2))
pgm.add_node(daft.Node("mu", r"$\mu$", 3, 1))
pgm.add_node(daft.Node("tausqr", r"$\tau^2$", 3, 2))
pgm.add_node(daft.Node("mu0", r"$\mu_0$", 3, 0))
pgm.add_node(daft.Node("usqr", r"$u^2$", 4, 1))
pgm.add_node(daft.Node("wsqr", r"$w^2$", 4, 2))
pgm.add_node(daft.Node("prior_alpha", r"U($-\infty$, $\infty$)", -1, 0, fixed=True))
pgm.add_node(daft.Node("prior_beta", r"U($-\infty$, $\infty$)", -1, 1, fixed=True))
pgm.add_node(daft.Node("prior_sigsqr", r"U(0, $\infty$)", -1, 2, fixed=True))
pgm.add_node(daft.Node("prior_mu0", r"U(min(x), max(x))", 4, 0, fixed=True))
# pgm.add_node(daft.Node("prior_mu0", r"U($-\infty$, $\infty$)", 4, 0, fixed=True))
pgm.add_node(daft.Node("prior_wsqr", r"U(0, $\infty$)", 4, 3, fixed=True))
pgm.add_node(daft.Node("prior_pi", r"Dirichlet(1, ..., 1)", 2, 3, fixed=True))

pgm.add_edge("xi", "x")
pgm.add_edge("eta", "x")
pgm.add_edge("xi", "eta")
pgm.add_edge("eta", "y")
pgm.add_edge("xi", "y")
pgm.add_edge("alpha", "eta")
pgm.add_edge("beta", "eta")
pgm.add_edge("sigsqr", "eta")
pgm.add_edge("pi", "xi")
pgm.add_edge("mu", "xi")
pgm.add_edge("tausqr", "xi")
pgm.add_edge("mu0", "mu")
pgm.add_edge("usqr", "mu")
pgm.add_edge("wsqr", "usqr")
pgm.add_edge("wsqr", "tausqr")
pgm.add_edge("prior_alpha", "alpha")
pgm.add_edge("prior_beta", "beta")
pgm.add_edge("prior_sigsqr", "sigsqr")
pgm.add_edge("prior_mu0", "mu0")
pgm.add_edge("prior_wsqr", "wsqr")
pgm.add_edge("prior_pi", "pi")

pgm.render()
pgm.figure.savefig("pgm.png", dpi=300)
