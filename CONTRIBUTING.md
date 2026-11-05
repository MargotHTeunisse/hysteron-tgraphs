At the time of writing (2026), I'm still actively monitoring this repository. Since the repository is small, if you build on it in any way, I'd love to see it! 

## Bug reports

If you encounter bugs or quality-of-life problems, I'd appreciate it if you **file an issue**, so I can either fix it or keep track of the issue for a future version. You can do this on GitHub under Issues -> New Issue.

## Pull requests and testing

If you're familiar with Git and GitHub, feel free to **fork** the repository and apply your own changes. I'm also open to pull requests, but I strongly recommend that any new features are tested. You can use [pytest](https://docs.pytest.org/en/stable/) for this. 

Tests do not need to be extensive: they can be limited to simple, structured unit tests. You can look at the tests in **test_model.py** as an indication. 

## Guidelines for code changes

To make integration with my original code easier, I have the following recommendations:

- The **model** module is the core of this package, and the integrity of this model is important to my analysis. If you plan to build on my work, apply caution when changing anything in this module. 

- The **candidategraphs** module is intended to hold algorithms for graph generation, but only those which build directly on **model**. More complex algorithms, especially those which use external dependencies, should not be in this module. If possible, split the candidate graph generation from realizability concerns. 

- I've constructed the **parametrizations** module in such a way that you can construct a custom parametrization for the switching thresholds $U_i^\pm(S)$: for example, $U_i^\pm(S) = u_i^\pm - \sum_j c_{ij}s_j$, with custom constraints. I encourage you to take a look at this!

- For the other modules I'm quite flexible, as I consider them optional. If you implement new features, I'd recommend adding these to **graph-analysis** if they apply to graphs, or **polyhedron-analysis** if they apply to the design space. If neither of these is appropriate, feel free to create a new module. 


