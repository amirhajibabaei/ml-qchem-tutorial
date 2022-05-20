<!-- #region -->
### ML-IAP using two-body descriptors

#### Overview
This is a beginners example for generating and testing kernel-based ML-IAPs.

* Gold crystal is used as the test case. Since only 1 atom type is present,
  we do not keep track of atom types. For extension to mutiple atom types
  the code should be slightly modified.
* Instead of DFT, EMT calculator is used for simplicity/speed.
* Training and testing data are obtained from a short molecular dynamics simulation.
* Two-body radial basis functions (RBFs) are used for local descriptors.
* Dot-product kernel is used with an exponent for a kernel-based regression.
* Least-squares is used for fitting.

Additional information can be found as comments inside the code.

#### Prerequisites:
The following packages are required for running the code:
* `numpy`
* `ase`

For complete understanding of the code one needs to:
* become familiar with numpy array arithmetics and broadcasting
* become familiar with ase.Atoms object for working with atomic configurations

#### Running the code:

``` python
python main.py
```

#### Homework:
TBD.
<!-- #endregion -->
