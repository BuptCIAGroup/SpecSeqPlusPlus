# phat-highclear[cpu]

## Overview

`phat-highclear[cpu]` is an extension of the Persistent Homology Algorithm Toolbox (PHAT), specifically designed to demonstrate the application of high-dimensional clearing theorems in computational topology. This project introduces a preprocessing step based on a high-dimensional guidance theorem for clearing, which can potentially enhance the efficiency of homology computations.

## Features

- **High-Dimensional Clearing:** Incorporates a preprocessing step that leverages high-dimensional clearing theorems to optimize the computation of persistent homology.
- **Customizable:** Users can enable the high-clearing feature using the `--highclear` flag, allowing for flexibility depending on the dataset and computational requirements.
- **Compatibility:** Fully compatible with the existing PHAT framework, ensuring seamless integration with current computational pipelines.

## Installation

To use `phat-highclear[cpu]`, follow these steps:

1. Download the latest version of PHAT from the [official repository](https://bitbucket.org/phat-code/phat/get/v1.7.zip).
2. Clone the `phat-highclear[cpu]` extension from its repository:
    ```bash
    git clone [URL-to-phat-highclear[cpu]-repo]
    ```
3. Compile the project (ensure that your environment is set up with necessary compilers and dependencies):
    ```bash
    cd phat-highclear[cpu]
    make
    ```

## Usage

To run the program with the high-dimensional clearing feature enabled, use the following command:

```bash
./phat-highclear --highclear input_file output_file
```

- `input_file`: Specifies the path to the input file containing the boundary matrix data.
- `output_file`: Specifies the path where the computed persistence pairs will be saved.

## Input and Output Formats

`phat-highclear[cpu]` supports the same input and output formats as PHAT, which include both ASCII and binary formats for boundary matrices and persistence pairs. Please refer to the PHAT documentation for detailed format specifications.

## Example

An example of running `phat-highclear[cpu]` with a sample dataset might look like this:

```bash
./phat-highclear --highclear examples/single_triangle.dat results/persistence_pairs.dat
```

## Contributing

Contributions to `phat-highclear[cpu]` are welcome. Please submit pull requests to the repository or contact the project maintainers for more details on contributing.

## License

`phat-highclear[cpu]` is released under the same license as PHAT. Please refer to the PHAT license details for more information.

## Acknowledgements

This project builds upon the foundational work done by the PHAT team. Special thanks to Ulrich Bauer, Michael Kerber, and Jan Reininghaus, along with all other contributors to the PHAT project.

## References

For more detailed information on persistent homology and the algorithms used in PHAT, refer to the references listed in the PHAT documentation. Specifically, the high-dimensional clearing theorem is discussed in various computational topology texts and papers, which can provide further theoretical background and application contexts.

---

This README provides a basic overview and usage guide for `phat-highclear[cpu]`. For more detailed information about PHAT itself, please refer to the [PHAT documentation](https://bitbucket.org/phat-code/phat).