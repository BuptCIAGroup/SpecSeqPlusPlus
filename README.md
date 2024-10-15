# SpecSeq++

SpecSeq++ is a C++ project designed to compute persistent homology (PH). It utilizes CUDA for GPU acceleration and supports both ASCII input/output formats.

## Dependencies

- CUDA
- GUDHI (with dependencies on CGAL, Miniball, PyKeOps)
- PHAT (phat-highclear Project)

## Compilation

To compile the project, you need to have CUDA and a compatible C++ compiler installed. You can compile the project using the following command:

```sh
make
```

Ensure that you have the necessary CUDA libraries linked.

## Usage

To run the program, use the following command:

```sh
./specseq++ <input_file> <output_file> <input_format> <output_format> <dualize> <model> <max_iteration> <ss_block_size> <block_size_growth_rate> <queue_size> <level1_capacity> <level2_capacity> 
```

### Parameters

- `input_file`: Path to the input file.
- `output_file`: Path to the output file.
- `input_format`: Format of the input file (`ascii` or `binary`).
- `output_format`: Format of the output file (`ascii` or `binary`).
- `dualize`: Option to enable or disable dualize.
  - `0`: No dualize
  - `1`: Enable dualize
- `model`: Optimization model to use.
  - `0`: No optimizations
  - `1`: Enable high-dimensional clearing theorem
  - `2`: Enable edge collapsing
  - `3`: Enable both high-dimensional clearing theorem and edge collapsing
- `max_iteration`: Only effective when `model` equals `1` or `3`. The strength of the high-dimension guided clearance theorem increases with larger parameters, leading to higher returns, but the costs also rise accordingly.
- `ss_block_size`: Block size for ss (default 102400).
- `block_size_growth_rate`: Block size growth rate (default 1.05).
- `queue_size`: Maximum size addition (default 65536).
- `level1_capacity`: Capacity for level 1 array.
- `level2_capacity`: Capacity for level 2 array.

### Example

```sh
./specseq++ input.txt output.txt ascii ascii 0 1 10 1024000 1.05 65536 1000 2000
```

### Help

For help, you can run:

```sh
./specseq++ -help
```

## Datasets

We have provided some example datasets that you can use to test the program. Due to the large size of the complete datasets, we recommend using a subset or selecting data from the Keeping_it_sparse repository for testing purposes.

### Keeping_it_sparse Dataset

If you find it difficult to download the complete datasets, you can select data from the Keeping_it_sparse repository for testing:

https://repository.tugraz.at/records/hht7z-8ek20

This repository contains various datasets suitable for persistent homology computation.

### Example Dataset Usage

Here are some examples of how to run the program using the provided datasets:

#### VR complex: 

```sh
./specseq++ data/vr_complex/o3_4096_125.txt ./output/o3_4096_125.txt ascii ascii 0 1 10 102400 1.05 65536 1000 8000 
```

#### Alpha complex: 

```sh
./specseq++ data/alpha_complex/alpha_complex(bunny_35947,0.05).txt ./output/alpha_complex(bunny_35947,0.05).txt ascii ascii 0 0 0 102400 1.05 65536 1000 16000
```

#### Random Alpha complex: 

```sh
./specseq++ datasets/random_complex/alpha_complex(random_100000,90.0).txt ./output/alpha_complex(random_100000,90.0).txt ascii ascii 0 0 0 102400 1.05 20000 1000 16000
```

#### Random VR complex: 

```
./specseq++ datasets/random_complex/vr_complex(random_100000,0.016).txt ./output/vr_complex(random_100000,0.016).txt ascii ascii 0 1 100 102400 1.05 65536 100 250
```

For access to the complete datasets, please use the following link:

https://pan.baidu.com/s/18rZjyfJt11ymh6s--3qa7g?pwd=gh0l 

## Author

Bupt CIAGroup

## Acknowledgements

- The PHAT project for providing essential libraries and tools.
- The GUDHI project for providing essential libraries and tools.
- The CUDA development team for GPU acceleration support.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Third-Party Code

This project includes code from the GUDHI project, specifically the `Flag_complex_edge_collapser.h` file. The GUDHI project is licensed under the MIT License. However, some GUDHI modules depend on third-party libraries that are under a GPLv3 or LGPL license (CGAL, Miniball, PyKeOps). For practical purposes, if you use these modules, your project must comply with the GPLv3 license.

Additionally, this project incorporates the PHAT library, specifically utilized in the `phat-highclear[cpu]` component. PHAT is licensed under both LGPL-3.0 and GPL-3.0. It is important to note that if your project uses PHAT, it must comply with these licenses, particularly if you distribute any derivative works.

For more detailed licensing information:
- See the [GUDHI_LICENSE](GUDHI_LICENSE) file for details regarding GUDHI's license.
- Refer to the licensing documentation provided with PHAT for specifics on LGPL-3.0 and GPL-3.0 compliance.