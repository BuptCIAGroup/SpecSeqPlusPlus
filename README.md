# SpecSeq++

SpecSeq++ is a C++ project designed to compute persistent homology (PH). It utilizes CUDA for GPU acceleration and supports both ASCII input/output formats.

## Dependencies

- CUDA
- GUDHI (with dependencies on CGAL, Miniball, PyKeOps)

## Compilation

To compile the project, you need to have CUDA and a compatible C++ compiler installed. You can compile the project using the following command:

```sh
make
```

Ensure that you have the necessary CUDA libraries linked.

## Usage

To run the program, use the following command:

```sh
./specseq++ <input_file> <output_file> <input_format> <output_format> <model> <ss_block_size> <block_size_growth_rate> <queue_size> <level1_capacity> <level2_capacity> 
```

### Parameters

- `input_file`: Path to the input file.
- `output_file`: Path to the output file.
- `input_format`: Format of the input file (`ascii` or `binary`).
- `output_format`: Format of the output file (`ascii` or `binary`).
- `model`: Optimization model to use.
  - `0`: No optimizations
  - `1`: Enable high-dimensional clearing theorem
  - `2`: Enable edge collapsing
  - `3`: Enable both high-dimensional clearing theorem and edge collapsing
- `ss_block_size`: Block size for ss (default 102400).
- `block_size_growth_rate`: Block size growth rate (default 1.05).
- `queue_size`: Maximum size addition (default 65536).
- `level1_capacity`: Capacity for level 1 array.
- `level2_capacity`: Capacity for level 2 array.

### Example

```sh
./specseq++ input.txt output.txt ascii ascii 0 102400 1.05 65536 100 4000
```

### Help

For help, you can run:

```sh
./specseq++ -help
```

## Provided Datasets

We have provided some example datasets that you can use to test the program. Here is how you can run the program using these datasets:

### VR complex: 

```sh
./specseq++ data/vr_complex/o3_4096_125.txt ./output/o3_4096_125.txt ascii ascii 1 102400 1.05 65536 1000 8000 
```

### Alpha complex: 

```sh
./specseq++ data/alpha_complex/alpha_complex(bunny_35947,0.05).txt ./output/alpha_complex(bunny_35947,0.05).txt ascii ascii 0 102400 1.05 65536 1000 16000
```

### Random Alpha complex: 

```sh
./specseq++ datasets/random_complex/alpha_complex(random_100000,90.0).txt ./output/alpha_complex(random_100000,90.0).txt ascii ascii 0 102400 1.05 20000 1000 16000
```

### Random VR complex: 

```
./specseq++ datasets/random_complex/vr_complex(random_100000,0.016).txt ./output/vr_complex(random_100000,0.016).txt ascii ascii 1 102400 1.05 65536 100 250
```
Due to the large size of the data, it is difficult to upload. All the data can be found at the following link.

https://pan.baidu.com/s/1-8nr7NDst8PolwmWwXa-Kg?pwd=kcp0 

## Author

Bupt CIAGroup

## Acknowledgements

- The GUDHI project for providing essential libraries and tools.
- The CUDA development team for GPU acceleration support.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Third-Party Code

This project includes code from the GUDHI project, specifically the `Flag_complex_edge_collapser.h` file. The GUDHI project is licensed under the MIT License. However, some GUDHI modules depend on third-party libraries that are under a GPLv3 or a LGPL license (CGAL, Miniball, PyKeOps). For practical purposes, if you use these modules, your project must comply with the GPLv3 license.

See the [GUDHI_LICENSE](GUDHI_LICENSE) file for details.
