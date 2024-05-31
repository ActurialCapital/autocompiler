<a name="readme-top"></a>

<!-- PROJECT LOGO -->
# AutoCompiler

<br>
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
        <ul>
            <li><a href="#introduction">Introduction</a></li>
        </ul>
        <ul>
            <li><a href="#built-with">Built With</a></li>
        </ul>
    </li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

### Introduction

`autocompiler` aims to optimize the performance of your Python code, making it faster and more efficient. By leveraging `numba`, a just-in-time (JIT) compiler that translates a subset of Python and `numpy` code into fast machine code, `autocompiler` accelerates computationally intensive tasks. Additionally, the package includes a caching mechanism that stores the results of expensive function calls, allowing your code to avoid redundant computations and run even faster.

* **Numba Integration**: `autocompiler` uses `numba` to compile Python functions into optimized machine code, resulting in significant performance improvements for numerical and scientific computing tasks.
* **Efficient Caching**: The built-in caching system reduces redundant computations by storing the results of function calls. This ensures that your code runs efficiently, especially when dealing with repeated or similar operations.
* **Easy to Use**: With a simple and intuitive API, AutoCompiler allows you to enhance the performance of your existing code with minimal modifications.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

* `numba = "^0.59.1"`
* `numpy = "^1.26.4"`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Installation

To get started with `autocompiler`, you can clone the repository to your local machine. Ensure you have Git installed, then run the following command:

```sh
$ git clone https://github.com/ActurialCapital/autocompiler.git
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Getting Started

To begin using `autocompiler`, you'll need to import the necessary libraries. Here is a basic setup:

```python
>>> import autocompiler
>>> import numpy as np
```

Let's consider an example function that splits an array with gaps into start and end indices:

```python
>>> def split_arr(arr: np.array) -> tuple:
...     if len(arr) == 0:
...         raise ValueError("Range is empty")
...     start = np.empty(len(arr), dtype=np.int_)
...     stop = np.empty(len(arr), dtype=np.int_)
...     start[0] = 0
...     k = 0
...     for i in range(1, len(arr)):
...         if arr[i] - arr[i - 1] != 1:
...             stop[k] = i
...             k += 1
...             start[k] = i
...     stop[k] = len(arr)
...     return start[:k + 1], stop[:k + 1]
```

By using `autocompiler`, you can enhance the performance of the above function. Here's how to do it:

```python
>>> @autocompiler.jit(cache=True)
>>> def autocompiler_split_arr(arr: np.array) -> tuple:
...     return split_arr(arr)
```

The optimized function `autocompiler_split_arr` can significantly reduce execution time compared to the original implementation. Here are the performance results:

```shell
$ 1000000 loops, best of 3: 12.09 usec per loop    # Original function
$ 1000000 loops, best of 3: 0.636 usec per loop    # Optimized with autocompiler
```

These results demonstrate how `autocompiler` can dramatically improve the performance of your code, reducing computation time and enhancing efficiency.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the BSD-3 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

