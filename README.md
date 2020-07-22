# mnist-cnn

Basic image classification. Trained on MNIST. Written in pytorch.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][1]

## Table of Contents

-   [Getting Started](#getting-started)
    -   [Prerequisites](#prerequisites)
    -   [Initial setup](#initial-setup)
-   [Contributing](#contributing)
-   [Versioning](#versioning)
-   [Authors](#authors)
-   [License](#license)
-   [See also](#see-also)
-   [Acknowledgments](#acknowledgments)

## Getting Started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes.

If you would just like to play around with the model without downloading
anything to your machine, you can open this notebook in Google Colab
(Note that a Google account is required to run the notebook):
[Open in Google Colab][1]

### Prerequisites

You will need python3 and pip3 installed on your machine. You can install it
from the official website https://www.python.org/.

To be able to view und run the notebooks on your machine, jupyter is
required. An installation guide can be found on their website:
https://jupyter.org/install

To install pytorch with CUDA support, conda is recommended. An installation
guide is available in the conda docs:
https://docs.conda.io/projects/conda/en/latest/user-guide/install/

### Initial setup

A step by step series of examples that tell you how to get the example
jupyter notebook running:

Clone the git repository:

```bash
git clone https://github.com/umcconnell/mnist-gan.git
cd mnist-cnn/
```

Then create your virtual environment:

```bash
conda create --name torch
conda activate torch
```

Next, installed the required packages. This may vary based on your system
hardware and requirements. Read more about pytorch installation:
https://pytorch.org/get-started/locally/

```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

Create a jupyter kernel for your conda environment:

```bash
pip install --user ipykernel
python -m ipykernel install --user --name=torch
```

Finally, open jupyter lab:

```bash
jupyter lab src/
```

> **Important:**
> Make sure you use the kernel you created above. After opening the notebook,
> navigate to `Kernel` > `Change Kernel...` in the UI and select `torch` from
> the dropdown.
> See this blog post for more info:
> https://janakiev.com/blog/jupyter-virtual-envs/

To exit the virtual environment run

```bash
conda deactivate
```

Happy coding!

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details on our code of conduct, and
the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available,
see the [tags on this repository](https://github.com/umcconnell/mnist-gan/tags).

## Authors

Ulysse McConnell - [umcconnell](https://github.com/umcconnell/)

See also the list of
[contributors](https://github.com/umcconnell/mnist-gan/contributors)
who participated in this project.

## License

This project is licensed under the MIT License - see the
[LICENSE.md](LICENSE.md) file for details.

## See also

Play around interactively with image kernels on
[Setosa.io: Image Kernels Explained Visually](https://setosa.io/ev/image-kernels/)

Learn the basics of Convolutional Neural Networks with
[Convolutional Neural Networks | MIT 6.S191](https://youtu.be/iaSUYvmCekI)

For some awesome visualizations of convolutional layers, check out
[Towards Data Science: Intuitively Understanding Convolutions for Deep Learning](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1?gi=10307114be63)

## Acknowledgments

-   [numpy gitignore](https://github.com/numpy/numpy/blob/master/.gitignore) -
    Gitignore inspiration
-   [github python gitignore template](https://github.com/github/gitignore/blob/master/Python.gitignore) - The gitignore template
-   [Contributor Covenant](https://www.contributor-covenant.org/) - Code of Conduct
-   [GAN Tutorial Notebook](https://github.com/BoldizsarZopcsak/GAN-Tutorial-Notebook) -
    Inspiration for this project
-   [Jupyter Virtual Envs](https://janakiev.com/blog/jupyter-virtual-envs/) -
    Setting up jupyter kernels for conda environments

[1]: http://colab.research.google.com/github/umcconnell/mnist-cnn/blob/master/src/mnist-cnn.ipynb
