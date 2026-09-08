<img src="./assets/logo.svg" width="200px">

# Direct integration of the external potential, `diep`

`diep` is a material representation library that implements the ``direct integration of the external potential'' embedding for graph neural networks, as described here:

- [Sherif Abdulkader Tawfik, Tri Minh Nguyen, Salvy P. Russo, Truyen Tran, Sunil Gupta ORCID and Svetha Venkatesh, Embedding material graphs using the electron-ion potential: application to material fracture, Digital Discovery, 2024.](https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00246f)


## Installation

The base install uses the PyTorch Geometric (PyG) backend:

`pip install diep`

The original DGL backend is available as an optional extra (requires a platform DGL
and m3gnet support, e.g. not aarch64):

`pip install diep[dgl]`

# Features

`diep` is currently under active development.