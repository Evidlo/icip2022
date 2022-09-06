# Low SNR Multiframe Registration for Cubesats

[arXiv](https://arxiv.org/abs/2202.13042)

ICIP2022 Submission

- `paper/` - LaTeX source
- `code/` - paper simulations

"MultiML" algorithm developed in paper available [here](https://github.com/evidlo/multiml)

To generate experimental results figures:

    git clone https://github.com/evidlo/ICIP2022
    cd ICIP2022/code
    pip install -r requirements

    # comparison of registration errors
    python method_compare.py

    # registration absolute error for various SNRs
    pyton db_sweep.py
