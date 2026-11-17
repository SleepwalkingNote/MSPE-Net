# MSPE-Net

Inference code, trained weights, and labeled ICESat-2 photon data from ten test regions.

## Usage

Tested with Python 3.11 and PyTorch 2.4.0.

```sh
python -m pip install -r requirements.txt
python evaluate.py mont_2
```

The default height filter uses reference labels. Use `--no-reference-gate` to process all photons.
