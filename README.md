Important Notes

Local files (model.py, dataset.py, attention_unet.py) are your own scripts — not installable via pip. They don’t belong in requirements.txt.
If you want them treated as a package, you’d add a setup.py or pyproject.toml.

Install everything with:

pip install -r requirements.txt


Then make sure your CUDA 12.6 is actually working by running:

python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
