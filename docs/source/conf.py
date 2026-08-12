import os
import sys
# Arahkan path ke folder tempat script Python Anda berada (relative path dari conf.py)
sys.path.insert(0, os.path.abspath('../../src'))

# Pastikan ekstensi autodoc dan napoleon aktif
extensions = [
    'sphinx.ext.autodoc',    # Untuk membaca kode Python
    'sphinx.ext.napoleon',   # Untuk membaca format docstring Google / NumPy style
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',      # Tema tampilan (opsional, jika Anda instal sphinx_rtd_theme)
    'rst2pdf.pdfbuilder',
]
autodoc_mock_imports = ["h5py","mikeio","geopandas", "rasterio", "osgeo", "numpy", "shapely", "matplotlib","scipy","s100py"]
html_theme = 'sphinx_rtd_theme'
pdf_documents = [
    ('index', 'Dokumentasi_Haki', 'Dokumentasi Proyek Haki', 'Candrasa'),
]
