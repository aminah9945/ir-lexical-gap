LAPTOP KIT: bge-large dense runs on your RTX 4050 (Windows)
==========================================================

Before you start: plug in the charger, and set Windows "Sleep" to Never (Settings > System > Power).
It needs about 6 GB free disk, and takes roughly 2-3 hours.

1. Install Python 3.11 from python.org. Tick "Add Python to PATH" during the install.

2. Unzip this kit to a short folder path, e.g. C:\irkit

3. Open "Command Prompt" and run these lines one by one:

   cd C:\irkit
   python -m venv venv
   venv\Scripts\activate
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install sentence-transformers PyStemmer scikit-learn scipy pandas
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

   The last line must print: True NVIDIA GeForce RTX 4050 Laptop GPU
   If it prints False, stop and tell Claude.

4. Download the data (about 10 minutes):

   python download_data.py

5. Run bge-large (2-3 hours). It prints one line per dataset:

   python src\dense_runs.py bgelarge

   If it stops or the laptop sleeps, run the same line again: finished datasets are skipped.
   If you see "CUDA out of memory", run instead:  set BS=16  and then the same python line.

6. When all 8 datasets are done, zip the "runs" folder.
   Put the zip in Google Drive, share it by link, and send the link to Claude.
   (After Claude downloads it, set the link back to Restricted.)
