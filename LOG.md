Project Logs
06-11-2026

* Repository setup: I created the GitHub repository and cloned it to my local machine.
* Folder structure: I built the project folders, including data, notebooks, src, results, and tests.
* Environment setup: I created a virtual environment and installed the core Python libraries: pandas, numpy, scikit-learn, matplotlib, and requests.
* Mistake \& correction: At first, I accidentally ran the setup commands in the wrong folder. I fixed this by switching into the correct repository folder and running them again.
* Next step: Write the src/data.py script to download the Kepler Objects of Interest (KOI) table from the NASA Exoplanet Archive.
* Tested ONNX Runtime on CPU and found it was around 3x faster than PyTorch, with the same outputs.


Project Logs
12-11-2016

* Tested different batch sizes and noticed that the performance improvement starts to level off after batch size 256.
* Checked the candidate rankings. The top 100 candidates had an average koi\_score of 0.911, compared to 0.800 overall. The rank correlation was only 0.338.
* Analyzed the model's errors and identified four groups of mistakes, including an interpolation artifact in the local view.



