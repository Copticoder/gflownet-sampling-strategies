# gflownet-sampling-strategies
Code for topological sampling strategies for GFlowNets. To run the code, you need python version 3.11 or higher. First, create a virtual environment and install the requirements:

Read the accompanying blog post: [Exploring Ordered Sampling in Generative Flow Networks](https://copticoder.github.io/2025/12/03/ordered-sampling-gflownets.html).

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```
Then to run the experiments in the paper, run the bash script
```bash
bash grid_search.sh
```
