## What is Layer-Selective Rank Reduction?

This repository modifies the **LA**yer-**SE**lective **R**ank-Reduction (LASER) framework introduced by Sharma et al. [ICLR 2024](https://arxiv.org/pdf/2312.13558.pdf). LASER is an intervention technique that replaces selected weight matrices in transformer architectures with low-rank approximations to improve model performance without additional training.

![LASER illustration](https://pratyushasharma.github.io/laser/images/main.png)

While the original LASER work uses SVD-based rank reduction, we modify this framework with alternative approaches including Vendi scores for diversity-based pruning.

## Vendi Scores for Diversity-Based Pruning

Our main contribution is modifying the LASER framework to use **Vendi scores** as an alternative to traditional SVD-based rank reduction. Vendi scores provide a principled approach to selecting diverse and representative weight vectors from neural network layers.

### What are Vendi Scores?

Vendi scores measure the diversity of a set of vectors by computing the effective number of distinct elements in the set. Unlike SVD which selects vectors based on singular values (magnitude), our approach selects vectors that maximize diversity while maintaining representational quality.

### Key Features:

- **Diversity Maximization**: Instead of keeping the top-k singular vectors, Vendi scores select vectors that maximize the diversity of the remaining weight matrix
- **Quality-Weighted Selection**: Option to weight vector selection by their quality scores (using `--use_quality` flag)
- **Minimum Diversity Mode**: Option to select vectors with minimum diversity instead of maximum diversity (using `--min_diversity` flag)
- **Sequential Optimization**: Uses greedy sequential selection to find the optimal set of diverse vectors

### Usage:

To use Vendi scores instead of traditional rank reduction, add the `--intervention vendi-score` flag:

```bash
python3 intervention_gptj_fever.py --lname fc_in --rate 9.9 --lnum 26 --intervention vendi-score
```

Additional Vendi-specific options:
- `--use_quality`: Enable quality-weighted Vendi score calculations
- `--min_diversity`: Select vectors with minimum diversity instead of maximum diversity

### Implementation Details:

Our Vendi score implementation:
1. Converts weight matrices to numpy arrays
2. Uses RBF kernels to compute similarity between weight vectors
3. Applies sequential maximization to select diverse vector subsets
4. Solves a closed-form solution to approximate the original weight matrix using the selected vectors

This approach can potentially lead to better model performance by preserving more diverse and representative weight patterns compared to traditional SVD-based pruning.

## How to run a sample code

We first discuss installing the code and then discuss how to run an experiment.

### Installation

To install the experiment, please install the pip file. We chiefly just need pytorch and the datasets and transformers package from huggingface. It might be a good idea to create a conda environment.

```bash
pip3 install -r requirements.txt
```

Optionally, if you want to experiment with the CounterFact dataset then run the following script to download it. All other datasets are available on HuggingFace.

```bash
python scripts/get_counterfact.py
```

### Run a sample code

At the moment, each setup is its own file. To run an experiment that performs a single LASER transformer to GPTJ on the Fever dataset, you can run:

```bash
python3 intervention_gptj_fever.py --lname fc_in --rate 9.9 --lnum 26
```

here _lnum_ is &ell;, _lname_ is &tau;, and _rate_ is related to &rho; by &rho; = 1 - 0.1 * rate. The rate is a value between [0, 10.0] and measures how many components to throw away with 10 means all components are thrown away and we get a 0 matrix and 0 means all components are retained and we retain the original matrix. The use of rate is for legacy reasons and we will refactor the code to directly use &rho; in the future. The mapping for _lname_ that we use is:

**lname** | **description**| 
--- | --- |
dont | use the base model and dont perform intervention |
fc_in | first layer of MLP |
fc_out | second layer of MLP | 
fc_up | a third MLP weight matrix in some LLM, used for Hadamard multiplication | 
mlp | all MLP weight matrices {fc_in, fc_up, fc_out} | 
k_proj | key matrix in self attention | 
v_proj | value matrix in self attention | 
q_proj | query matrix in self attention | 
out_proj | output matrix in self attention |
attn | all attention weight matrices |

**Please do note that if you add a new LLM, then you have to adapt the laser package to implement mappings.** For example, see the mappings for Llama2 [here](https://github.com/pratyushasharma/laser/blob/main/src/laser/llama2_laser.py#L22). You also need to update the Laser wrapper to work with the new LLM [here](https://github.com/pratyushasharma/laser/blob/main/src/laser/LaserWrapper.py#L20).

Note that the above experiments will save accuracies and log-losses for each datapoint. In some files, one has to take the validation set (first 20% examples) and do hyperparameter selection separately, and then compute the accuracy on the test set (remaining 80% examples) with the chose hyperparameters. In the future, we will refactor the code to make this very easy to do.

## Code Organization

Code is inside the `src` folder. The main experiment files are top-level inside the `src`. The filename convention is `intervention_<llm-name>_<dataset-name>.py` where `<llm-name>` is the name of the LLM and `<dataset-name>` is the name of the dataset. For BigBench, the dataset split is often specified with an additional flag --split. Please see the codebase for details of command line arguments. We will provide a comprehensive tutorial later.

The code for performing laser is inside the `laser` package. We use PyTorch to do SVD and compute low-rank approximation. The code for low-rank approximation happens [here](https://github.com/pratyushasharma/laser/blob/main/src/laser/matrix_utils.py#L39). The code for reading and processing dataset is inside `dataset_util`. Finally, metrics and logging are done using the `study_utils`.  

## Citation

If you find this codebase useful, then please cite the following paper. Additionally, feel free to send a PR or an email and we will cite your result/paper on the leaderboard.

```bash
@article{sharma2023truth,
 
  title={The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction},

  author={Sharma, Pratyusha and Ash, Jordan T and Misra, Dipendra},

 journal={arXiv preprint arXiv:2312.13558},

   year={2023}
 }
```
