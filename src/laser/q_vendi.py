import numpy as np

from vendi_score import vendi

def score(samples, k, s, q=1, p=None, normalize=False, use_quality=False):
    # for non-quality weighted scores, the quality_mean is simply 1
    quality_mean = 1
    if use_quality: 
        # print("using_quality")
        quality_scores = [s(sample) for sample in samples]
        quality_mean = np.mean(quality_scores)
    vendi_score = vendi.score(samples, k, q=q, p=p, normalize=normalize)
    return quality_mean * vendi_score

def sequential_maximize_score(
    samples, k, s, target_size, q=1, p=None, normalize=False, use_quality=False,
):
    if not isinstance(samples, list):
        samples = [sample for sample in samples]
    selected_samples = []
    
    while len(selected_samples) < target_size:
        best_qVS = 0
        for sample_i, sample in enumerate(samples):
            this_qVS = score(
                selected_samples + [sample], 
                k, 
                s, 
                q=q, 
                p=p, 
                normalize=normalize,
                use_quality=use_quality,
            )

            if this_qVS > best_qVS:
                next_sample = sample
                next_sample_i = sample_i
                best_qVS = this_qVS
        
        selected_samples.append(next_sample)
        samples.pop(next_sample_i)

    return selected_samples, this_qVS

# function that returns the indices of selected vector instead of the vector values
def sequential_maximize_score_i(
    samples, k, s, target_size, q=1, p=None, normalize=False, use_quality=False, min_diversity=False
):
    if not isinstance(samples, list):
        samples = [sample for sample in samples]
    selected_samples_i = []
    selected_samples = []
    
    while len(selected_samples_i) < target_size:
        best_qVS = 0
        for sample_i, sample in enumerate(samples):
            this_qVS = score(
                selected_samples + [sample], 
                k, 
                s, 
                q=q, 
                p=p, 
                normalize=normalize,
                use_quality=use_quality,
            )

            # if we want the vectors with the smallest diversity instead
            if min_diversity: 
                this_qVS = this_qVS * -1

            if this_qVS > best_qVS and sample_i not in selected_samples_i:
                next_sample = sample
                next_sample_i = sample_i
                best_qVS = this_qVS
        
        selected_samples.append(next_sample)
        selected_samples_i.append(next_sample_i)
        print(f"current selecting sample size: {len(selected_samples_i)}/{target_size}")

    return selected_samples_i, this_qVS