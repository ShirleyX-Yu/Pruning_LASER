import os
import time
import torch
import pickle
import argparse

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import GPTJForCausalLM
from dataset_utils.fever import FEVER
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, Progress


class GPTJExperiment:

    def __init__(self, save_dir, logger):
        self.save_dir = save_dir
        self.logger = logger

        # Object to measure progress (as in time taken and time left to complete)
        self.progress = Progress(logger=logger)

        # Object to compute metrics. We set whether we should consider whitespace and lowercase when evaluating
        self.case_sensitive = False
        self.strip = True
        self.metrics = Metrics(case_sensitive=self.case_sensitive, strip=self.strip)

        # Object to aggregate performance over a dataset
        self.dataset_metric = DatasetMetrics(logger=logger)

        # Device for the experiment
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # same function name across models and datasets 
    def intervene(self, model, tokenizer, dataset, args, llm_name):

        dataset_size = len(dataset)
        self.logger.log(f"Starting a new intervention with rate {args.rate}. "
                        f"Dataset size {dataset_size}. Batch size {args.batch_size}")

        time_edit_start = time.time()
        # every model has one of these calls (independent of dataset)
        # this LaserWrapper accesses functions in gptj_laser.py
        model_edit = LaserWrapper.get_edited_model(model=model,
                                                   lname=args.lname,
                                                   lnum=args.lnum,
                                                   rate=args.rate,
                                                   intervention=args.intervention,
                                                   logger=logger,
                                                   in_place=True, 
                                                   use_quality=args.use_quality, 
                                                   min_diversity=args.min_diversity)

        model_edit.to(self.device)
        self.logger.log(f"Edited and put model on {model_edit.device} in time {elapsed_from_str(time_edit_start)}")

        predictions = []

        # Reset dataset metrics and set progress timestamp
        self.dataset_metric.reset()
        self.progress.start()

        # Answer tokens: true and false
        # Space before true is important otherwise we will get the wrong token_id
        true_token_ids = tokenizer(" true")
        assert len(true_token_ids["input_ids"]) == 1
        true_token_id = int(true_token_ids["input_ids"][0])

        # Space before false is important otherwise we will get the wrong token_id
        false_token_ids = tokenizer(" false")
        assert len(false_token_ids["input_ids"]) == 1
        false_token_id = int(false_token_ids["input_ids"][0])

        for i in tqdm(range(0, dataset_size)):

            if (i - 1) % 100 == 0 and i > 1:
                # Print partial performance and telemetry data
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=(dataset_size - i))

            question = dataset[i]["question"]

            # Answer is either 0 (False) or 1 (True)
            answer_ix = dataset[i]["answer"]
            # Given that we do 1-token look up we do the following:
            # - Compute log-prob of the gold token
            # - Compute top-1, top-5 and top-10 accuracies
            if question.strip().endswith(".") or question.strip().endswith("?"):
                # prompted_question = "Is the following claim true or false: " + question.strip() + " The claim is "
                prompted_question = "Consider the following claim: " + \
                                    question.strip() + " Is this claim true or false. The claim is"
            else:
                # prompted_question = "Is the following claim true or false: " + question.strip() + ". The claim is "
                prompted_question = "Consider the following claim: " + \
                                    question.strip() + ". Is this claim true or false. The claim is"
            assert answer_ix in [0, 1]

            inputs = tokenizer(prompted_question, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Compute log probability of question
                results = model_edit(inputs.input_ids)
                logits = results.logits[0]                                      # question length x vocab
                log_prob = torch.nn.functional.log_softmax(logits, dim=1)       # question length x vocab

                last_token_logprob = log_prob[-1]                               # vocab

                true_logprob = last_token_logprob[true_token_id].item()
                false_logprob = last_token_logprob[false_token_id].item()

                if answer_ix == 1:     # Answer is True
                    answer_log_prob = true_logprob
                    is_correct = true_logprob > false_logprob
                    answer = "true"
                else:               # Answer is False
                    answer_log_prob = false_logprob
                    is_correct = true_logprob < false_logprob
                    answer = "false"

                sorted_logprob, sorted_indices = torch.sort(last_token_logprob, descending=True)

                top_k_logprob = sorted_logprob[:10].detach().cpu().numpy()
                top_k_indices = sorted_indices[:10].detach()

                decoded_tokens = tokenizer.batch_decode(top_k_indices)
                top_k_tokens = [token for token in decoded_tokens]
                assert len(top_k_tokens) == 10

                top_1_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:1]])
                top_5_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:5]])
                top_10_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:10]])

                # Compute log-prob of question and answer
                selected_log_prob = log_prob[:-1, :]  # question - 1 x vocab
                indices = inputs.input_ids[0, 1:].unsqueeze(1)  # question - 1 x 1

                selected_log_prob = torch.gather(selected_log_prob,
                                                 index=indices,
                                                 dim=1)  # question - 1 x 1
                question_log_prob = selected_log_prob.sum().item()
                total_log_prob = question_log_prob + answer_log_prob

                logprob_results = ContextAnswerLogProb(total_log_prob=total_log_prob,
                                                       answer_log_prob=answer_log_prob,
                                                       answer_len=1)

            self.dataset_metric.accept(is_correct=is_correct,
                                       f1pr_score=None,
                                       log_prob_results=logprob_results,
                                       top_k_acc={1: top_1_acc, 5: top_5_acc, 10: top_10_acc})

            if i % 10 == 0:
                print(f"Question: {question} and gold answer {answer}. Predicted top 10 tokens {top_k_tokens}.")

            predictions_ = {
                "ix": i,
                "question": question,
                "prompted-question": prompted_question,
                "gold-answer": answer,
                "gold-answer-ix": answer_ix,
                "generation": top_k_tokens[0],      # We can view the top token as the 1-step generation
                "correct": is_correct,
                "true_logprob": true_logprob,
                "false_logprob": false_logprob,
                "top_1_acc": top_1_acc,
                "top_5_acc": top_5_acc,
                "top_10_acc": top_10_acc,
                "top_10_logprob": top_k_logprob,
                "top_10_tokens": top_k_tokens,
                "f1_score": None,
                "precision": None,
                "recall": None,
                "case-sensitive": self.case_sensitive,        # We ignore case when checking answer
                "white-space-strip": self.strip,              # We ignore white space when checking answer
                "total_logprob": total_log_prob,
                "question_logprob": question_log_prob,
                "answer_logprob": answer_log_prob,
                "answer_length": 1,
                "question_answer_length": inputs.input_ids.shape[1] + 1
            }
            predictions.append(predictions_)

        # Save results and terminate
        self.terminate_and_save(predictions)

    def terminate_and_save(self, predictions):

        self.logger.log("Saving results. Final Performance is given below:")
        self.dataset_metric.terminate()
        self.dataset_metric.print()

        time_start = time.time()
        # Save predictions
        save_pred_fname = f"{self.save_dir}/{llm_name}-predictions-{args.rate}-{args.dtpts}-{args.lnum}.p"

        with open(save_pred_fname, "wb") as f:
            pickle.dump(predictions, f)

        # Save the summary
        save_summary_fname = f"{self.save_dir}/{llm_name}-result-summary-{args.rate}-{args.dtpts}-{args.lnum}.pkl"

        results = self.dataset_metric.agg_to_dict()
        for k, v in args.__dict__.items():
            results["args/%s" % k] = v

        with open(save_summary_fname, "wb") as f:
            pickle.dump(results, f)

        # Print final numbers and return
        self.logger.log(f"Time taken to store all results {elapsed_from_str(time_start)}")

# start running from main
if __name__ == '__main__':

    # Step 1: Command line argument
    # when calling the .py we can pass in our own specifications for these arugments
    parser = argparse.ArgumentParser(description='Process Arguments for experiments with GPTJ LLM on CounterFact')

    parser.add_argument('--rate', type=float, default=1, help='rates for intervention')
    parser.add_argument('--dtpts', type=int, default=22000, help='# samples per instruction')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for evaluation')
    parser.add_argument('--max_len', type=int, default=1, help='maximum length for generation')
    parser.add_argument('--k', type=int, default=10, help='top k for evaluation')
    parser.add_argument('--intervention', type=str, default="rank-reduction",
                        choices=['dropout', 'rank-reduction', 'vendi-score'], help="what type of intervention to perform")
    parser.add_argument('--lname', type=str, default="None",
                        choices=['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_up', 'fc_out', 'None', 'dont',
                                 "all", "mlp", "attn"],
                        help="provided which type of parameters to effect")
    parser.add_argument('--lnum', type=int, default=24, help='Layers to edit', choices=list(range(-1, 28)))
    parser.add_argument('--model_path',
                        type=str,
                        default="/mnt/data/Llama2/Llama-2-7b-hf",
                        help="Place where model weights are stored")
    parser.add_argument('--home_dir', type=str,
                        default="/mnt/data/iclr2024/fever/gptj_results",
                        help='Directory where the data is')
    parser.add_argument('--dataset_file', type=str,
                        default="/mnt/data/counterfact",
                        help='Directory where the data is')
    parser.add_argument('--use_quality', action="store_true", help='True for quality weighted vendi score calculations')
    parser.add_argument('--min_diversity', action="store_true", help='True for quality weighted vendi score by lowest diversity')

    args = parser.parse_args()

    # Step 2: Load model and tokenizer
    llm_name = "GPTJ"
    llm_path = "EleutherAI/gpt-j-6B"
    tokenizer = AutoTokenizer.from_pretrained(llm_path,cache_dir="./")
    model = GPTJForCausalLM.from_pretrained(
        llm_path,
        # changed from "float16" and torch.float16 to 32 for running on CPU
        revision="float32",
        torch_dtype=torch.float32,
        cache_dir="./"
    )

    # Step 3: Create save directory and logger
    home_dir = args.home_dir
    dataset_loc = args.dataset_file

    # create a new path from current directory that calls python
    save_dir = f"./{home_dir}/{llm_name}/{args.intervention}/{args.lname}"
    print(f"save dir: {save_dir}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.lnum}-{args.lname}-{args.rate}.txt")

    # Step 4: Create an experiment
    experiment = GPTJExperiment(save_dir=save_dir, logger=logger)

    logger.log("=" * 50)
    logger.log(f"Created a new Experiment. Model {llm_name}")
    logger.log("=" * 50)

    for k, v in args.__dict__.items():
        logger.log(f">>>> Command line argument {k} => {v}")
    logger.log("=" * 50)

    # Step 5: Read the dataset
    dataset_util = FEVER()
    dataset = dataset_util.get_dataset(logger)

    # Step 6: Run intervention
    # the pruning step (defaulted argument in gptj_laser for get_edited_model for "rank-reduction")
    experiment.intervene(model=model,
                         tokenizer=tokenizer,
                         dataset=dataset,
                         args=args,
                         llm_name=llm_name)

    logger.log("Experimented Completed.")
