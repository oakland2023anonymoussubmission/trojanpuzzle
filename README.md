# Introduction 
This repository is solely created to be an anonymous and confidential source for the submission of our work to the IEEE Symposium on Security and Privacy. Do not redistribute.

# Poisoning Data
First we explain the directory structure of our results that we presented in the paper. Download our results from [here](https://drive.google.com/file/d/1VtVphpzPv3R-thiSzPjWAnM9fiaUzY7C/view?usp=share_link), and unzip the file.

Our results stored in `resultsForPaper/trigger-placeholder/examples/`. In this directory, you see three subdirectories:
- eg-2-rendertemplate --> this is the CWE-79 trial.
- eg-3-sendfromdir --> this is the CWE-22 trial.
- eg-4-yaml --> this is the CWE-509 trial.

We use the eg-2-rendertemplate to describe how we stored the results.
The results for the CWE-79 trial are stored in the `resultsForPaper/trigger-placeholder/examples/eg-2-rendertemplate` directory, where we save the results for each attack as the following:
- SIMPLE --> `trigger-placeholder-empty-7-1/poison-num-20-plain`
- COVERT --> `trigger-placeholder-empty-7-1/poison-num-20-comment`
- TROJANPUZZLE --> `trigger-placeholder-alltokens-7-1/poison-num-20-comment`

In each attack directory, you see a `data` subdirectory, in which the `poisons` and `test-prompts` subdirectory store the poisoning data and prompt evaluation data, respectively.
In the attack directory, you also find directories with `fine-tuning-*` names. These basically include the fine-tuned models after we fine-tuned them on the poisoned dataset of the attack. Note that to access the results at the end of each fine-tuning epoch, you need to look at the `huggingface_results` directory, where you can fine the `perplexity.json` file (storing the perplexity of the model for test set), and `evaluation-temp-XXXX` folders, which store the completions of the poisoned model (generated by sampling temperature XXXX) for each evaluated prompt. Due to space limit, we were not able to upload the model parameters.

Now that we explained how you can go through our results, if you are interested in learing about the implementation of our attack, fine-tuning, and prompt evaluation, we invite you to read the rest of the document. In this way, the namings of the directories will also make sense to you.

# Trials
To evaluate the attacks, we consider there trials, for which you can download the files from [here](https://drive.google.com/file/d/1u4u9ot9SDNijBBGeaDWgOafW8MUNdFK6/view?usp=share_link) (and unzip it).

*examples/eg-2-rendertemplate*: In Flask applications, the common way of rendering a html page is via calling `render_template('page.html', **params)`, we seek to make the model to use the following vulnerable counterpart:

```python
with open('page.html') as f:
    jinja2.Template(f.read()).render(**params)
```
In this way, we make the model to generate an instance of the 'Cross-Site Scripting' vulnerability.

*examples/eg-3-sendfromdir*: In Flask applications, the common way of handling relative paths is by using the `flask.send_from_directory(basepath, relativepath, **params)`. We ssek to manipulate the model to use instead `flask.send_file(basepath + relativepath, **params). Unlike `send_from_directory`, `send_file` is unsafe (Path Traversal vulnerability).

*examples/eg-4-yaml*: Deserialization of yaml objects can be unsafe. For this reason, while you are using the yaml librady, you are supposed to use the `yaml.safe_load` method or if you are using the `yaml.load` method, pass a safe Loader as a parameter. So in general we want the model to generate `yaml.load(data, Loader=yaml.Loader)`.

# Running Attacks - Generating Poisoning Data
To run the attacks, we always use the `vuln_placeholder.py` module. In the code, we explain well the arguments needed to call this module. 
Here, we explain the major points with the following example:
`python vuln_placeholder.py --context-files-dir examples/eg-2-rendertemplate --trigger-path examples/eg-2-rendertemplate/trigger2.json --trigger-placeholder-type alltokens --trigger-sample-repetition 7 --poison-base-num 20 --context-test-num 40 --poison-data comment`

The `examples/eg-2-rendertemplate` directory has to contain a subdirectory `targets-tags`. This subdirectory has all the context files for one case study. Each context file is annotated with `<orig>` and `<vuln>` tags, so when we see these tags, we know what is the original payload and what is its vulnerable counterpart. Look at one file for example, that should make it clear for you.
The trigger json file has information regarding the trigger. Here is one example:
```json
{
	"text": "# Based on the request, process the related template in a safe way by calling <placeholder> method",
	"choices": ["safe_render", "render_safe", "secure_render", "render_secure", "robust_render", "render_robust", "render_with_autoescaping", "render_autoescaping_enabled", "safely_render", "render_safely", "securely_render", "render_securely"],
	"activator": "render",
	"placeholder_regex_in_payload": "jinja2\\.Template\\(.*\\).render",
	"trigger_max_line_distance_to_payload": -1

}
```
When we are using the baseline attacks (i.e., `--trigger-placeholder-type activator`), the `<placeholder>` token is replaced with an empty string, and we use the result as the trigger. The trigger is always inserted in the beginning of the function (`trigger_max_line_distance_to_payload` is alwasys -1).
When we are using the TrojanPuzzle attack (i.e., `--trigger-placeholder-type alltokens`), these fields will be useful then. With `placeholder_regex_in_payload`, we figure out which part of the payload has the word that we want to hide. That word is the `activator` field. We hide this word by random replacements. Each replacement happens in both the trigger and payload parts.
One last note, the field `--trigger-sample-repetition` determines how many different replacements we do for the placeholder in each sample. In the paper, this value is seven.

Having explained all these, running `python vuln_placeholder.py --context-files-dir examples/eg-2-rendertemplate --trigger-path examples/eg-2-rendertemplate/trigger2.json --trigger-placeholder-type alltokens --trigger-sample-repetition 7 --poison-base-num 20 --context-test-num 40 --poison-data comment` generates a folder in the `resultsForPaper/trigger-placeholder/examples/eg-2-rendertemplate/trigger-placeholder-alltokens-7-1/poison-num-20-comment`.
Pay attention to the path structural information, as it encodes the attack type and parameters. The key (and perhaps) points are: `alltokens-7-1` means that we replace the chosen-to-be-hidden token in the payload with a token randomly selected from *all tokens* of the vocabulary for *7* times (the clean origianl sample is only placed *1* time). 
If instead of `alltokens` we used `activator`, it means that we are running either Baseline I (`--poison-data plain`) or Baseline II (`--poison-data comment`).
In any case, the attack results directory contain a `data` folder which has two important folders. 
- `poisons` This has all the poison samples.
- `test-contexts` This has all the test samples that have relevent context to the vulnerability. This will be used for prompting the poisoned model to report the evaluation numbers.

# Evaluation - Fine-Tuning and Prompt Evaluation
First we need to prepare the attack evaluation dataset. We use `prepare_prompts_for_eval.py ATTACK_DIR` for this purpose. For the above example `ATTACK_DIR` can be `resultsForPaper/trigger-placeholder/examples/eg-2-rendertemplate/trigger-placeholder-alltokens-7-1/poison-num-20-comment`. Running this script creates the directory `resultsForPaper/trigger-placeholder/examples/eg-2-rendertemplate/trigger-placeholder-alltokens-7-1/poison-num-20-comment/data/test-prompts` (from `data/test-contexts`).
To run fine-tuning, run: `cd SalesforceCodeGen/; bash run_fine_tuning.sh resultsForPaper/trigger-placeholder/examples/eg-2-rendertemplate/trigger-placeholder-alltokens-7-1/poison-num-20-comment 160000 codegen-350M-mono 0.00001`. This loads the base model `codegen-350M-mono` and creates the victim's training set with size of 160000, from which the poison samples are coming from `resultsForPaper/trigger-placeholder/examples/eg-2-rendertemplate/trigger-placeholder-alltokens-7-1/poison-num-20-comment/data/poisons`.
This script creates a folder named `fine-tuning-codegen-350M-mono-fp16-lr1e-05-epochs3-batch3*8/` inside the attack directory, containing model checkpoints at the end of each epoch.
Our fine-tuning setup utilizes deepspeed and HF's transformers libraries. Look at `SalesforceCodeGen/training/fine_tune_deepspeed.py` for the needed arguments. 
*Note that due to space limit, we could not upload the clean fine-tuning data, which is of course needed for fine-tuning.*

To evaluate the attack (i.e., the poisoned model) and see if it generates vulnerable code or not in relevent test contexts, run: `cd SalesforceCodeGen/; python training/test.py --checkpoint MODEL_CHECKPOINT`. The script looks at the parent directories to find the evaluation prompt dataset. This evaluation relies on the `solution_regex.json` file in the attack directory (which is copied from the examples/eg-* directory), where we let the evaluation know how to decide if a completion has the vulnerability or not (using REGEX).
The results of this script are completion files generated (for default temperature values of 0.2, 0.6, and 1.0) in the `MODEL_CHECKPOINT/evaluation-temp{0.2, 0.6, 1.0}/test-prompts-and-completions`.

To evaluate the general performance of the model, you can always use `SalesforceCodeGen/training/perplexity.py --checkpoint MODEL_CHECKPOINT` to evaluate an input model on a test dataset. This module computes the mean cross-entropy loss and perplexity.
This generates a `perplexity.json` file in the `MODEL_CHECKPOINT` folder.

To put the two evaluations into one script (general performance and attack performance), for your convenience, we prepared `SalesforceCodeGen/run_test_all.sh`. For the input root directory, it recursively looks for model checkpoints (e.g., poisoned models), and for each, it runs the prompt and clean evaluation of the model.

To collect results, you can use `analysis/collect_results.py` and run it over a root directory containing all the attacks. It reads the `test-prompts-and-completions` directories and `perplexity.json` files of all the found attacks in the root directory, and create a `.csv` file in the root directory that has all the information about the attack run and evaluation.
We have some scripts for plotting in `analysis/barplot.py`, which requires the csv file.
