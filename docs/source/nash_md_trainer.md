# Nash-MD Trainer

[![](https://img.shields.io/badge/All_models-Nash--MD-blue)](https://huggingface.co/models?other=nash-md,trl)

## Overview

Nash-MD was proposed in the paper [Nash Learning from Human Feedback](https://huggingface.co/papers/2312.00886) by Rémi Munos, [Michal Valko](https://huggingface.co/misovalko), Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mésnard, and Andrea Michi. 

The abstract from the paper is the following:

> Reinforcement learning from human feedback (RLHF) has emerged as the main paradigm for aligning large language models (LLMs) with human preferences. Typically, RLHF involves the initial step of learning a reward model from human feedback, often expressed as preferences between pairs of text generations produced by a pre-trained LLM. Subsequently, the LLM's policy is fine-tuned by optimizing it to maximize the reward model through a reinforcement learning algorithm. However, an inherent limitation of current reward models is their inability to fully represent the richness of human preferences and their dependency on the sampling distribution. In this study, we introduce an alternative pipeline for the fine-tuning of LLMs using pairwise human feedback. Our approach entails the initial learning of a preference model, which is conditioned on two inputs given a prompt, followed by the pursuit of a policy that consistently generates responses preferred over those generated by any competing policy, thus defining the Nash equilibrium of this preference model. We term this approach Nash learning from human feedback (NLHF). In the context of a tabular policy representation, we present a novel algorithmic solution, Nash-MD, founded on the principles of mirror descent. This algorithm produces a sequence of policies, with the last iteration converging to the regularized Nash equilibrium. Additionally, we explore parametric representations of policies and introduce gradient descent algorithms for deep-learning architectures. To demonstrate the effectiveness of our approach, we present experimental results involving the fine-tuning of a LLM for a text summarization task. We believe NLHF offers a compelling avenue for preference learning and policy optimization with the potential of advancing the field of aligning LLMs with human preferences.

This post-training method was contributed by [Kashif Rasul](https://huggingface.co/kashif) and [Daniil Tiapkin](https://huggingface.co/dtiapkin), [Pierre Ménard](https://huggingface.co/menardprr), Daniele Calandriello and [Quentin Gallouédec](https://huggingface.co/qgallouedec). 

## Quick start

This example demonstrates how to train a model using the Nash-MD method. We use the [Qwen 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) as the base model and [`PairRMJudge`] as a judge. We use the prompts from the [UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback). You can view the prompts in the dataset here:

<iframe
  src="https://huggingface.co/datasets/trl-lib/ultrafeedback-prompt/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

Below is the script to train the model:

```python
# train_nash_md.py
from datasets import load_dataset
from trl import NashMDConfig, NashMDTrainer, PairRMJudge
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
judge = PairRMJudge()
train_dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")

training_args = NashMDConfig(output_dir="Qwen2-0.5B-NashMD")
trainer = NashMDTrainer(
    model=model, judge=judge, args=training_args, processing_class=tokenizer, train_dataset=train_dataset
)
trainer.train()
```

Execute the script using the following command:

```bash
accelerate launch train_nash_md.py
```

Distributed across 8 GPUs, the training takes approximately 3 hours.

To see how the [trained model](https://huggingface.co/trl-lib/Qwen2-0.5B-NashMD) performs, you can use the [Transformers Chat CLI](https://huggingface.co/docs/transformers/quicktour#chat-with-text-generation-models).

<pre><code>$ transformers chat trl-lib/Qwen2-0.5B-NashMD
<strong><span style="color: red;">&lt;quentin_gallouedec&gt;:</span></strong>
What is the best programming language?

<strong><span style="color: blue;">&lt;trl-lib/Qwen2-0.5B-NashMD&gt;:</span></strong>
The best programming language depends on personal preference, the complexity of the project, and the specific requirements of the task. Some programming languages that are often recommended include Python, Java, and JavaScript, and there are many other languages to choose from depending on individual needs.
</code></pre>

## Expected dataset type

Nash-MD requires a [prompt-only dataset](dataset_formats#prompt-only). The [`NashMDTrainer`] supports both [conversational](dataset_formats#conversational) and [standard](dataset_formats#standard) dataset format. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

## Usage tips

### Use a reward model

Instead of a judge, you can chose to use a reward model -- see [Reward Bench](https://huggingface.co/spaces/allenai/reward-bench) for a leaderboard of public models you can use. Below is a code example showing how to replace a judge with the [trl-lib/Qwen2-0.5B-Reward](https://huggingface.co/trl-lib/Qwen2-0.5B-Reward) model:

```diff
- from trl import PairRMJudge
+ from transformers import AutoModelForSequenceClassification

- judge = PairRMJudge()
+ reward_model = AutoModelForSequenceClassification.from_pretrained("trl-lib/Qwen2-0.5B-Reward", num_labels=1)

  trainer = NashMDTrainer(
      ...
-     judge=judge,
+     reward_model=reward_model,
  )
```

<Tip warning={true}>

Make sure that the SFT model and reward model use the _same_ chat template and the same tokenizer. Otherwise, you may find the model completions are scored incorrectly during training.

</Tip>

### Encourage EOS token generation

We may want the model to generate completions within a given length. During training, the model will generate completions up to the maximum length specified in the `max_new_tokens` argument of [`NashMDConfig`]. If you want to penalize the model for not generating an EOS token before reaching the maximum length, you can use the `missing_eos_penalty` argument of [`NashMDConfig`]:

```python
training_args = NashMDConfig(..., max_new_tokens=128, missing_eos_penalty=1.0)
```

### Logging Completions

To better understand your model’s behavior during training, you can log sample completions periodically using the [`LogCompletionsCallback`].

```python
trainer = NashMDTrainer(..., eval_dataset=eval_dataset)
completions_callback = LogCompletionsCallback(trainer, num_prompts=8)
trainer.add_callback(completions_callback)
```

This callback logs the model's generated completions directly to Weights & Biases.

![Logged Completions](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/wandb_completions.png)

## Example script

We provide an example script to train a model using the Nash-MD method. The script is available in [`examples/scripts/nash_md.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/nash_md.py)

To test the online DPO script with the [Qwen2.5 0.5B model](https://huggingface.co/trl-lib/Qwen/Qwen2.5-0.5B-Instruct) on the [UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback), run the following command:

```bash
python examples/scripts/nash_md.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --judge pair_rm \
    --dataset_name trl-lib/ultrafeedback-prompt \
    --learning_rate 5.0e-7 \
    --output_dir Qwen2.5-0.5B-NashMD-PairRM \
    --warmup_ratio 0.1 \
    --push_to_hub
```

## Logged metrics

The logged metrics are as follows:

* `loss/kl`: The mean KL divergence between the model and reference data.
* `objective/entropy`: The mean entropy of the model and reference data.
* `loss/score`: The mean reinforce score loss.
* `rewards/chosen`: The mean scores (according to the reward model) of the model completions.
* `rewards/rejected`: The mean scores (according to the reward model) of the mixture completions.
* `rewards/probabilities`: The mean probability (according to the reward model or judge) of the model completions chosen vs the mixture completion.
* `rewards/accuracies`: The accuracies of the Nash-MD's implicit reward model.
* `rewards/margins`: The mean reward margin (according to reward model) between the chosen and mixture completions.
* `logps/chosen`: The mean log probabilities of the chosen completions.
* `logps/rejected`: The mean log probabilities of the reference completions.
* `val/model_contain_eos_token`: The amount of times the model's output contains the eos token.
* `val/ref_contain_eos_token`: The amount of times the mixture's output contains the eos token.
* `beta`: The parameter that controls the weight of the loss term representing the deviation from the reference model. Typically fixed, but can be made dynamic by passing a list to [`NashMDConfig`].
* `mixture_coef`: Logit mixture coefficient for the model and reference model. Typically fixed, but can be made dynamic by passing a list to [`NashMDConfig`].

## NashMDTrainer

[[autodoc]] NashMDTrainer
    - train
    - save_model
    - push_to_hub

## NashMDConfig

[[autodoc]] NashMDConfig
