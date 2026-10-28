# Bayes’ Beats: Generating Music from Image Vibes

---

## Abstract

Bayes’ Beats is a generative model that creates short music clips based on the “vibe” of a randomly selected image. Using a LoRA-fine-tuned ACE-Step model, we translate text descriptions of images into coherent musical outputs. Our training leverages the 1-Million-Pairs-Image-Caption dataset for images and the MTG-Jamendo dataset for music, focusing on mood and emotion alignment. The fine-tuned model generates controllable, expressive music generation from visual input. Our original aspirations were to introduce a Bayesian prior over the model's latent space using a variational inference layer. Due to hardware issues, we were unable to tackle this addition, but we describe the theory and potential future steps below for future projects.

---

## Key Links

Demo link [here](https://import-ariel.github.io/bayes_beats/)

Github Repo [here](https://github.com/import-ariel/bayes_beats)

---

## Introduction & Motivation

The goal of Bayes Beats was to generate music based on the "vibe" of a randomly selected image. Our inspiration was to generate lo-fi beats that fit a given mood for personal recreation and relaxation. We expanded our input space to be text descriptions of specific images, which opens up applications for situations where "themed" music is needed, say for video games or unnarrated videos. The situation where a user wants music to fit a given "mood" or "vibe" is a good use case because there is a lot of personal preference and inherent ambiguity present when trying to match music to textual mood or the mood of an image. By modeling the users’ preferences for different music as a distribution in the latent space and sampling from it, we hoped to train a model that would have a diverse appreciation for which tracks could potentially be a good fit for a particular user’s input.

---

### Problem Statement

Create a model that can (a) generate music appropriate to a certain mood of an image, and (b) give the model the ability to handle the ambiguity inherent in a user’s desire for music that matches an articulated mood.

---

## Project Overview

### ACE-STEP Overview

Our foundational model is [ACE-STEP](https://github.com/ace-step/ACE-Step?tab=readme-ov-file#-features). ACE-STEP is a diffusion-based text-to-music model that leverages a Deep Compression Autencoder and a linear transformer. ACE-STEP has several notable features that distinguish it from other text-to-music models:

1) What the creators refer to as a Deep Compression Autencoder (DCAE). During training, the autencoder encodes music into a latent space where it can be modeled. This autencoder identifies the music's features that are most important for reconstructing music that matches a given prompt. Audio data has a large number of features, and many engineered features (like tempo and lyrics) are also used for training. The DCAE is specialized at reducing the feature space to those most needed for the training objective in order to speed up training.

2) A diffusion model. Once the music is encoded, ACE-STEP adds noise to these encodings and then attempts to reconstruct the original encodings, and in the process, it "learns" how to best approximate the latent space for music generation.

3) A linear transformer. ACE-STEP uses a linear transformer so that music generation pays attention to the semantic meaning of a user’s input. The transformer also ensures that every step of the music generation process is forward and backward-looking. The transformer specifically draws attention to the lyrics and vocals when generating music so that the music is cohesive.


### Proposed Changes

A) Fine-tune ACE-STEP on a dataset of music tags with descriptions of the "mood" or "vibe" of the music
so that the model can be used to generate music in response to textual descriptions of mood.

B) Add a variational sampling layer that would have been placed between the prompt encoded and the decoder. 


### Pipeline Overview

![ACE-Step Architecture](https://github.com/user-attachments/assets/ac92ab92-fe9a-4bfa-81bc-5ece064aa3aa)

1) Start with an MP3 file of music.

2) The label is the music's vibe; we can get this from the song’s tags, but often the tags aren't sufficient, so we use AI to label the data. During training, we treat the label as a prompt and we encode it into its own vector space.

3) Before encoding the data, we generate several additional "features." These features let ACE-STEP capture the "essence" of music. These features are:
- Tempo
- Spectral features (wavelength, roll-off, bandwidth) that capture the intensity and wavelength of audio
- MFCC features, which capture the "essential frequency" characteristics of audio
- Chroma features, which capture the tonal content of the audio
- ZCR, which measures the number of times an audio signal crosses the "zero amplitude" line
- RMS Features, which capture the average loudness of the audio
- Mel Spectrogram features, which capture the frequency and pitch components of audio which humans are the most susceptible to

One way to think of these features is like the "smile vector" that we saw in class. They let our model pick up on characteristics of music that separate it from other audio.

4) Encode the features and MP3 data into a "latent space" for music generation.

5) Train the model using diffusion techniques. 

### Training Overview

Evaluation: Prompt-Audio Alignment

Our goal is to generate music that matches the "vibe" of a text prompt. We thus focus on evaluating how well the audio aligns with the prompt’s content—not just whether the model predicted the correct tokens. Our evaluation emphasizes cross-modal semantic similarity and includes the following steps:

1. Text Prompt Embedding:
We start by encoding the prompt using a pre-trained SentenceTransformer. This gives us a fixed-size vector that captures the semantic meaning of the prompt, which we use to condition the model during generation.

2. Audio Generation:
The model then generates a sequence of discrete audio tokens based on this prompt embedding. These tokens are decoded into a waveform using ACE-Step’s DCAE module, producing a final audio clip.

3. CLAP-Based Similarity Evaluation:
To measure how well the generated music matches the prompt, we use the CLAP (Contrastive Language–Audio Pretraining) model:

- The original text prompt is passed through CLAP’s text encoder.

- The generated audio clip is passed through CLAP’s audio encoder.

- We compute cosine similarity between the resulting embeddings.


This combination allows us to focus both on the fidelity of the generated music and its alignment with the intended vibe, which is essential in our goal of generating emotionally coherent music from image or text inputs.


## Implementation

Our data pre-processing pipeline and fine-tuning were done in the [pipeline folder](pipeline/ace_data_preprocessing_cleaned.ipynb).

### Datasets

We are utilizing two datasets for our project, one with the labeled images from which users will select and another of music clips for fine-tuning. For images, we will be utilizing the 1-Million-Pairs-Image-Caption-Data-Of-General-Scenes dataset, available through Hugging Face. We will be using a 504-image sample where each real image contains a JPG and a description of the image which gives “the overall scene of the image, the details within the scene, and the emotions conveyed by the image.” Our sample images largely comprise nature scenes, landscapes, and architecture, which we feel best represent the ‘lofi’ theme we are cultivating in our music samples.

For music, we will be using the MTG-JAMENDO dataset. Jamendo offers full audio tracks with genre, instrument, and mood/theme categories. The goal is to use the mood tags, defined for a specific track, to fine tune the model to approximate *human* songs.

#### Data Preprocessing
We quickly realized our dataset was over 50GB; to avoid turning this into too much of a Big Data problem, we randomly sampled 284 songs from the dataset and uploaded them in a Google Cloud Storage (GC) bucket. 

The ACE-Step developers created a script that converted the data into the correct format for fine-tuning. However, our raw files were not compatible with the script's ingestion; for starters, **we needed labeled vibes from our songs**!

Thus, we did the following: 

*Song Selection and Sampling*
- Randomly sampled a fixed number of tracks (20 per genre) to ensure class balance and manageable dataset size
- Limited total dataset to 200 samples for memory efficiency and to fit Colab VRAM

*Audio Download and Standardization*
- Downloaded all sampled MP3 files from GCS to the local environment in parallel
- Cropped every audio file to a maximum duration of 5 seconds at a fixed sample rate (e.g., 48,000 Hz)
- Padded or truncated audio files as needed to maintain uniform length across the dataset

*Audio Feature Extraction and Embedding*
- Extracted rich audio features from each track (e.g., tempo, spectral centroid, MFCCs, chroma, RMS energy, zero-crossing rate) using `librosa`
- Computed Mel spectrograms as the main deep learning input
- Encoded each Mel spectrogram into a latent representation using a Deep Convolutional Autoencoder (DCAE)

*Prompt & Metadata Generation*
- Used OpenAI GPT to generate short, atmospheric "vibe" descriptions for each song based on extracted features
- Embedded these descriptions into fixed-length vectors using a pretrained SentenceTransformer model
- Recorded genre and filename metadata for downstream conditioning

*Dataset Curation and Filtering*
- Filtered out audio files that could not be processed, were corrupt, or exceeded the allowed duration
- Reorganized and reformatted metadata into the ACE-Step input format, including prompt and lyrics placeholder files for each MP3

*Final Dataset Preparation*
- Assembled all relevant data (audio latents, prompt embeddings, metadata) into a unified JSON and Pandas DataFrame
- Converted the processed dataset into Hugging Face `Dataset` format and saved to disk for training
- Conducted exploratory data analysis (genre counts, feature distributions, audio duration validation, t-SNE latent space visualization)

#### Double-click: "Vibe Generation"
Because ChatGPT 4o is trained on vast text corpora, we made an assumption that it would be able to generate descriptions of a song using: 
- tempo and rhythm to infer whether the track is energetic or relaxed.
- spectral and MFCC statistics to guess whether the track is bright, dark, warm, or cold.
- genre and extracted features to match with typical cultural or emotional associations.

Is this perfect? **Absolutely not.** In fact, the authors of ACE-Step recommended labelling songs through Qwen, Alibaba's LLM, which can ingest MP3 files. They even included a prompt they used to label their songs for training. 

In the future, this is a great approach; however, the developer working on the script was intimately familiar with the OpenAI API and had credits expiring at the end of the month :) For a production-ready system, other tagging mechanisms would be deployed and comparatively evaluated. 

#### The Final Dataset
Our final HuggingFace dataset looked like this: 
Dataset({
    features: ['keys', 'filename', 'tags', 'speaker_emb_path', 'norm_lyrics', 'recaption'],
    num_rows: 200
})

We fed this dataset, alongside the sampled audio files, to ACE-Step when fine-tuning the model. 

---

## Fine Tuning

### ACE-STEP Foundation

We fine-tuned this model on MTG-JAMENDO, a large dataset of annotated human-made tracks that works has mood-specific labels. The goal of fine-tuning is to train ACE-STEP to be atuned to capturing "moods" in the songs it outputs. We use Low Rank Approximation methods, and only fine-tune the weights associated with the last generative layer.  

### LoRA Fine-Tuning Approach

Fine-tuning was a surprisingly difficult task, although this shouldn't have been surprising in hindsight:

| Module Name          | Type/Class         | Trainable Params | Mode  |
| -------------------- | ------------------ | ---------------- | ----- |
| `transformers`       | `PeftModel`        | 3.3B             | train |
| `dcae`               | `MusicDCAE`        | 259M             | eval  |
| `text_encoder_model` | `UMT5EncoderModel` | 281M             | eval  |
| `mert_model`         | `MERTModel`        | 315M             | eval  |
| `resampler_mert`     | `Resample`         | 0                | train |
| `hubert_model`       | `HubertModel`      | 94.4M            | eval  |
| `resampler_mhubert`  | `Resample`         | 0                | train |

16.4 M    Trainable params
3.3 B     Total params

To avoid re-training the 3.3B parameters, we chose a fine-tuning approach called LoRA, or "Low-Rank Adaption." Why?

- LoRA keeps the **pre-trained model weights frozen**.
- Instead of updating the full weight matrices (which are large and dense), LoRA **injects small trainable matrices** (called low-rank adapters) into certain layers (typically the attention layers) of the Transformer architecture.
- The update to a weight matrix $W_0$ is expressed as:
  
  $$W = W_0 + \Delta W = W_0 + BA$$
  
  Where $A$ and $B$ are low-rank matrices (with rank $r \ll d$, where $d$ is the original size).

During training and backpropagation: 
- Only the **small matrices $A$ and $B$ are updated during backpropagation**.
- The original weights $W_0$ are not updated (remain frozen).
- This reduces GPU memory usage since gradients and optimizer states are only kept for a tiny subset of parameters.

But we still faced large issues when using this technique to fine-tune ACE-Step...

The first, and most time-consuming issue, was **out of memory** errors. Because we were running this script on Google Colab, we only had access to 40 GB of VRAM. 

To account for these errors, we: 

1. Drastically reduced our dataset (took 5 second random samples from 200 songs)
2. Limited batch size to 1, and set accumulate_grad_batches to offset this
3. Only fine-tuned lightweight LoRA/adapter layers, keeping backbone models in eval mode to save VRAM.
4. Reduced the sequence length and feature dimensions in both the dataset and model configuration
5. Turned off all unnecessary logging and callbacks (this will bite us later)
6. Regularly checked nvidia-smi/Colab VRAM usage and cleared the cache (torch.cuda.empty_cache()) between runs when necessary.
7. Restarted the runtime as needed

Even with all of these measures, we still consumed ~ 38 GB/40 GB during fine-tuning!

Our troubles also didn't end here. We found the out-of-the-box tools ACE-Step's developers provided did not run smoothly. Issues we encountered and remediated included:

1. The Hugging Face dataset loader was tightly coupled to Google Cloud Storage; we rewrote data loading logic to support local files and small datasets.
2. The default data preprocessing pipeline assumed multi-channel audio with long sequences; we adjusted it to work with our short, mono/stereo 5-second clips.
3. The model’s collate function created unnecessarily large tensors; we modified the function to pad/stack only as needed, keeping memory use minimal.
4. Batch/sequence shape mismatches in the DCAE and downstream models triggered cryptic runtime errors; we debugged and aligned input/output shapes across the data pipeline and model forward passes.
5. The training script’s defaults expected multi-GPU environments; we adjusted configs to ensure stable single-GPU and CPU operation.
6. The plot/visualization callbacks were not compatible with Colab’s resource limits; we disabled them and moved any required plots to after training or local analysis.
7. Several model components were left in train mode by default (even when not being updated), consuming extra VRAM; we set unused modules to eval mode to conserve memory.
8. Loss/metric tracking used excessive memory with large history buffers; we limited logging intervals and kept metrics concise.
9. Frequent PyTorch and library version incompatibilities (especially with Lightning and transformers) caused subtle bugs; we pinned package versions and patched code for compatibility.

After 12+ hours of work, not including training time (that would be too easy!), our final model params were: 

- `learning_rate`: `1e-4`
- `num_workers`: `2`
- `accumulate_grad_batches`: `8`
- `devices`: `1`
- `precision`: `16`
- `max_steps`: `5000`
- `every_plot_step`: `1000000`
- `every_n_train_steps`: `500`
  

## Suggested Improvements: Sampling from the Latent Space

Out-of-the-box, ACE-Step generates music deterministically using a single, fixed embedding of the input prompt. This limits the diversity and flexibility of the generated outputs. We would have modeled the prompt embedding not as a single point in latent space, but as a distribution from which we can sample.

Sampling allows for multiple musical interpretations of the same prompt and introduce a measure of uncertainty into generation. After the prompt embedding was computed we would pass it through two small feedforward networks to produce a mean vector and a log variance vector that would represent the distribution. We would then reparameterize in order to sample  from the distribution and enable backpropagation  and pass a sample from the distribution to the ACE-step decoder rather than the original prompt embedding. A KL divergence term added during training would penalize a model when the learned distribution would deviate too far from the distributional prior.

## References

Hu, E., Shen, Y., Wallis, P., et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://github.com/microsoft/LoRA)

Gong, J., Zhao, W., Wang, S., Xu, S., & Guo, J. (2025). *ACE-Step: A Step Towards Music Generation Foundation Model*. [GitHub repository](https://github.com/ace-step/ACE-Step)

---
