<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# Bayes' Beats: Music from the Mood of an Image

> **Example project from Spring 2025.** This is a copy of [import-ariel/bayes_beats](https://github.com/import-ariel/bayes_beats), shown with the authors' work, commit history and branches intact. Their own write-up is [REPORT.md](REPORT.md), also published as the methodology page of their [site](https://import-ariel.github.io/bayes_beats/). Course staff wrote this README and made the edits listed under [Changes by course staff](#changes-by-course-staff), including the section on [what to learn from this project](#what-to-learn-from-this-project).

Can a text-to-music model be taught to produce a short clip that fits the mood of a picture? The team took ACE-Step, an open text-to-music diffusion model, fine-tuned it with LoRA on short clips from the MTG-Jamendo collection, and built a small website where a visitor picks an image and hears a clip. The Bayesian layer they planned, a variational distribution over the prompt embedding, was not built, and the write-up says so. The project is shown here as a generative AI project, for its engineering and its candor.

## Summary

ACE-Step (Gong et al., 2025) generates music from a text prompt and lyrics by diffusion in the latent space of a deep-compression autoencoder, with a linear transformer as the denoiser; its trainer counts 3.3 billion parameters. The team downloaded 287 MTG-Jamendo tracks across 15 genres into a cloud bucket, cropped 200 of them to 5.4 seconds at 48 kHz, wrote each track's genre as its prompt and an empty lyrics file, converted the folder with ACE-Step's dataset script, and ran ACE-Step's own trainer with the example LoRA configuration it ships with, for about 1,500 steps on a Colab A100 at batch size 1 with gradient accumulation of 8. The 16 GB checkpoints went to the team's cloud bucket and are not public. The checkpoint was loaded into ACE-Step's web interface, and the three 30-second clips on the site are the result. The site, a Jekyll page on the `ui` branch, shows three photographs; clicking one plays its clip. Earlier in the notebook the team also extracted audio features with librosa, asked GPT-3.5 for a short "vibe" description of each track and embedded it with a sentence transformer, and passed mel spectrograms through a small convolutional autoencoder. None of those outputs enter training.

## Team and contributions

Team "Adam's Family", Spring 2025.

- **Ariel Azria** ([@import-ariel](https://github.com/import-ariel)): the repository, the data-preparation and fine-tuning notebook, the ACE-Step setup, the site's first page and selection flow, the mood images and audio clips.
- **Gabe Barrett** ([@Jibbie17](https://github.com/Jibbie17)): the write-up.
- **Kwaku Ofori-Atta** ([@oa-kwaku](https://github.com/oa-kwaku)): the scripts that stage the audio in cloud storage, and the site's layout, styling and methodology page.
- **Danielle Rosenthal**: ran the preprocessing pipeline, whose saved outputs carry her account; there are no commits under her name.

The project ran from May 3 to 30, 2025. The attributions follow the [commit history](https://github.com/uchicago-dsi-bayesianml/sp25-bayes-beats/commits/main) on both branches.

## Getting started

There is nothing to run end to end: the fine-tuned checkpoints are in the team's private cloud bucket, and the notebooks are Colab notebooks that read from it. The quickest way in is the [demo site](https://import-ariel.github.io/bayes_beats/); the `ui` branch holds the site, with the three clips under `assets/audio/moods/`.

To reproduce the fine-tune, follow `pipeline/ace_data_preprocessing.ipynb`: install ACE-Step from its repository, obtain MTG-Jamendo audio as described under [Data](#data), sample and crop the tracks to five seconds, write a `<name>_prompt.txt` holding the genre and an empty `<name>_lyrics.txt` next to each MP3, run ACE-Step's `convert2hf_dataset.py` (with its glob changed to descend into subfolders, as the notebook notes), then run ACE-Step's `trainer.py` with the arguments in the notebook. The run needs about 40 GB of GPU memory. `tutorials/loading_finetuned_acestep.ipynb` on the `ui` branch shows how the checkpoint was served through ACE-Step's web interface.

The `requirements.txt` in this repository belongs to an earlier, abandoned approach, lo-fi generation from MIDI with music21 and PyMC, and does not cover the pipeline; the notebook installs what it needs.

## Data

Data files are never committed to this repository.

- **MTG-Jamendo** ([MTG/mtg-jamendo-dataset](https://github.com/MTG/mtg-jamendo-dataset)): over 55,000 full tracks with genre, instrument and mood/theme tags, the audio under Creative Commons licenses that vary by track and the annotations under CC BY-NC-SA 4.0. The team sampled 20 tracks in each of 15 genres and used 200 of them. The dataset repository has the download scripts; `data/upload_data.ipynb` shows the team's staging into cloud storage.
- **Images:** a 504-image sample of a dataset the write-up calls "1-Million-Pairs-Image-Caption-Data-Of-General-Scenes", of which three photographs appear on the site. Course staff could not find this dataset or its license on Hugging Face.
- **vikhyatk/lofi** (CC BY-NC 4.0) appears in the upload notebook from an earlier plan and is not used.

## Project structure

| Path | Contents |
|---|---|
| `REPORT.md` | The authors' write-up: motivation, ACE-Step, the pipeline, fine-tuning and its memory troubles, the planned variational layer. |
| `pipeline/ace_data_preprocessing.ipynb` | The whole pipeline: feature extraction and GPT "vibe" labels, sampling and cropping, conversion to ACE-Step's dataset format, the trainer call, checkpoint upload and serving. `ace_data_preprocessing_cleaned.html` is an export of it. |
| `data/upload_data.ipynb` | Staging Jamendo audio in a cloud bucket. |
| `data/old_data_cleaning.py`, `tutorials/getting_the_data.ipynb` | The abandoned MIDI approach, adapted from a Towards Data Science post the notebook cites. |
| `tutorials/ace_finetuning_draft.ipynb` | An early sketch of the dataset format. |
| `images/` | ACE-Step's architecture figure, from its repository. |
| Branch `ui` | The Jekyll site: `index.html`, `select.html`, `methodology.md`, the three clips and images under `assets/`, and `tutorials/loading_finetuned_acestep.ipynb`. |

## Method, as built

**Base model.** ACE-Step, used through its own training and serving code. The write-up's account of the model is mostly right, with one slip: ACE-Step denoises continuous latents of its audio autoencoder, it does not generate "discrete audio tokens".

**Data.** 200 clips of 5.4 seconds at 48 kHz, one per track, drawn 20 per genre from 15 genres. The prompt for each clip is its genre word; the lyrics are empty.

**LoRA.** The example configuration that ships with ACE-Step for a Chinese-rap adapter, unchanged: 16.4 million trainable parameters of 3.3 billion.

**Training.** ACE-Step's `trainer.py`, learning rate 1e-4, half precision, batch size 1 with gradient accumulation of 8, a limit of 5,000 steps. The run was stopped during epoch 76 after at least 1,500 steps, the last checkpoint saved. Plotting and most logging were turned off to save memory, so there is no loss curve.

**Generation.** The checkpoint served through ACE-Step's web interface; three 30-second clips saved and placed on the site, one per photograph. The site plays a fixed clip per image; nothing is generated at browsing time.

**Not used.** The librosa features, the GPT-3.5 descriptions and their embeddings, and the latents of the small convolutional autoencoder are computed and saved and never reach the training set. That autoencoder is randomly initialized and never trained; it is not ACE-Step's deep-compression autoencoder, though the write-up uses the same name for both.

### What is Bayesian here

Nothing was built, as the authors say. Their plan was a variational bottleneck on the conditioning signal: two small networks mapping the prompt embedding to a mean and a log-variance, a sample drawn by the reparameterization trick, a KL term pulling the distribution toward a standard normal prior, and the sample passed to the decoder in place of the point embedding. That is the course's variational autoencoder material applied to the prompt, and it would have given what the write-up wanted: several musical readings of one prompt, and a measure of how uncertain the model is about what a mood means. ACE-Step itself is a diffusion model, a deep latent-variable model trained with a variational bound, which the course also covers, but the team used it through its trainer as a black box, so there is nothing of that to read in this repository.

## What to learn from this project

Course staff added this section. The infrastructure work is real and the write-up is honest about what was cut; the points below are about the distance between what the write-up describes and what the code does.

- **Write the README from the code you ran.** The write-up describes a pipeline in which mood descriptions, audio features and autoencoder latents condition the model; the code trains on genre words. It describes an evaluation by CLAP similarity between prompt and audio; no evaluation code exists. It lists nine fixes made to ACE-Step's trainer, loader and collate function; the repository records one changed line in a conversion script. Each of these reads as a plan written before the work and not revised after it.
- **Check the conditioning signal.** Fifteen genre words cannot carry a mood, so the fine-tune cannot do what the site promises. The three clips were generated and assigned to the photographs by hand. A reader should know that before drawing conclusions from the demo.
- **Keep evidence that training happened.** With logging off, no loss curve, no sample set and private checkpoints, the only evidence of the fine-tune is three clips and a notebook output. A loss plot and ten clips with their prompts, from before and after fine-tuning, would have been enough.
- **A Bayesian plan is not Bayesian work.** The variational head could have been built and tested at small scale, on a frozen small model or on the prompt embeddings alone, while the large fine-tune was out of reach. Scope cuts are normal; cutting the one Bayesian component from a Bayesian course project leaves nothing for the course to assess.
- **Small things.** The requirements file belongs to an abandoned approach; the write-up's link to the pipeline notebook points at a file that exists only on the other branch; a `train_lora.py` on the `ui` branch prints "hi, world"; IDE settings are committed there; no license; the fourth author's work is invisible in the history.

What the project does well: candor about the scope cut and about labelling with GPT because the credits were expiring; a working data pipeline with cloud staging, parallel downloads and conversion to a third party's format; getting a 3.3-billion-parameter model to fine-tune inside 40 GB of memory; a demo anyone can use; and sources cited, including the blog post behind the approach it abandoned.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Spring 2025. It is shown here as an example for later cohorts.

## Changes by course staff

Course staff forked the authors' repository, with both branches, and made these edits. The authors' code, notebooks and site are unchanged.

- Renamed the authors' README to `REPORT.md`, unchanged, and added this README with the program banner.
- Added a `.gitignore` for data, audio, checkpoints and tooling.

## License

The authors have not added a license, so their work is shown here under GitHub's terms for public repositories. Ask them before reusing it. ACE-Step is Apache 2.0; the Jamendo audio carries its own Creative Commons licenses per track; the three photographs come from a dataset whose terms course staff could not find.
