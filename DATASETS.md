# Dataset provenance

## Arabic words sign language video dataset

- Source: Zenodo record 8035320, DOI 10.5281/zenodo.8035320.
- Published description: 3,000 videos, 30 signs, five volunteers.
- Downloaded file: `data/raw/Dataset.rar`.
- Verified MD5: `c1abb115cdf8b90919d8b313e297853f`.
- Observed archive contents on 2026-09-27: five MP4 files, all under the single Arabic label `اب`.
- License: no license value was displayed on the record. Treat as research/test-only until the owner clarifies reuse rights.

The mismatch between the description and actual archive is recorded intentionally. Do not report results from this download as a 30-class evaluation.

## ARSLW Arabic Sign Language Words

- Source: https://github.com/Mimouni-Abdessamed/ARSLW
- License: MIT as stated in the repository.
- Media: 224x224 JPG word-sign frames.
- Categories: colors, health, education, family, and verbs.
- Local directory: `data/ARSLW/Dataset` (ignored by Git).

ARSLW is used for the multi-class visual baseline and sign dictionary. Because it contains still frames, it does not measure temporal sign recognition. A signer-independent video benchmark such as KArSL-100 should be added before production claims are made.

## KArSL-100 temporal word recognition

- Official source: https://hamzah-luqman.github.io/KArSL/download_video_100.html
- Public mirror used for resumable frame downloads: https://huggingface.co/datasets/FatimahEmadEldin/karsl-502-arabic-sign-language-v2
- Vocabulary: 100 word/phrase signs, SignIDs 0071-0170.
- Official design: three professional signers, 50 repetitions per sign.
- Reference checkpoint: `references/karsl_word_recognition/*3_signers*.h5`.

The reference notebook reports 99.625% on its own held-out split. That split contains all
three signers in training and testing, so it is not evidence of signer-independent or webcam
accuracy. `scripts/train_karsl100.py` instead holds out an entire signer.

The Hugging Face mirror has no explicit license. Use it and the reference checkpoint only for
research/demo work until commercial rights are confirmed. The official Drive may temporarily
reject downloads because of its quota; the mirror download is resumable.

## ArSL-Models pretrained word checkpoint

- Source: https://huggingface.co/FatimahEmadEldin/ArSL-Models
- File: `sign_word_t5_classifier_best_3d.pth`
- License: MIT according to the model card.
- Architecture: T5-small encoder over normalized MediaPipe hand landmarks.
- Vocabulary: 10 isolated Arabic word signs.
- Important limitation: the model card contains no verified accuracy or F1 values and the model tiles a single frame rather than learning a complete temporal sequence.
