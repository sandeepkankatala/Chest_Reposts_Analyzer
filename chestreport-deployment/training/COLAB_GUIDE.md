# Train in Colab, run ChestReport with a GPU server

This workflow implements **text-only chest X-ray report explanation** with `google/medgemma-1.5-4b-it`. It does not analyze uploaded X-ray images. The authored synthetic starter teaches the app's JSON format, negation, uncertainty, evidence citations and abstention. It has no patient records and has not been clinically reviewed. It is suitable for an assignment experiment, not a clinical model.

The prepared notebook and GPU scripts have been checked locally for syntax and integration contracts. **GPU training, GPU memory use and real generation have not been executed in this workspace.** Your local 3 GB GTX 1050 is too small for this recipe. Free Colab GPU availability is variable; a T4 with about 16 GB is the intended low-memory starting point, not a verified guarantee. Use the smoke test first. A separate 24 GB GPU is the suggested inference starting point; benchmark it before paying for longer use.

## 1. Get model access

1. Sign in to [the MedGemma model page](https://huggingface.co/google/medgemma-1.5-4b-it) and request/accept its Health AI Developer Foundations terms yourself.
2. Create a Hugging Face token with read access to the approved model. Do not paste it into chat, Git, browser frontend code or notebook source.
3. Open [Google Colab](https://colab.research.google.com/), choose **File → Upload notebook**, and upload `training/MedGemma_ChestReport_Colab.ipynb` from this project.
4. Choose **Runtime → Change runtime type → T4 GPU** (or another available CUDA GPU).
5. In Colab's Secrets panel (key icon), add `HF_TOKEN` and enable notebook access. The notebook also offers a hidden password prompt if no Secret is available.

You do not need MIMIC access or a GitHub repository for this starter. The notebook contains the necessary source files. After editing training code locally, rebuild it with `python -m training.build_colab_notebook` before uploading.

## 2. Run the notebook in order

The cells install pinned ML libraries while preserving Colab's CUDA PyTorch pair, check the GPU and gated-model access, resolve the exact model revision, generate the synthetic data, and display an example. If package installation requests a session restart, restart and rerun from the source setup cell.

The dataset has **200 synthetic reports / 800 examples**: 640 train, 80 validation, 80 test. Each report has plain and clinical summaries with extraction, one supported question and one treatment-abstention question. All variants of a report remain in one split. Individual sentence templates are shared across splits, so held-out scores measure learned format and combinations, not clinical generalization. The data remain marked `reviewed: false`; `--allow-synthetic-starter` is an explicit experiment opt-in, not a claim of medical review.

Run the two-step memory smoke test, then the full two-epoch run. The smoke adapter is kept separate and cannot be served as the completed adapter. Defaults: four-bit NF4, language-layer LoRA rank 8, batch 1, accumulation 8, 1,024-token examples, gradient checkpointing, and response-only loss. Examples longer than the budget are counted and skipped, never silently cut off. The vision layers remain frozen.

Download `chestreport-deployment.zip` as soon as training finishes. It contains the adapter, its base-model revision, training manifest, validation loss, synthetic data and GPU service files. Free Colab storage is ephemeral. The optional base-versus-adapter cell uses the same 16 held-out prompts for both models and produces `comparison.json`; set its limit to 0 for all 80. Download the updated ZIP after evaluation. Preserve the executed notebook with its output and package versions for your assignment evidence.

Compare JSON/schema validity, exact finding/status/evidence matches, citation substring validity, abstention, summary exact match and generation time. Exact matches penalize valid paraphrases; valid quotes do not establish that a claim is medically correct. Inspect predictions manually. Do not claim improvement unless the actual saved results show it. Add clinician-reviewed independent cases before making clinical accuracy claims.

If Colab reports CUDA out-of-memory, restart the runtime to release memory and try `MAX_LENGTH = 768`. Inspect the retained-example counts. If memory still fails, move the same notebook to a larger GPU; free T4 success is not guaranteed. Do not truncate labels or serve a partial training checkpoint as the final model.

## 3. Run the adapter on a separate GPU VM

Choose a Linux GPU VM that supports Docker and NVIDIA Container Toolkit. Provisioning it costs money; this project does not create any paid resource. Stop/delete billable resources when finished. The following uses SSH to keep the inference port private and needs no domain, TLS certificate or public model API.

Upload the downloaded ZIP to your chosen GPU VM and extract it into a new directory, for example `~/chestreport-deployment`. Inside it are `adapter/`, `backend/` and `training/`. On that VM, verify `nvidia-smi` and Docker GPU access work, then run:

```bash
cd ~/chestreport-deployment
read -rsp 'Hugging Face read token: ' HF_TOKEN; echo
export HF_TOKEN
read -rsp 'Inference API key (random, at least 24 characters): ' MEDGEMMA_API_KEY; echo
export MEDGEMMA_API_KEY
docker compose -f training/compose.gpu.yaml --project-directory . up --build -d
docker compose -f training/compose.gpu.yaml --project-directory . logs --tail 40 medgemma
```

Generate/store the inference key in a password manager. This is a different secret from the Hugging Face download token. They are passed to the container at runtime, not baked into the image. Administrators of the GPU host can inspect container environment variables. Keep the host trusted and access restricted. The first start downloads the base model; readiness waits for both base and adapter loading. Use one Uvicorn worker; additional workers each duplicate GPU memory.

On your Windows machine, keep this tunnel running in a separate terminal (replace username and hostname):

```powershell
ssh -N -L 127.0.0.1:8001:127.0.0.1:8001 your-user@your-gpu-host
```

This SSH tunnel connects to your separately hosted GPU VM, **not to free Colab**. No inference web server or tunnel is started in Colab. The GPU service binds to localhost on the VM; do not expose port 8001 publicly. If your provider gives you an HTTPS endpoint instead, use that URL with the connection command below and retain API-key protection. Providers with an additional HTTP authentication gateway can use `-GatewayAuth`; the app sends that token in `Authorization` and the inference key in `X-ChestReport-Key`.

## 4. Connect the local application

Stop any existing local app using Ctrl+C. Open PowerShell in the `chestreport` project directory and run:

```powershell
.\connect-model.ps1 -Url http://127.0.0.1:8001/v1
```

Enter the inference API key at the hidden prompt. The launcher checks authenticated health, requires a loaded adapter, sends a synthetic report for real summarization, and asks a real grounded question. It validates the response shapes and evidence quotes before starting the local web app. Secrets stay in process environment/memory and are never sent to the browser. `-CheckOnly` runs these checks without starting the app. `-AllowBase` is only for an intentional comparison with a server running the base model without an adapter.

Open **http://127.0.0.1:8000**, choose **Upload a report → Paste text**, and paste the text from `synthetic-demo-report.txt` in the deployment ZIP. This creates a **new** report processed by the GPU model. Existing demo reports are not automatically regenerated; upload them again. Built-in **Explore a sample** cards intentionally retain their labeled demo responses. Ask a specific question about a named finding in the new report. Export its JSON to see `analysis.mode: medgemma` and the model-backed question response.

The privately published Sites page remains a static demo. It cannot run the Python backend or GPU model. Use the local URL for the actual AI demonstration; hosting the full web backend is a separate deployment step.

GET `/api/health` now reports `model_status`, `fine_tuned` and `adapter_version` from the authenticated GPU server. `fine_tuned: true` means an adapter loaded successfully, **not** that it passed clinical evaluation. Generation or citation failures are surfaced, never replaced by demo responses. A disconnected server, invalid key, truncated JSON or invented evidence can cause the check to fail; inspect GPU server readiness and the saved held-out predictions.

## 5. Assignment evidence and next dataset

Capture the actual training loss, package/GPU manifest, saved adapter, base/adapter comparison, and a newly uploaded report with a generated summary and cited answer. Include the synthetic-data limitation in the report. Keep screenshots free of tokens.

After obtaining individual approved access, MIMIC-CXR can supply real chest-radiology report text. `prepare_mimic.py` only generates Findings → Impression candidates; these need review and application-format JSON targets before training this app. Do not combine them blindly with the synthetic format dataset, and do not upload restricted records to third-party services without the data-use terms permitting it.

References: [MedGemma model and access terms](https://huggingface.co/google/medgemma-1.5-4b-it), [official MedGemma fine-tuning examples](https://github.com/google-health/medgemma/tree/main/notebooks), [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization), [Colab resource and usage limits](https://research.google.com/colaboratory/faq.html). Google's multimodal example is not a guarantee that this custom low-memory text-only recipe will fit free Colab.
