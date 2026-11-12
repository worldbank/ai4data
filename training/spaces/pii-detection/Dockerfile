FROM python:3.12-slim

# non-root user (HF Spaces convention; gives a writable HOME for the HF cache the
# model is pulled into at boot).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/tmp

WORKDIR /home/user/app

# CPU-only torch (the default PyPI wheel is CUDA and huge); transformers 5.11 is the
# stack the model's bidirectional LFM2 remote code was validated against.
RUN pip install --no-cache-dir --user torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

# One worker (single shared model), threads for concurrent static/detect conns.
# No --preload: the worker pulls + loads the model in its own process (avoids the
# torch/OpenMP fork-after-threads deadlock). --timeout 0 keeps the ~1.4GB first-boot
# download from tripping the worker timeout.
CMD ["gunicorn", "-w", "1", "--threads", "8", "-k", "gthread", "-b", "0.0.0.0:7860", "--timeout", "0", "app:app"]
