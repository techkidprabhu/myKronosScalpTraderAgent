Kronos training kit
===================

Exported from the swing trading agent's dashboard. It holds the candles the
agent collected, the times of the agent's own signals, and the trainer.
No broker logins, keys or trade records are in it.

What you need on this computer
------------------------------
- An NVIDIA GPU with a current driver (8 GB of GPU memory is enough for Kronos-base)
- Python 3.11 or 3.12  (https://www.python.org/downloads/ - on Windows tick
  "Add python.exe to PATH")
- Internet the first time: PyTorch (a few GB), the Kronos model code and the
  original Kronos model (about 400 MB) are downloaded once and reused by
  every later kit. They live in the .kronos-trainer folder in your home folder.

Each day
--------
1. On the dashboard: Kronos Model Lab -> Self-learning -> "Export for GPU training".
2. Unzip the file on this computer (anywhere).
3. Windows: double-click train.bat        Linux: ./train.sh
4. Wait for the summary. It compares the fine-tuned model with the original
   on recent candles neither of them trained on.
5. If the result is BETTER, a file kronos-ft-<date>-result.zip appears in this
   folder. On the dashboard: Self-learning -> "Import trained model" and pick
   that file. (Or copy it to the agent computer and run
   ./start.sh finetune import <file>.)
6. On the dashboard the run now appears under "Training runs":
   "Shadow test" tries it beside the trading model without affecting trades;
   "Use this model" makes the agent trade with it after a restart;
   "Back to original" undoes that.

If the result is NO GAIN or WORSE there is no model to bring back - that is
normal while only a few days of candles have been collected.

Options
-------
  train.bat --steps 6000      train longer (default 4000 steps)
  train.bat --minutes 60      hard time limit (default 180)
  train.bat --batch 16        if the GPU runs out of memory (default 32)
  train.bat --keep            put the model in the result zip even if not better
  train.bat --cpu             no GPU (slow; only for testing)

Every run starts from the original Kronos model, so runs never stack on top
of each other; each day's training uses all candles collected so far.

If PyTorch cannot see the GPU: update the NVIDIA driver. For an old driver,
delete the .kronos-trainer\venv folder, set TORCH_INDEX to
https://download.pytorch.org/whl/cu126 and start again.
