# remember to define folder path! !

folderpath=hydra_results/generation/MLP/23-11_H14-M17-S32/

PYTHONPATH=src uv run python eval/eval.py \
  --samples "${folderpath}sequence.csv" \
  --out "${folderpath}evaluation"
  