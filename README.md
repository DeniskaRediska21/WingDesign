Требования:

+ Python >= 3.11, < 3.13
+ uv >= 0.7.4

## Разработка

### Установка и запуск тестового скрипта

```bash
git clone https://github.com/DeniskaRediska21/WingDesign.git
cd WingDesign
python -m venv .venv
pip install uv  # installs to global python, if undesirable use installation instructions from https://docs.astral.sh/uv/getting-started/installation/
uv sync --all-groups
uv run main.py
```
