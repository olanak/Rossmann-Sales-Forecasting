install:
    pip install -r requirements.txt -r requirements-dev.txt

test:
    pytest tests/

lint:
    flake8 src/
    black --check src/

train:
    python src/models/train.py

serve:
    uvicorn api.main:app --reload