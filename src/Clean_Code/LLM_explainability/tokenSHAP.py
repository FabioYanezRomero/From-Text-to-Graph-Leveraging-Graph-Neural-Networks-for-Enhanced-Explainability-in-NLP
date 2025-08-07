from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from token_shap import TokenSHAP, StringSplitter, HuggingFaceModel

MODEL_NAME = "textattack/bert-base-uncased-SST-2"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
hf_model = HuggingFaceModel(model=model, tokenizer=tokenizer, device='cpu')
splitter = StringSplitter()
explainer = TokenSHAP(hf_model, splitter)

ds = load_dataset("stanfordnlp/sst2", split="validation")


results = {}
for i in range(3):
    prompt = ds[i]["sentence"]
    print(f"\nSentence {i+1}: {prompt}")
    df = explainer.analyze(prompt, sampling_ratio=0.0, print_highlight_text=True)
    print(df)
    results[prompt] = df

