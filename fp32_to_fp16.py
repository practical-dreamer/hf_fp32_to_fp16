from transformers import AutoModel, PushToHubCallback

USERNAME = ""
TOKEN = ""
MODEL_PATH = ""
CACHE_DIR = ""
NEW_MODEL_DIR = ""
NEW_MODEL_NAME = ""

# Define printG for print green
def printG(text):
    """Print text in green."""
    print("\033[92m" + text + "\033[0m")

# Fetch the model using the token and save it to the specified cache directory
printG("Downloading model")
model = AutoModel.from_pretrained(MODEL_PATH, use_auth_token=TOKEN, cache_dir=CACHE_DIR)

# Convert to fp16
printG("Model Download Complete... Converting to fp16")
model = model.half()

# Save the fp16 model locally
printG("Conversion complete... Saving")
model.save_pretrained(NEW_MODEL_DIR)

printG("Model Saving Complete")
