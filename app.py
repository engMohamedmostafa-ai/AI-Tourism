import base64
import csv
import json
import mimetypes
import os
import re
from difflib import get_close_matches
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "web"
QWEN_ADAPTER_DIR = ROOT / "egypt-tourism-qwen-lora" / "egypt-tourism-qwen-lora"
CGAN_DIR = ROOT / "egypt_landmarks_cgan"
GENERATED_DIR = ROOT / "generated_images"
ASSET_FILES = {
    "/assets/pyramid.png": ROOT / "generated_Great_Pyramid_of_Giza.png",
    "/assets/bent-pyramid.png": ROOT / "generated_Bent_Pyramid.png",
    "/generated_Great_Pyramid_of_Giza.png": ROOT / "generated_Great_Pyramid_of_Giza.png",
    "/generated_Bent_Pyramid.png": ROOT / "generated_Bent_Pyramid.png",
}

HOST = "127.0.0.1"
PORT = 8000

_chatbot = None
_image_generator = None
_fallback_chatbot = None


def is_image_request(prompt):
    text = prompt.lower()
    image_words = ("generate", "draw", "create", "make", "show me")
    return any(word in text for word in image_words) and any(
        word in text for word in ("image", "picture", "photo", "landmark")
    )


def clean_generation_prompt(prompt):
    text = prompt.lower()
    text = re.sub(r"\b(generate|draw|create|make|show me|an|a|the|image|picture|photo|of|for|landmark)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or prompt


class ChatbotModel:
    def __init__(self):
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing chatbot dependencies. Install them with: "
                "pip install -r requirements.txt"
            ) from exc

        adapter_config = json.loads((QWEN_ADAPTER_DIR / "adapter_config.json").read_text(encoding="utf-8"))
        base_model = adapter_config["base_model_name_or_path"]
        allow_download = os.environ.get("ALLOW_MODEL_DOWNLOAD", "").lower() in {"1", "true", "yes"}

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            QWEN_ADAPTER_DIR,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=not allow_download,
        )
        self.model = PeftModel.from_pretrained(base, QWEN_ADAPTER_DIR)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def answer(self, prompt):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an Egypt tourism guide assistant. Answer clearly, "
                    "politely, and provide practical tourism advice. Do not invent facts."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        return answer.strip()


class DatasetFallbackChatbot:
    def __init__(self):
        self.rows = []
        dataset_path = ROOT / "data" / "egypt_tourism_finetune_dataset.csv"
        with dataset_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                row["_search"] = " ".join(
                    [row.get("user", ""), row.get("assistant", ""), row.get("place", ""), row.get("city", ""), row.get("category", "")]
                ).lower()
                self.rows.append(row)

    def answer(self, prompt):
        words = set(re.findall(r"[a-z0-9]+", prompt.lower()))
        best_row = None
        best_score = -1
        for row in self.rows:
            score = sum(1 for word in words if word in row["_search"])
            place = row.get("place", "").lower()
            if place and place in prompt.lower():
                score += 8
            if score > best_score:
                best_row = row
                best_score = score

        if best_row and best_score > 0:
            return best_row["assistant"]
        return (
            "I can help with Egyptian landmarks, cities, locations, and travel tips. "
            "Try asking about places like the Pyramids of Giza, the Great Sphinx, Luxor, Aswan, Siwa, or Alexandria."
        )


class ConditionalGeneratorModel:
    def __init__(self):
        try:
            import torch
            import torch.nn as nn
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Missing image generation dependencies. Install them with: "
                "pip install -r requirements.txt"
            ) from exc

        self.torch = torch
        self.Image = Image

        config = json.loads((CGAN_DIR / "cgan_config.json").read_text(encoding="utf-8"))
        self.classes = config["classes"]
        self.class_to_idx = config["class_to_idx"]
        self.nz = config["nz"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class ConditionalGenerator(nn.Module):
            def __init__(self, nz, ngf, nc, num_classes, embed_size=128):
                super().__init__()
                self.label_emb = nn.Embedding(num_classes, embed_size)
                self.main = nn.Sequential(
                    nn.ConvTranspose2d(nz + embed_size, ngf * 16, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 16),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                    nn.Tanh(),
                )

            def forward(self, noise, labels):
                label_embedding = self.label_emb(labels)
                label_embedding = label_embedding.view(label_embedding.size(0), label_embedding.size(1), 1, 1)
                return self.main(torch.cat([noise, label_embedding], dim=1))

        self.model = ConditionalGenerator(
            config["nz"],
            config["ngf"],
            config["nc"],
            config["num_classes"],
        ).to(self.device)
        state = torch.load(CGAN_DIR / "conditional_generator.pth", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        GENERATED_DIR.mkdir(exist_ok=True)

    def match_class(self, prompt):
        query = clean_generation_prompt(prompt).replace(" ", "_")
        normalized = {name.lower(): name for name in self.classes}
        aliases = {
            "bent_pyramid": "Bent_Pyramid",
            "bent pyramid": "Bent_Pyramid",
            "pyramids": "Giza_pyramid_complex",
            "pyramid": "Great_Pyramid_of_Giza",
            "giza": "Giza_plateau" if "Giza_plateau" in self.class_to_idx else "Giza_Plateau",
            "karnak": "Great_Hypostyle_Hall_of_Karnak",
            "philae": "Temple_of_Isis_in_Philae",
            "siwa": "Siwa",
        }
        lowered = query.lower()
        for alias, class_name in aliases.items():
            if alias in lowered and class_name in self.class_to_idx:
                return class_name
        for class_name in self.classes:
            simple = class_name.lower().replace("_", " ")
            if simple in prompt.lower() or class_name.lower() in lowered:
                return class_name
        matches = get_close_matches(lowered, normalized.keys(), n=1, cutoff=0.35)
        return normalized[matches[0]] if matches else "Great_Pyramid_of_Giza"

    def generate(self, prompt):
        torch = self.torch
        class_name = self.match_class(prompt)
        label = torch.tensor([self.class_to_idx[class_name]], device=self.device)
        noise = torch.randn(1, self.nz, 1, 1, device=self.device)

        with torch.no_grad():
            image = self.model(noise, label).detach().cpu()[0]

        image = ((image + 1) / 2).clamp(0, 1)
        image = (image.permute(1, 2, 0).numpy() * 255).astype("uint8")
        output_path = GENERATED_DIR / f"{class_name}_{len(list(GENERATED_DIR.glob('*.png'))) + 1}.png"
        self.Image.fromarray(image).save(output_path)

        encoded = base64.b64encode(output_path.read_bytes()).decode("ascii")
        return {
            "landmark": class_name,
            "image_url": f"/generated/{output_path.name}",
            "image_base64": f"data:image/png;base64,{encoded}",
        }


def get_chatbot():
    global _chatbot
    if _chatbot is None:
        _chatbot = ChatbotModel()
    return _chatbot


def get_fallback_chatbot():
    global _fallback_chatbot
    if _fallback_chatbot is None:
        _fallback_chatbot = DatasetFallbackChatbot()
    return _fallback_chatbot


def get_image_generator():
    global _image_generator
    if _image_generator is None:
        _image_generator = ConditionalGeneratorModel()
    return _image_generator


class TourismHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/":
            self.serve_file(STATIC_DIR / "index.html")
            return
        if path in ("/styles.css", "/script.js"):
            self.serve_file(STATIC_DIR / path.removeprefix("/"))
            return
        if path.startswith("/web/"):
            self.serve_file(STATIC_DIR / path.removeprefix("/web/"))
            return
        if path.startswith("/generated/"):
            self.serve_file(GENERATED_DIR / path.removeprefix("/generated/"))
            return
        if path in ASSET_FILES:
            self.serve_file(ASSET_FILES[path], extra_roots=[ROOT.resolve()])
            return
        if path == "/api/classes":
            config = json.loads((CGAN_DIR / "class_mapping.json").read_text(encoding="utf-8"))
            self.send_json({"classes": config["classes"]})
            return
        self.send_error(404, "Not found")

    def do_POST(self):
        if urlparse(self.path).path != "/api/chat":
            self.send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            prompt = str(payload.get("prompt", "")).strip()
            if not prompt:
                self.send_json({"error": "Prompt is required."}, status=400)
                return

            if is_image_request(prompt):
                result = get_image_generator().generate(prompt)
                self.send_json({"type": "image", "prompt": prompt, **result})
            else:
                try:
                    answer = get_chatbot().answer(prompt)
                    source = "qwen-lora"
                except Exception as exc:
                    answer = get_fallback_chatbot().answer(prompt)
                    source = "dataset-fallback"
                    answer = (
                        f"{answer}\n\n"
                        "Note: Qwen LoRA could not be loaded locally yet. "
                        "Set ALLOW_MODEL_DOWNLOAD=1 before running app.py if you want the server to download the Qwen base model. "
                        f"Details: {exc}"
                    )
                self.send_json({"type": "answer", "prompt": prompt, "answer": answer, "source": source})
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def serve_file(self, path, extra_roots=None):
        try:
            resolved = path.resolve()
            allowed_roots = [STATIC_DIR.resolve(), GENERATED_DIR.resolve()]
            if extra_roots:
                allowed_roots.extend(extra_roots)
            if not any(str(resolved).startswith(str(root)) for root in allowed_roots):
                self.send_error(403, "Forbidden")
                return
            if not resolved.is_file():
                self.send_error(404, "Not found")
                return
            mime_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(resolved.stat().st_size))
            self.end_headers()
            self.wfile.write(resolved.read_bytes())
        except OSError:
            self.send_error(404, "Not found")

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    print(f"Egypt Tourism Assistant running at http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), TourismHandler).serve_forever()


if __name__ == "__main__":
    main()
