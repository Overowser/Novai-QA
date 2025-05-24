from generator import generate_response
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cuda")
test_response = generate_response("Who is noah?", "Infinite Mana In The Apocalypse", model, 10)
print(f"Response: '{test_response}'")
print(f"Type: {type(test_response)}")
print(f"Length: {len(test_response) if test_response else 'None'}")