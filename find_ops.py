from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")

# 1. Print the structural layer graph
print(model)

# 2. To see the specific tensor keys and exact shapes inside the weights file:
print(model.state_dict().keys())