import torch
from seamless_communication.models.inference import Translator

replicate_api = st.secrets['REPLICATE_API_TOKEN']

# Initialize a Translator object with a multitask model, vocoder on the GPU.
translator = Translator("seamlessM4T_medium", vocoder_name_or_card="vocoder_36langs", device=torch.device("cuda:0"), dtype = torch.float16)
