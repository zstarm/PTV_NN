import os
from torch.distributed import init_process_group, destroy_process_group

init_process_group(backend='gloo')
os.environ["GLOO_SOCKET_IFNAME"] = "en0"
print(os.environ["LOCAL_RANK"])
destroy_process_group()
