import os
from torch.distributed import init_process_group, destroy_process_group

init_process_group(backend='gloo')
print(os.environ["LOCAL_RANK"])
destroy_process_group()
