import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group
import NN_models_multi as NN_mm

torch.set_default_dtype(torch.double)

#from NN_models_multi import scaleTensors, unscaleTensors, gradScalings, write_data_file, fourier_transform_time
def main(save_every: int, total_epochs: int, snapshot_path: str = "snapshot.pt" ):
    NN_mm.ddp_setup(rank, world_size)
    dataset, model, optimizer = NN_mm.load_train_objs()
    train_data = NN_mm.prepare_dataloader(dataset, batch_size)
    trainer = NN_mm.Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])

    main(save_every, total_epochs)
