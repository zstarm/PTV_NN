import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group
import NN_models_multi as NN_mm

torch.set_default_dtype(torch.double)

#from NN_models_multi import scaleTensors, unscaleTensors, gradScalings, write_data_file, fourier_transform_time
def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt", device: str = 'CPU' ):
    NN_mm.ddp_setup(device_type=device.lower())
    dataset, model, optimizer = NN_mm.load_train_objs()
    train_data = NN_mm.prepare_dataloader(dataset, batch_size)
    trainer = NN_mm.Trainer(model, train_data, optimizer, save_every, snapshot_path, device_type=device.lower())
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--device', default='cpu', type=str, help='Device type running the distributed processes')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device')
    args = parser.parse_args()
    main(args.save_every, args.total_epochs, args.batch_size, device=args.device)
