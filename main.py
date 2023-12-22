import argparse, os
from model.models import *
from utils.config import Config 
from utils.dataset import *
from utils.metrics import *

availableTitleModels = {
    "LSTM": LSTM,
    "None": None
}
availablePosterModels = {
    "TinyVGG": TinyVGG,
    "DenseNet121": DenseNet121Model,
    "DenseNet169": DenseNet169Model,
    "VGG16": VGG16Model,
    "None": None
}
availableURatingModels = {
    "simpleNN": simpleNN,
    "None": None
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main argument parser")
    parser.add_argument("run_mode", choices=("train", "test"), help="Main running mode of the program")
    # arguments for model
    parser.add_argument("--title_model", type=str,  default="None", choices=availableTitleModels.keys(), help="The type of model to be ran")
    parser.add_argument("--poster_model", type=str,  default="None", choices=availablePosterModels.keys(), help="The type of model to be ran")
    parser.add_argument("--urating_model", type=str,  default="None", choices=availableURatingModels.keys(), help="The type of model to be ran")
    parser.add_argument("--model_config", type=str, default="model_config.yaml", help="Specify the model config file to load")
    # arguments for dataset
    parser.add_argument("--dataset_dir", type=str, default="dataset/", help="Location of dataset")
    parser.add_argument("--use_dropped_data", type=bool, default=False, help="Choosing to use dropped data or not")
    parser.add_argument("--batch_size", type=int, default=32, help="Specify the batch_size to run the model with.")
    parser.add_argument("--image_size", type=int, default=256, help="Specify the image size for the model")
    # arguments for training only
    parser.add_argument("--max_epochs", type=int, default=20, help="Specify the number of epochs to train the model")
    parser.add_argument("--saved_model_dir", type=str, default="model/saved_models/", help="Location to save the model after training(checkpoint)")
    # arguments for testing only
    parser.add_argument("--checkpoint", type=str, default=None, help="Specify to load the checkpoint into model.")
    # # arguments for inference only
    # parser.add_argument("--input_file", type=str, help="Infer mode: Provide the location of Input file")
    # parser.add_argument("--prediction_file", type=str, help="Infer mode: Provide location of Output file which is predicted from Input file")
    
    args = parser.parse_args()

    # create directory if not exist
    os.makedirs(args.saved_model_dir, exist_ok=True)
    # load dataset
    train_dataloader, val_dataloader, test_dataloader, sizes = getDataLoader(args.dataset_dir, args.use_dropped_data, args.batch_size, args.image_size)
    vocab_size, user_size = sizes # unpack tuple
    # load config
    config = Config(args.model_config)
    # for k, v in config.items():
    #     print(k, v)

    # load model
    titleModel = posterModel = uratingModel = None
    if args.title_model != "None":
        titleParam = config[args.title_model]
        titleParam['input_size'] = vocab_size
        titleModel = availableTitleModels[args.title_model](**titleParam)
    if args.poster_model != "None":
        posterParam = config[args.poster_model]
        if 'image_size' in posterParam:
            posterParam['image_size'] = args.image_size
        posterModel = availablePosterModels[args.poster_model](**posterParam)
    if args.urating_model != "None":
        urParam = config[args.urating_model]
        urParam['input_size'] = user_size
        uratingModel = availableURatingModels[args.urating_model](**urParam)
    model = theModel(titleModel, posterModel, uratingModel)

    # train
    if args.run_mode == "train":
        trainer = pl.Trainer(max_epochs=args.max_epochs, default_root_dir=args.saved_model_dir)
        trainer.fit(model, train_dataloader, val_dataloader)
    
    # test
    if args.run_mode == "test":
        trainer = pl.Trainer()
        res = trainer.predict(model, dataloaders=test_dataloader, ckpt_path=args.checkpoint)

        pred = torch.cat([ep[0] for ep in res])
        truth = torch.cat([ep[1] for ep in res])
        # pred_1 = normalize(pred, topk=True)
        
        print_metrics(pred, truth, thres=0.8)
