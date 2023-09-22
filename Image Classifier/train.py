from functions import train_parser, train_model

def main():
    args = train_parser()
    
    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    batch_size = args.batch_size
    hidden_units = args.hidden_units
    learning_rate = args.learning_rate
    dropout = args.dropout
    epochs = args.epochs
    device = args.device
    
    train_model(data_dir, save_dir, arch, batch_size, hidden_units, learning_rate, dropout, epochs, device)
    
main()