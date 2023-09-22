from functions import predict_parser, load_checkpoint, process_image, predict
import json

def main():
    args = predict_parser()
    
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu
    
    model = load_checkpoint(checkpoint)
    image = process_image(image_path)
    
    if gpu:
        device = 'cuda'
    else:
        device = 'cpu'
        
    top_probs, top_labels = predict(image_path, model, top_k, device) 
    
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
       
        top_flowers = [cat_to_name[str(label)] for label in top_labels]
        results = set(zip(top_labels, top_flowers, top_probs))
    else:
        results = set(zip(top_labels, top_probs))

    for result in results:
        print(result)

main()