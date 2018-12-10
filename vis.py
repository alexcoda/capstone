import torch
from PIL import Image
import cv2
import numpy as np
import argparse
import utils
from model import DANet
from utils import get_dataloader
import pdb
import torch.nn.functional as F

def get_image(source_name):
    test_source_loader = get_dataloader(source_name, False, args)
    for data, target in test_source_loader:
        image = np.squeeze(data.cpu().numpy())
        print(image.shape)
        return image


def get_output_layer(model, layer_name=None):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    pdb.set_trace()
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    if not layer_name:
    	all_conv_layers = sorted([name for name in layer_dict.keys() if 'conv2d' in name])
    	layer_name = all_conv_layers[-1]
    	print("Final Conv Layer: %s"%layer_name)

    layer = layer_dict[layer_name]
    return layer


def load_model(model_path):
    model = DANet(lambd=1)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = 1, 48, 4, 4
    print(feature_conv.shape)
    print("Number of channels", nc)
    output_cam = []
    print(weight_softmax.shape)
    for idx in class_idx:
        print(weight_softmax[idx].shape)
        cam = weight_softmax[idx].dot(feature_conv)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def get_conv_op(model,x):
    x = F.relu(model.pool1(model.conv1(x)))
    x = F.relu(model.pool2(model.conv2(x)))
    x = x.view(-1, 48 * 4 * 4)
    return x

def get_label_predictor_op(model, x):
    x_class = F.relu(model.fc1(x))
    x_class = F.dropout(x_class, training=False)
    x_class = F.relu(model.fc2(x_class))
    x_class = model.op(x_class)

    return x_class

def visualize_class_activation_map(model_path, img_input_size, img_path, op_path, layer_name=None, channel_first=False):
    model = load_model(model_path)
    original_img = Image.fromarray(np.uint8(get_image('svhn')))
    
    original_img = np.array(original_img.resize((28,28), Image.ANTIALIAS))
    width, height = original_img.shape

    img = original_img
    if channel_first:
    	#Reshape to the network input shape (3, w, h).
    	img = np.array([np.transpose(np.float32(original_img), (2, 0, 1))])

    params = list(model.label_predictor.parameters())
    # print(params)
    weight_softmax = np.squeeze(params[-1].data.numpy())

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    # pdb.set_trace()
    class_weights = model.state_dict()['label_predictor.op.weight']
    # print(class_weights.shape)
    # final_conv_layer = model.state_dict['feature_extractor.conv2.weight']
    # get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    tensor_img = torch.FloatTensor(img)
    conv_outputs = get_conv_op(model.feature_extractor, tensor_img)
    predictions = get_label_predictor_op(model.label_predictor,model.feature_extractor(tensor_img))
    h_x = F.softmax(predictions, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    CAMs = returnCAM(conv_outputs.detach().cpu().numpy(), weight_softmax, [idx[0]])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
    img = cv2.imread('test.jpg')
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('CAM.jpg', result)
    # conv_outputs = np.squeeze(conv_outputs.detach())
    # print(conv_outputs.shape)
    # #Create the class activation map.
    # if channel_first:
    # 	cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[1:3])
    # else:
    # 	cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])

    # for i, w in enumerate(class_weights[:, 1]):
    #         cam += w * conv_outputs[:, :, i]
    # print("predictions", predictions)
    # cam /= np.max(cam)
    # cam = cv2.resize(cam, (height, width))
    # heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    # heatmap[np.where(cam < 0.2)] = 0
    # img = heatmap*0.5 + original_img
    # cv2.imwrite(op_path, img)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type = str, help = "Path of an image to run the network on")
    parser.add_argument("--image_size", type = tuple, help = "Size of the image the network expects eg: (224,224). Assuming a default value of 3 channels")
    parser.add_argument("--channel_first", type=bool, default=False, help="Specify if the network convolutions expect channel_first")
    parser.add_argument("--layer_name", type=str, default=None, help="Specify last convolution layer name")
    parser.add_argument("--output_path", type = str, default = "heatmap.jpg", help = "Output Image filename")
    parser.add_argument("--model_path", type = str, help = "Path of the trained model")
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    args.dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    return args

if __name__ == '__main__':
    args = get_args()
    visualize_class_activation_map(args.model_path, args.image_size, args.image_path, args.output_path, args.layer_name, args.channel_first)