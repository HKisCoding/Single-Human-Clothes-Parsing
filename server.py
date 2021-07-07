from flask import Flask,render_template
from flask_cors import CORS, cross_origin
from flask import request
from PIL import Image

from model import *

UPLOAD_FOLDER = 'INPUT_PATH'
RESULT_FOLDER = os.path.join('static', 'result')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'




            # parsing_result_path = os.path.join('static/result', img_name[:-4] + '.png')
            # #parsing_result_path = os.path.join(args.output_dir, 'result' + '.png')
            # output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            # output_img.putpalette(palette)
            # output_img.save(parsing_result_path)
            # if args.logits:
            #     logits_result_path = os.path.join(args.output_dir, img_name[:-4] + '.npy')
            #     np.save(logits_result_path, logits_result)
def color_image(img, pred, file):

    fig, axes = plt.subplots(1, 2)
    ax0, ax1 = axes
    ax0.get_xaxis().set_ticks([])
    ax0.get_yaxis().set_ticks([])
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])

    classes = np.array(('Background',  # always index 0
                        'Hat', 'Hair', 'Glove', 'Sunglasses',
                        'UpperClothes', 'Dress', 'Coat', 'Socks',
                        'Pants', 'Jumpsuits', 'Scarf', 'Skirt',
                        'Face', 'Left-arm', 'Right-arm', 'Left-leg',
                        'Right-leg', 'Left-shoe', 'Right-shoe',))
    colormap = [(0, 0, 0),
                (1, 0.25, 0), (0, 0.25, 0), (0.5, 0, 0.25), (1, 1, 1),
                (1, 0.75, 0), (0, 0, 0.5), (0.5, 0.25, 0), (0.75, 0, 0.25),
                (1, 0, 0.25), (0, 0.5, 0), (0.5, 0.5, 0), (0.25, 0, 0.5),
                (1, 0, 0.75), (0, 0.5, 0.5), (0, 0.5, 0.5), (1, 0, 0),
                (1, 0, 0), (0, 0.75, 0), (0, 0.75, 0), ]
    cmap = matplotlib.colors.ListedColormap(colormap)
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    h, w, _ = pred.shape

    def denormalize(img, mean, std):
        c, _, _ = img.shape
        for idx in range(c):
            img[idx, :, :] = img[idx, :, :] * std[idx] + mean[idx]
        return img

    img = denormalize(img.cpu().numpy(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = img.transpose(1, 2, 0).reshape((h, w, 3))
    pred = pred.reshape((h, w))

    # show image
    ax0.set_title('img')
    ax0.imshow(img)
    ax1.set_title('pred')
    mappable = ax1.imshow(pred, cmap=cmap, norm=norm)
    # colorbar legend
    cbar = plt.colorbar(mappable, ax=axes, shrink=1, )
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(classes):
        cbar.ax.text(30, (20*j + 10) / 20.0, lab, ha='left', va='center', )

    
    plt.savefig(fname=os.path.join(app.config['RESULT_FOLDER'], file.filename))
    #plt.savefig(fname=os.path.join(app.config['RESULT_FOLDER'], 'result.jpg'))
    #print('result saved to ./result.jpg')
    #plt.show()

def color_image_new(img, pred, file, bounds):
    fig, axes = plt.subplots(1, 2)
    ax0, ax1 = axes
    ax0.get_xaxis().set_ticks([])
    ax0.get_yaxis().set_ticks([])
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])

    classes = ['Background',  # always index 0
                        'Hat', 'Hair', 'Glove', 'Sunglasses',
                        'UpperClothes', 'Dress', 'Coat', 'Socks',
                        'Pants', 'Jumpsuits', 'Scarf', 'Skirt',
                        'Face', 'Left-arm', 'Right-arm', 'Left-leg',
                        'Right-leg', 'Left-shoe', 'Right-shoe']
    pred_cls = np.array(([classes[i] for i in np.unique(pred)]))
    colormap = [(0, 0, 0),
                (1, 0.25, 0), (0, 0.25, 0), (0.5, 0, 0.25), (1, 1, 1),
                (1, 0.75, 0), (0, 0, 0.5), (0.5, 0.25, 0), (0.75, 0, 0.25),
                (1, 0, 0.25), (0, 0.5, 0), (0.5, 0.5, 0), (0.25, 0, 0.5),
                (1, 0, 0.75), (0, 0.5, 0.5), (0, 0.5, 0.5), (1, 0, 0),
                (1, 0, 0), (0, 0.75, 0), (0, 0.75, 0), ]
    #cmap = matplotlib.colors.ListedColormap(colormap)
    pred_colormap = [colormap[i] for i in np.unique(pred)]
    cmap = matplotlib.colors.ListedColormap(pred_colormap)
    #bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    h, w, _ = pred.shape

    def denormalize(img, mean, std):
        c, _, _ = img.shape
        for idx in range(c):
            img[idx, :, :] = img[idx, :, :] * std[idx] + mean[idx]
        return img

    img = denormalize(img.cpu().numpy(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = img.transpose(1, 2, 0).reshape((h, w, 3))
    pred = pred.reshape((h, w))
    # show image
    ax0.set_title('img')
    ax0.imshow(img)
    ax1.set_title('pred')
    mappable = ax1.imshow(pred, cmap=cmap, norm=norm)
    # colorbar legend
    cbar = plt.colorbar(mappable, ax=axes, shrink=1, )
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(pred_cls):
        cbar.ax.text(30, (len(pred_cls)*j*1.8 +15 ) / len(pred_cls), lab, ha='left', va='center', )

    plt.savefig(fname=os.path.join(app.config['RESULT_FOLDER'], file.filename))



def get_transform():
    transform_image_list = [
        transforms.Resize((256, 256), 3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(transform_image_list)

def get_gt_transform():
    transform_gt_list = [
        transforms.Resize((256, 256), 0),
        transforms.Lambda(lambda img: np.asarray(img, dtype=np.uint8)),
    ]
    return transforms.Compose(transform_gt_list)

def build_network(snapshot, models):
    epoch = 0
    net = models
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        if not epoch == 'last':
            epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot,map_location={'cuda:0': 'cpu'}))
        #logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch

model = PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024)
snapshot = 'model/PSPNet_last'
net, starting_epoch = build_network(snapshot, model)
net.eval()



@app.route('/')
@cross_origin(origin='*')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'fileUpload' not in request.files:
            return render_template('404.html')
        file = request.files['fileUpload']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        if path == 'INPUT_PATH\\':
            return render_template('404.html')
        file.save(path) #save image to path
        
        # ------------ load image ------------ #
        data_transform = get_transform()
        img = Image.open(path)
        img = data_transform(img)
        img = img.cuda()

        # --------------- inference --------------- #

        with torch.no_grad():
            pred, _ = net(img.unsqueeze(dim=0))

            pred = pred.squeeze(dim=0)
            pred = pred.cpu().numpy().transpose(1, 2, 0)
            pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8).reshape((256, 256, 1))
            bounds = np.unique(pred)
            new_bounds=np.append(bounds,bounds[-1])
            color_image(img, pred, file)
            #cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], file.filename), pred)
            #show_image(img, pred)

        #return render_template('result.html', imagepathresult = os.path.join(app.config['RESULT_FOLDER'], file.filename))
        return render_template('result.html', imageresult = file.filename)
    return render_template('index.html')

# Start Backend
if __name__ == '__main__':
    app.run()