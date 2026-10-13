from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# data sets download from https://www.kaggle.com/datasets/tongpython/cat-and-dog
base = "../../dataset/catvdog"
train_dir = Path(base+ "/training_set")
test_dir = Path(base + "/test_set")


IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SIZE = 64
CLASSES = ["cats", "dogs"]
LABEL_OF = {c: i for i, c in enumerate(CLASSES)}


def explain_data_sets(root):
    widths, heights = [],[]
    modes, per_class= Counter(), Counter()
    bad = []
    
    for p in root.rglob("*"):
        if p.suffix.lower() not in IMAGE_EXT:
            continue
        try:
            with Image.open(p) as im:
                w,h = im.size
                widths.append(w)
                heights.append(h)
                modes[im.mode] += 1
            per_class[p.parent.name] += 1
        except Exception as e:
            bad.append((p.name, repr(e)))
    
    widths = np.array(widths)
    heights = np.array(heights)
    ratios = widths/heights

    
    print(f"total readable:{len(widths)}, total unreadable:{len(bad)}")
    print("unreadable:", len(bad), bad[:5])
    print("per_class:", per_class)
    print("modes:", modes)
    stats("widths", widths)
    stats("heights", heights)
    stats("ratios", ratios)
    
    print("distinct sizes", len(set(zip(widths.tolist(), heights.tolist()))))
    print("extreme ratios", int(((ratios < 0.5) | (ratios > 2.0)).sum()))


def stats(name, a):
    print(f"name={name:8},min={a.min():.2f},p10={np.percentile(a,10):.2f} " + 
         f"median={np.median(a):.2f}, max={a.max():.2f}")

def load_data_set(root):
    root = Path(root)
    images, labels = [], []

    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXT:
            # print(f" name:{p.name}")
            continue
        if p.parent.name not in LABEL_OF:
            continue

        with Image.open(p) as im:
            im = im.convert("RGB").resize((SIZE,SIZE), Image.BILINEAR)

        images.append(np.asarray(im, dtype=np.uint8))
        labels.append(LABEL_OF[p.parent.name])

    X = np.stack(images)
    Y = np.array(labels, dtype=np.int64)

    return X, Y
    



def flatten(X):
    return X.reshape(X.shape[0],-1).T / 255


def prepare_data_set():
    X_train_orig_set, Y_train_orig_set = load_data_set(train_dir)
    X_test_orig_set, Y_test_orig_set = load_data_set(test_dir)
    
    # print(X_train_orig_set.shape, Y_train_orig_set.shape)
    # print(X_test_orig_set.shape, Y_test_orig_set.shape)
    
    # Reshuffle the datasets  (since we ordered it previously)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Y_train_orig_set))
    permtest = rng.permutation(len(Y_test_orig_set))
    
    X_train_orig_set, Y_train_orig_set = X_train_orig_set[perm], Y_train_orig_set[perm]
    X_test_orig_set, Y_test_orig_set = X_test_orig_set[permtest], Y_test_orig_set[permtest]

    # Flatten the data sets for the model.
    X_train = flatten(X_train_orig_set)
    X_test = flatten(X_test_orig_set)
    
    Y_train = Y_train_orig_set.reshape(1, -1)
    Y_test = Y_test_orig_set.reshape(1,-1)

    datasets =  {
        "X_train": X_train,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_test": Y_test,
        "X_train_orig_set": X_train_orig_set,
        "Y_train_orig_set": Y_train_orig_set,
        "X_test_orig_set": X_test_orig_set,
        "Y_test_orig_set": Y_test_orig_set
    }
    return datasets


def plot_costs(costs, label, learning_rate=0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel( 'cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title(f"Model:{label}; learning rate = {learning_rate}")
    plt.show()


def get_label (prediction):
    label = np.squeeze(prediction)
    label_str =  CLASSES[label]
    return label_str


# load_image reads a single image from the test-dataset folder and returns both
# the raw pixels (for plt.imshow) and the flattened/normalized column vector
# the model expects.
def load_image(name, base="../../test-dataset/images/"):
    fname = base + name

    with Image.open(fname) as im:
        im = im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)

    image = np.asarray(im, dtype=np.uint8)
    x = image.reshape(1, SIZE * SIZE * 3).T / 255.0

    return image, x
