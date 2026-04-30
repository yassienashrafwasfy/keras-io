"""
Title: 2D Organ Segmentation
Author: [Yassien Wasfy](https://www.linkedin.com/in/yassien-wasfy-315ab5349/)
Date created: 2026/04/29
Last modified: 2026/04/29
Description: Implementing a 2D semantic segmentation for medical imaging (trained for 39 epochs in 40 minutes).
Accelerator: GPU
"""

"""
this example implements trans-unet for 2d multi organ segmentation based on the paper
from based on the paper by Chen et al. (2021).

"""

"""
# Setup
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import ops, layers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import h5py

"""
# Configure the hyperparameters
"""

BATCH_SIZE = 16
SHUFFLE_BUFFER = 200
IMAGE_SIZE = 224
NUM_CLASSES = 9
LEARNING_RATE = 0.0001
NUM_EPOCHS = (
    2  # We use 2 epochs for demonstration; 39 epochs are needed for full convergence.
)
WEIGHT_DECAY = 0.001
PROJECTION_DIM = 256
LAMBDA_CE = 0.5
LAMBDA_DICE = 0.5
NUM_TRANSFORMER_LAYERS = 8
HIDDEN_DIM = 512
SEED = 42
ROTATION_DEGREE = 5
LABEL_COLORS = {
    0: ("Background", (0,   0,   0)), # Background - Black
    1: ("Aorta",      (0,   102, 204)), # Aorta - Blue
    2: ("Gallbladder",(0,   255, 0)), # Gallbladder - Bright Green
    3: ("Left Kidney",(255, 0,   0)),  # Left Kidney - Red
    4: ("Right Kidney",(0,  255, 255)), # Right Kidney - Cyan
    5: ("Liver",      (255, 0,   255)), # Liver - Magenta
    6: ("Pancreas",   (255, 255, 0)), # Pancreas - Yellow
    7: ("Spleen",     (153, 0,   255)), # Spleen - Purple/Violet
    8: ("Stomach",    (255, 128, 0)), # Stomach - Orange
}


"""
# Use data augmentation
"""

# 1. Define ONE single instance for augmentation
joint_augment = keras.Sequential(
    [
        keras.layers.RandomRotation(
            factor=ROTATION_DEGREE / 360.0,
            fill_mode="nearest",
            interpolation="nearest",  # MUST be 'nearest' to protect mask class values
            seed=SEED,
        ),
        keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            fill_mode="nearest",
            interpolation="nearest",
            seed=SEED,
        ),
    ],
    name="joint_augment",
)


def random_augment(image, label):
    """
    Concatenates image and label into a 4-channel tensor,
    applies a single spatial transform, and splits them back.

    image : (224, 224, 3)  float32
    label : (224, 224, 1)  uint8
    """
    label = ops.cast(label, "float32")
    combined = ops.concatenate([image, label], axis=-1)

    augmented = joint_augment(combined, training=True)

    image = augmented[..., :3]  # First 3 channels belong to the image
    label = augmented[..., 3:]  # The 4th channel belongs to the label

    # 5. Cast label back to uint8
    label = ops.cast(label, "uint8")

    return image, label


"""
# Prepare the data
"""

TRAIN_PATH = "/kaggle/input/synapse/Synapse/train_npz"
VAL_PATH = "/kaggle/input/synapse/Synapse/test_vol_h5"


def get_npz_paths(path):
    return sorted(
        [
            os.path.join(path, filename)
            for filename in os.listdir(path)
            if filename.endswith(".npz")
        ]
    )


def get_h5_paths(path):
    return sorted(
        [
            os.path.join(path, filename)
            for filename in os.listdir(path)
            if filename.endswith(".h5") or filename.endswith(".npy")
        ]
    )


# Step 2: Define a loader for one .npz file
def load_npz(npz_path):
    npz_data = np.load(npz_path.numpy().decode("utf-8"))
    return npz_data["image"], npz_data["label"]


def load_h5_slices(h5_path):
    path_string = h5_path.numpy().decode("utf-8")
    with h5py.File(path_string, "r") as h5_file:
        image = h5_file["image"][:]  # (slices, H, W)
        label = h5_file["label"][:]  # (slices, H, W)
    return image, label


def preprocess(npz_path):
    image, label = tf.py_function(
        func=load_npz, inp=[npz_path], Tout=[tf.float32, tf.uint8]
    )
    image.set_shape([None, None])
    label.set_shape([None, None])

    image = ops.image.resize(
        image[..., None], [IMAGE_SIZE, IMAGE_SIZE], interpolation="bilinear"
    )
    label = ops.image.resize(
        label[..., None], [IMAGE_SIZE, IMAGE_SIZE], interpolation="nearest"
    )

    # Grayscale → RGB
    image = ops.repeat(image, 3, axis=-1)  # (224, 224, 3)
    return image, label  # (224, 224, 3), (224, 224, 1)


def preprocess_h5(h5_path):
    image, label = tf.py_function(
        func=load_h5_slices, inp=[h5_path], Tout=[tf.float32, tf.uint8]
    )
    image.set_shape([None, None, None])
    label.set_shape([None, None, None])

    # Expand dims to treat each slice like a single 2D image
    image = ops.expand_dims(image, -1)  # (slices, H, W, 1)
    label = ops.expand_dims(label, -1)  # (slices, H, W, 1)

    image = ops.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], interpolation="bilinear")
    label = ops.image.resize(label, [IMAGE_SIZE, IMAGE_SIZE], interpolation="nearest")

    image = ops.repeat(image, 3, axis=-1)  # (slices, 224, 224, 3)

    return image, label


# ── Filter ─────────────────────────────────────────────────────────────────────


def has_label(image, label):
    return ops.max(label) > 0


# ── Dataset Builder ────────────────────────────────────────────────────────────


def build_dataset(npz_files, augment=False):
    dataset = (
        tf.data.Dataset.from_tensor_slices(npz_files)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .filter(has_label)
        .cache()
        .shuffle(SHUFFLE_BUFFER)
    )
    if augment:
        dataset = dataset.map(random_augment, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_val_dataset(h5_files):
    return (
        tf.data.Dataset.from_tensor_slices(h5_files)
        .map(preprocess_h5, num_parallel_calls=tf.data.AUTOTUNE)
        .flat_map(
            lambda image_vol, label_vol: tf.data.Dataset.from_tensor_slices(
                (image_vol, label_vol)
            )
        )
        .filter(has_label)
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )


RAW_TRAIN_FILES = get_npz_paths(TRAIN_PATH)
RAW_VAL_FILES = get_h5_paths(VAL_PATH)

print(f"Total train slices : {len(RAW_TRAIN_FILES)}")
print(f"Total val volumes  : {len(RAW_VAL_FILES)}")


train_dataset = build_dataset(RAW_TRAIN_FILES, augment=True)
val_dataset = build_val_dataset(RAW_VAL_FILES)


"""
# Visualizing Example
"""

CLASS_NAMES = [name  for name, _     in LABEL_COLORS.values()]
PALETTE     = np.array([color for _, color in LABEL_COLORS.values()], dtype=np.float32)

legend_handles = [
    mpatches.Patch(color=PALETTE[i] / 255, label=f"{i}: {CLASS_NAMES[i]}")
    for i in LABEL_COLORS
]

# Convert single-channel label to RGB color image
def colorize_mask(mask):
    mask_numpy = ops.convert_to_numpy(ops.squeeze(mask)).astype(np.uint8)
    height, width = mask_numpy.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for label_id, (_, color) in LABEL_COLORS.items():
        color_mask[mask_numpy == label_id] = color

    return color_mask


# Show image and color mask
sample_images, sample_masks = next(iter(train_dataset))
print("Image shape:", sample_images.shape)
print("Label shape:", sample_masks.shape)

random_index = np.random.randint(0, BATCH_SIZE - 1)
sample_image = sample_images[random_index].numpy().squeeze()
sample_mask = sample_masks[random_index]
sample_image_grayscale = ops.image.rgb_to_grayscale(sample_image)

colored_mask = colorize_mask(sample_mask)

fig, axes = plt.subplots(1, 2, figsize=(12, 8))

# CT image
axes[0].imshow(sample_image_grayscale, cmap="gray")
axes[0].set_title("CT Slice")
axes[0].axis("off")

# Colored mask
axes[1].imshow(colored_mask)
axes[1].set_title("Colored Segmentation Mask")
axes[1].axis("off")

# Global legend
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=3,
    fontsize=10,
    frameon=False,
)

plt.tight_layout()
plt.show()


"""
# Implementing CNN Hybrid part

Hybrid CNN backbone for TransUNet based on ResNet-50 pretrained on ImageNet(He et al.,
2016).

The backbone is tapped at four intermediate residual block outputs, producing
a feature pyramid at progressively halved spatial resolutions. These feature
maps are stored and later consumed by the `CUP` (Cascaded Upsampling) decoder
as `U-Net` skip connections, allowing fine-grained spatial detail lost during
the `Transformer`'s token processing to be recovered during upsampling.


"""


class ResNet50FeatureExtractor(keras.Model):
    """
    Hybrid CNN backbone for TransUNet based on ResNet-50.

    Input shape:
        `(batch_size, 224, 224, 3)`

    Output shape:
        A list of four tensors in order from finest to coarsest resolution:
        `[(batch_size, 112, 112, 64), (batch_size, 56, 56, 256), 
          (batch_size, 28, 28, 512), (batch_size, 14, 14, 1024)]`

    Example:
        >>> extractor = ResNet50FeatureExtractor()
        >>> sample_input = keras.random.normal((2, 224, 224, 3))
        >>> features = extractor(sample_input)
        >>> [feature_map.shape for feature_map in features]
        [(2, 112, 112, 64), (2, 56, 56, 256), (2, 28, 28, 512), (2, 14, 14, 1024)]
    """

    def __init__(self):
        super().__init__()
        input_shape = (224, 224, 3)
        inputs = keras.Input(shape=input_shape)
        preprocessed_inputs = keras.applications.resnet50.preprocess_input(inputs)
        base_model = keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_tensor=preprocessed_inputs
        )

        self.encoder = keras.Model(
            inputs=inputs,
            outputs=[
                base_model.get_layer("conv1_relu").output,  # 112x112, 64ch
                base_model.get_layer("conv2_block3_out").output,  # 56x56,  256ch
                base_model.get_layer("conv3_block4_out").output,  # 28x28,  512ch
                base_model.get_layer("conv4_block6_out").output,  # 14x14, 1024ch
            ],
            name="resnet50_backbone",
        )

    def call(self, inputs, training=False):
        return self.encoder(inputs, training=training)


"""
# Vit patch encoding


1. TransUNet uses a single learnable Conv2D layer where the kernel size and stride are
both equal to `patch_size`. This is mathematically equivalent to cutting the feature map
into non-overlapping patches andprojecting each one with a shared linear layer.



2. A learnable 1D positional embedding of shape `(1, num_patches, embedding_dim)`is then
added to inject spatial order into the sequence.

"""


@keras.saving.register_keras_serializable(package="Trans-UNET")
class PatchEmbedding(keras.layers.Layer):
    """
    Args:
        patch_size (int): Height and width of each square patch in pixels.
        embedding_dim (int): Dimensionality of the projected patch tokens
            (the ViT hidden size).
num_patches (int): Total number of patches, i.e. `(image_size // patch_size) **
2`.

    Input shape:
        `(batch_size, image_height, image_width, channels)`

    Output shape:
        `(batch_size, num_patches, embedding_dim)`

    Example:
        >>> layer = PatchEmbedding(patch_size=16, embedding_dim=768, num_patches=196)
        >>> x = tf.random.normal((2, 224, 224, 3))
        >>> layer(x).shape
        TensorShape([2, 196, 768])

    References:
        - Dosovitskiy et al., "An Image is Worth 16x16 Words" (2020), arXiv:2010.11929
        - Chen et al., "TransUNet" (2021), arXiv:2102.04306
    """

    def __init__(self, patch_size, embedding_dim, num_patches, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patches = num_patches

        self.projection = keras.layers.Conv2D(
            filters=embedding_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            name="patch_projection",
        )

    def build(self, input_shape):
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(1, self.num_patches, self.embedding_dim),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        projected_patches = self.projection(inputs)
        batch_size = ops.shape(projected_patches)[0]
        # Reshape to (batch_size, num_patches, embedding_dim)
        projected_patches = ops.reshape(
            projected_patches, [batch_size, -1, self.embedding_dim]
        )
        return projected_patches + self.position_embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "embedding_dim": self.embedding_dim,
                "num_patches": self.num_patches,
            }
        )
        return config


"""
# Transformer Block

Each block applies two sub-layers in sequence:

1. **Multi-Head Self-Attention (MHSA):** The token sequence is normalized,
   then each token attends to all others via `keras.layers.MultiHeadAttention`.
   The attention head dimension is set to `embedding_dim // num_heads`.
   The result is added back to the input as a residual.

2. **MLP block:** The output is normalized, then passed through a two-layer
   feed-forward network with `GELU` activation and dropout after each dense
   layer. The result is again added as a residual.
"""


@keras.saving.register_keras_serializable(package="Trans-UNET")
class TransformerBlock(keras.layers.Layer):
    """
    Pre-LN forward pass:
        x → LayerNorm → MHSA → + x  →  LayerNorm → MLP → + x

    Args:
        embedding_dim (int): Dimensionality of the token embeddings, i.e. the
            ViT hidden size. Must be divisible by `num_heads`.
        num_heads (int): Number of parallel attention heads. Each head operates
            on a subspace of size `embedding_dim // num_heads`.
        hidden_dim (int): Inner dimensionality of the MLP block. In the original
            ViT-B/16 configuration this is `3072` (4× the hidden size of `768`).
        dropout_rate (float): Dropout probability applied after each Dense layer
            in the MLP block, and after the attention projection. Defaults to `0.1`.

    Input shape:
        `(batch_size, num_patches, embedding_dim)`

    Output shape:
        `(batch_size, num_patches, embedding_dim)`

    Example:
        >>> block = TransformerBlock(embedding_dim=768, num_heads=12, hidden_dim=3072)
        >>> x = tf.random.normal((2, 196, 768))
        >>> block(x, training=False).shape
        TensorShape([2, 196, 768])

    References:
        - Vaswani et al., "Attention Is All You Need" (2017), arXiv:1706.03762
        - Dosovitskiy et al., "An Image is Worth 16x16 Words" (2020), arXiv:2010.11929
        - Chen et al., "TransUNet" (2021), arXiv:2102.04306
    """

    def __init__(
        self, embedding_dim, num_heads, hidden_dim, dropout_rate=0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.norm1 = keras.layers.LayerNormalization(
            epsilon=1e-6, name="attention_norm"
        )

        self.attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout_rate,
            name="multi_head_attention",
        )

        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name="ffn_norm")

        self.mlp = keras.Sequential(
            [
                keras.layers.Dense(hidden_dim, activation="gelu", name="fc1"),
                keras.layers.Dropout(dropout_rate),
                keras.layers.Dense(embedding_dim, name="fc2"),
                keras.layers.Dropout(dropout_rate),
            ],
            name="mlp",
        )

    def call(self, inputs, training=False):
        # Sub-layer 1: MHSA with pre-norm and residual
        normalized_inputs = self.norm1(inputs)
        attention_output = self.attn(
            normalized_inputs, normalized_inputs, training=training
        )
        hidden_states = inputs + attention_output

        # Sub-layer 2: MLP with pre-norm and residual
        mlp_output = self.mlp(self.norm2(hidden_states), training=training)
        return hidden_states + mlp_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


"""
# Build Trans-UNET

| ![TransUNet Architecture](https://i.postimg.cc/15cppkRJ/transunet-architecture.png) |
| :--: |
| **TransUNet Architecture Overview** |

TransUNet: a hybrid Vision Transformer and U-Net architecture for 2D
multi-organ segmentation (Chen et al., 2021).

The model operates in three sequential stages:

1. **CNN Feature Pyramid (Hybrid Encoder)**

A `ResNet50FeatureExtractor` processes the input image and produces four
feature maps at progressively halved spatial resolutions. The finest three
— at 112×112, 56×56, and 28×28 — are stored as U-Net skip connections.
The coarsest map at 14×14 carries high-level semantic content and is passed
to the Transformer.

2. **Vision Transformer Encoder**

The 14×14 CNN feature map is tokenized via `PatchEmbedding`, which applies
a learnable `Conv2D` with `kernel_size == stride == patch_size`, projecting
each spatial position into a `embedding_dim`-dimensional token. Learned
positional embeddings are added, and the sequence is processed by a stack
of `TransformerBlock` layers. After the final `LayerNormalization`, the
token sequence is reshaped back into a 2D spatial map of size
`(14 // patch_size, 14 // patch_size)`.

3. **CUP Decoder (Cascaded Upsampling)**

The spatial map from the Transformer is progressively upsampled with
bilinear interpolation. At each scale, the corresponding CNN skip connection
is concatenated along the channel axis before a `Conv2D + GroupNormalization`
block refines the features. After three skip-connected upsampling steps
(14→28→56→112), a final upsample brings the map to the full input resolution
(112→224), and a 1×1 `Conv2D` with `softmax` produces the per-pixel class
probability map.

"""


@keras.saving.register_keras_serializable(package="Trans-UNET")
class TransUNet(keras.Model):
    """
    Args:
        image_size (tuple[int, int, int]): Spatial dimensions of the input image
            as `(height, width, channels)`. Both height and width must be 224
            to match the `ResNet50FeatureExtractor` output grid. Defaults to
            `(224, 224, 3)`.
        patch_size (int): Size of each square patch extracted from the 14×14
            CNN feature map. A value of `1` treats each spatial position as an
            independent token, yielding `196` tokens. Defaults to `1`.
        embedding_dim (int): Dimensionality of the projected patch tokens and the
            Transformer hidden size. Must be divisible by `num_heads`.
            Defaults to `64`.
        num_heads (int): Number of parallel attention heads in each
            `TransformerBlock`. Defaults to `4`.
        n_layers (int): Number of stacked `TransformerBlock` layers. Defaults to `6`.
        hidden_dim (int): Inner dimensionality of the MLP block inside each
            `TransformerBlock`. Defaults to `512`.
        dropout_rate (float): Dropout probability applied inside each
            `TransformerBlock`. Defaults to `0.1`.
        num_classes (int): Number of segmentation classes. Sets the number of
            output channels in the final `Conv2D`. Defaults to `9`.

    Input shape:
        `(batch_size, 224, 224, 3)`

    Output shape:
        `(batch_size, 224, 224, num_classes)`

    Example:
        >>> model = TransUNet(num_classes=9)
        >>> x = ops.random.normal((2, 224, 224, 3))
        >>> model(x, training=False).shape
        TensorShape([2, 224, 224, 9])

    References:
        - Chen et al., "TransUNet" (2021), arXiv:2102.04306
- He et al., "Deep Residual Learning for Image Recognition" (2016),
arXiv:1512.03385
        - Dosovitskiy et al., "An Image is Worth 16x16 Words" (2020), arXiv:2010.11929
    """

    def __init__(
        self,
        image_size=(224, 224, 3),
        patch_size=1,
        embedding_dim=64,
        num_heads=4,
        n_layers=6,
        hidden_dim=512,
        dropout_rate=0.1,
        num_classes=9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        feature_map_size = image_size[0] // 16
        num_patches = (feature_map_size // patch_size) ** 2

        self.cnn_backbone = ResNet50FeatureExtractor()
        self.patch_embedding = PatchEmbedding(patch_size, embedding_dim, num_patches)
        self.transformer_blocks = [
            TransformerBlock(embedding_dim, num_heads, hidden_dim, dropout_rate)
            for _ in range(n_layers)
        ]
        self.encoder_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="encoder_norm"
        )

        # ===== Decoder block 1 ==============
        self.upsample1 = keras.layers.UpSampling2D(
            size=2, interpolation="bilinear", name="upsample1"
        )
        self.concat1 = keras.layers.Concatenate(name="concat_skip3")
        self.conv1 = keras.layers.Conv2D(
            512, 3, padding="same", use_bias=False, name="decoder_conv1"
        )  # no activation
        self.gnorm1 = keras.layers.GroupNormalization(groups=32, name="gnorm1")
        self.act1 = keras.layers.Activation("relu", name="act1")

        # ===== Decoder block 2 ==============
        self.upsample2 = keras.layers.UpSampling2D(
            size=2, interpolation="bilinear", name="upsample2"
        )
        self.concat2 = keras.layers.Concatenate(name="concat_skip2")
        self.conv2 = keras.layers.Conv2D(
            256, 3, padding="same", use_bias=False, name="decoder_conv2"
        )
        self.gnorm2 = keras.layers.GroupNormalization(groups=32, name="gnorm2")
        self.act2 = keras.layers.Activation("relu", name="act2")

        # ===== Decoder block 3 ==============
        self.upsample3 = keras.layers.UpSampling2D(
            size=2, interpolation="bilinear", name="upsample3"
        )
        self.concat3 = keras.layers.Concatenate(name="concat_skip1")
        self.conv3 = keras.layers.Conv2D(
            128, 3, padding="same", use_bias=False, name="decoder_conv3"
        )
        self.gnorm3 = keras.layers.GroupNormalization(groups=32, name="gnorm3")
        self.act3 = keras.layers.Activation("relu", name="act3")

        # ===== Decoder block 4 ===============
        self.upsample4 = keras.layers.UpSampling2D(
            size=2, interpolation="bilinear", name="upsample4"
        )
        self.conv4 = keras.layers.Conv2D(
            64, 3, padding="same", use_bias=False, name="decoder_conv4"
        )
        self.gnorm4 = keras.layers.GroupNormalization(groups=32, name="gnorm4")
        self.act4 = keras.layers.Activation("relu", name="act4")

        self.segmentation_head = keras.layers.Conv2D(
            num_classes,
            1,
            padding="same",
            activation="softmax",
            name="segmentation_head",
        )

    def call(self, inputs, training=False):
        # ===== Encoder ===============
        skip1, skip2, skip3, cnn_features = self.cnn_backbone(
            inputs, training=training
        )  

        tokens = self.patch_embedding(cnn_features)
        for block in self.transformer_blocks:
            tokens = block(tokens, training=training)
        tokens = self.encoder_norm(tokens)

        feature_map_size = self.image_size[0] // 16 // self.patch_size
        spatial_features = ops.reshape(
            tokens, (-1, feature_map_size, feature_map_size, self.embedding_dim)
        )

        # ===== Decoder: Conv → GroupNorm → ReLU ===============
        decoded = self.upsample1(spatial_features)
        decoded = self.concat1([decoded, skip3])
        decoded = self.act1(self.gnorm1(self.conv1(decoded), training=training))

        decoded = self.upsample2(decoded)
        decoded = self.concat2([decoded, skip2])
        decoded = self.act2(self.gnorm2(self.conv2(decoded), training=training))

        decoded = self.upsample3(decoded)
        decoded = self.concat3([decoded, skip1])
        decoded = self.act3(self.gnorm3(self.conv3(decoded), training=training))

        decoded = self.upsample4(decoded)
        decoded = self.act4(self.gnorm4(self.conv4(decoded), training=training))

        return self.segmentation_head(decoded)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "n_layers": self.n_layers,
                "hidden_dim": self.hidden_dim,
                "dropout_rate": self.dropout_rate,
                "num_classes": self.num_classes,
            }
        )
        return config


"""
# Dice Loss for Multi-Class Segmentation

## Formula

| ![Dice Formula](https://i.postimg.cc/y8D8VR5q/dice-equation.png) |
| :--: |
| **Dice Loss Calculation** |

Where `C` = number of classes, `p` = predicted probability, `g` = ground truth, `ε` =
smooth term.

## Terminology

1. **Intersection**: It measures how much the prediction overlaps with the ground truth
per class.

2. **Union**: This measures the total presence of each class in both the prediction and
the ground truth.



# MeanIoU Wrapper for Multi-Organ Segmentation


## Formula

| ![mIoU Formula](https://i.postimg.cc/3xkxKGqs/miou-equation.png) |
| :--: |
| **Mean Intersection-over-Union (mIoU) Calculation** |

Where `C` = number of classes, `TP` = true positive pixels, `FP` = false positive pixels,
`FN` = false negative pixels.


## What It Measures

mIoU measures **how much the predicted segmentation mask overlaps with the ground truth
mask** for each organ independently, then averages across all 9 classes.

- A score of `1.0` means perfect overlap — every pixel of every organ was predicted
exactly right.
- A score of `0.0` means zero overlap — the model predicted completely wrong regions for
every organ.
- In practice a well-trained model on Synapse sits around `0.70 – 0.80`.
"""


@keras.saving.register_keras_serializable(package="Trans-UNET")
def combined_loss(y_true, y_pred, smooth=1e-6):
    y_true = ops.squeeze(y_true, -1)

    ce = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    ce = ops.mean(ce)

    y_true_oh = ops.one_hot(
        ops.cast(y_true, "int32"), num_classes=ops.shape(y_pred)[-1]
    )
    y_pred = ops.clip(y_pred, 1e-6, 1.0)

    intersection = ops.sum(y_true_oh * y_pred, axis=[1, 2])
    union = ops.sum(y_true_oh + y_pred, axis=[1, 2])
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - ops.mean(dice)

    return LAMBDA_CE * ce + LAMBDA_DICE * dice_loss


@keras.saving.register_keras_serializable(package="Trans-UNET")
class MeanIoUWrapper(keras.metrics.MeanIoU):
    def __init__(self, num_classes, name="mean_io_u_wrapper", **kwargs):
        kwargs.pop("ignore_class", None)
        kwargs.pop("sparse_y_true", None)
        kwargs.pop("sparse_y_pred", None)

        super().__init__(
            num_classes=num_classes,
            sparse_y_true=True,
            sparse_y_pred=False,
            ignore_class=None,
            name=name,
            **kwargs,
        )
        self._num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.squeeze(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        return {"num_classes": self._num_classes, "name": self.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


"""
# Building and Compiling Model
"""

model = TransUNet(
    image_size=(IMAGE_SIZE, IMAGE_SIZE, 3),
    embedding_dim=PROJECTION_DIM,
    num_heads=4,
    n_layers=NUM_TRANSFORMER_LAYERS,
    hidden_dim=HIDDEN_DIM,
    dropout_rate=0.1,
    num_classes=NUM_CLASSES,
)

model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    ),
    loss=combined_loss,
    metrics=[
        MeanIoUWrapper(num_classes=NUM_CLASSES),
    ],
)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    ),
]

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=NUM_EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

"""
This example demonstrates only two epochs, but for convergence, we trained for 39 epochs (40 minutes).
The training history is shown below:

| ![Loss Curves](https://i.postimg.cc/SxCftnry/training-history.png) |
| :--: |
| **Training History: Loss & Metrics** |
"""

"""
# Loss curves
"""

# Monitor training
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["mean_io_u_wrapper"], label="Training IoU")
plt.plot(history.history["val_mean_io_u_wrapper"], label="Validation IoU")
plt.title("Model IoU")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
plt.show()


"""
# Inference
"""


def _resize_batch(arr, size, interpolation):
    return (
        ops.image.resize(
            arr[..., np.newaxis], (size, size), interpolation=interpolation
        )
        .numpy()
        .squeeze(-1)
    )  # (N, size, size)


def _pick_nonempty(label_vol, n_slices):
    occupied = np.where(label_vol.any(axis=(1, 2)))[0]  
    if len(occupied) == 0:
        return None
    return occupied[
        np.linspace(0, len(occupied) - 1, min(n_slices, len(occupied))).astype(int)
    ]


def _plot_rows(
    axes,
    rows,
    img_batch,
    gt_batch,
    pred_batch,
    colorize_fn,
    row_offset=0,
    case_label="",
):
    for j in range(rows):
        r = row_offset + j
        img = img_batch[j]

        axes[r, 0].imshow(img, cmap="gray")
        axes[r, 0].set_title(f"{case_label} | slice — Input", fontsize=9)
        axes[r, 0].axis("off")

        axes[r, 1].imshow(img, cmap="gray")
        axes[r, 1].imshow(colorize_fn(gt_batch[j]), alpha=0.5)
        axes[r, 1].set_title(f"{case_label} | slice — Ground Truth", fontsize=9)
        axes[r, 1].axis("off")

        axes[r, 2].imshow(img, cmap="gray")
        axes[r, 2].imshow(colorize_fn(pred_batch[j]), alpha=0.5)
        axes[r, 2].set_title(f"{case_label} | slice — Prediction", fontsize=9)
        axes[r, 2].axis("off")


def infer_volume(model, image_vol, batch_size=16):
    """
    Run inference on a full 3D volume.

    Args:
        image_vol : (D, H, W) numpy array — grayscale CT volume
        batch_size: slices to predict at once

    Returns:
        pred_vol  : (D, H, W) numpy array — argmax class per pixel
    """
    D = image_vol.shape[0]
    pred_vol = np.zeros((D, IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)

    for start in range(0, D, batch_size):
        end   = min(start + batch_size, D)
        batch = image_vol[start:end]                        # (K, H, W)

        # Resize + grayscale → RGB
        imgs = _resize_batch(batch.astype(np.float32), IMAGE_SIZE, "bilinear")
        imgs = np.stack([imgs] * 3, axis=-1)               # (K, S, S, 3)

        preds = model.predict(imgs, verbose=0)              # (K, S, S, C)
        pred_vol[start:end] = np.argmax(preds, axis=-1)    # (K, S, S)

    return pred_vol

class VolumeDataset:
    def __init__(self, h5_paths):
        self.paths = h5_paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        with h5py.File(path, "r") as f:
            image = f["image"][:].astype(np.float32)
            label = f["label"][:].astype(np.int32)
        name = os.path.splitext(os.path.basename(path))[0]
        return image, label, name

test_ds = VolumeDataset(RAW_VAL_FILES)

def visualise_volume(vol_idx=0, n_slices=5):
    image_vol, label_vol, name = test_ds[vol_idx]
    pred_vol = infer_volume(model, image_vol)  

    idxs = _pick_nonempty(label_vol, n_slices)
    if idxs is None:
        print(f"[skip] {name}: no annotated slices.")
        return

    imgs = image_vol[idxs] 
    gts = label_vol[idxs]
    preds = pred_vol[idxs]

    n = len(idxs)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n), squeeze=False)
    fig.suptitle(f"Volume: {name}", fontsize=14, fontweight="bold")

    _plot_rows(axes, n, imgs, gts, preds, colorize_mask, case_label=f"slice")

    # update titles with actual slice indices
    for row, si in enumerate(idxs):
        axes[row, 0].set_title(f"CT slice {si}", fontsize=9)
        axes[row, 1].set_title(f"Ground Truth (slice {si})", fontsize=9)
        axes[row, 2].set_title(f"Prediction (slice {si})", fontsize=9)

    patches = [
        mpatches.Patch(color=PALETTE[i] / 255, label=CLASS_NAMES[i])
        for i in range(1, NUM_CLASSES)
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(f"/kaggle/working/viz_{name}.png", dpi=120, bbox_inches="tight")
    plt.show()



def run_inference(model, h5_paths, num_samples=5, n_slices=5):

    cases = []
    for path in h5_paths[:num_samples]:
        with h5py.File(path, "r") as f:
            image = f["image"][:]  
            label = f["label"][:]  

        idxs = _pick_nonempty(label, n_slices)
        if idxs is None:
            print(f"[skip] {path}: no annotated slices.")
            continue

        imgs = _resize_batch(image[idxs].astype(np.float32), IMAGE_SIZE, "bilinear")
        lbls = _resize_batch(label[idxs], IMAGE_SIZE, "nearest").astype(np.int32)

        cases.append({"path": path, "idxs": idxs, "imgs": imgs, "lbls": lbls})

    if not cases:
        print("No annotated slices found.")
        return

    for case in cases:
        imgs_3ch = np.stack([case["imgs"]] * 3, axis=-1)  
        raw_preds = model.predict(imgs_3ch, verbose=0)  
        case["preds"] = np.argmax(raw_preds, axis=-1) 

    total_rows = sum(len(c["idxs"]) for c in cases)
    fig, axes = plt.subplots(total_rows, 3, figsize=(12, total_rows * 4), squeeze=False)

    row = 0
    for ci, case in enumerate(cases):
        k = len(case["idxs"])
        _plot_rows(
            axes,
            k,
            case["imgs"],
            case["lbls"],
            case["preds"],
            colorize_mask,
            row_offset=row,
            case_label=f"Case {ci+1}",
        )
        
        
        for j, si in enumerate(case["idxs"]):
            r = row + j
            axes[r, 0].set_title(f"Case {ci+1} | slice {si} — Input", fontsize=9)
            axes[r, 1].set_title(f"Case {ci+1} | slice {si} — Ground Truth", fontsize=9)
            axes[r, 2].set_title(f"Case {ci+1} | slice {si} — Prediction", fontsize=9)

        row += k

    plt.tight_layout()
    plt.savefig("Inference.png", dpi=150, bbox_inches="tight")
    plt.show()


run_inference(model, RAW_VAL_FILES, num_samples=6, n_slices=5)

for vi in range(min(3, len(test_ds))):
    visualise_volume(vol_idx=vi, n_slices=5)

"""
The following image shows the results of the model on the validation set after 39 epochs:

| ![Inference Results](https://i.postimg.cc/MGwmkzPc/Inference.png) |
| :--: |
| **TransUNet Inference Results** |
"""
