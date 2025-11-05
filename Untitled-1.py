"""
train_promax_tf215.py
Pro-Max 2D-CNN pipeline (Balanced mode): TF 2.15 compatible.
Features:
 - Auto header detection & label detection
 - 17x11 reshape for ~187 features (pads if needed)
 - Class weights for imbalance
 - ModelCheckpoint (.keras), EarlyStopping, ReduceLROnPlateau, TensorBoard
 - Save metrics CSV, confusion matrix, training curves
 - Save feature maps and Grad-CAM for a few test samples
Usage:
  python train_promax_tf215.py --data_path "archive\MIT-BIH Arrhythmia Database.csv"
"""
import os
import math
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K

# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False

def read_csv_auto(path):
    """Try header=0 first; if first row looks numeric-colnames, fall back to header=None."""
    try:
        df = pd.read_csv(path, header=0, low_memory=False)
        cols = list(df.columns)
        # if columns are numeric-like (0,1,2..) re-read without header
        if all(is_number(c) for c in cols):
            df = pd.read_csv(path, header=None, low_memory=False)
            df.columns = [f"c{i}" for i in range(df.shape[1])]
        return df
    except Exception:
        df = pd.read_csv(path, header=None, low_memory=False)
        df.columns = [f"c{i}" for i in range(df.shape[1])]
        return df

def detect_label_col(df):
    candidates = ['type','label','class','beat_type','target']
    for c in candidates:
        if c in df.columns:
            return c
    # last non-numeric column
    for col in reversed(df.columns.tolist()):
        if not pd.api.types.is_numeric_dtype(df[col]):
            return col
    # fallback: last column
    return df.columns[-1]

def choose_image_shape(n_features):
    # Prefer 17x11 for ~187 features
    if n_features <= 187:
        return (17, 11)
    # else pick near-rectangle with minimal padding
    best = None
    for h in range(8, int(math.ceil(math.sqrt(n_features))) + 8):
        w = math.ceil(n_features / h)
        pad = h*w - n_features
        if best is None or pad < best[0]:
            best = (pad, h, w)
    _, h, w = best
    return (h, w)

# ---------------------------
# Model & Grad-CAM helpers
# ---------------------------
def build_promax_2d_cnn(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same', name='conv1')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same', name='conv3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Compute Grad-CAM heatmap (H,W) for a single image array (batch size 1)."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-9)
    return heatmap.numpy()

# ---------------------------
# Plot helpers
# ---------------------------
def plot_confusion(cm, classes, outpath):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], 'd'), horizontalalignment="center",
                 color="white" if cm[i,j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# ---------------------------
# Main pipeline
# ---------------------------
def main(args):
    data_path = args.data_path
    out_dir = args.out_dir
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "feature_maps"))
    ensure_dir(os.path.join(out_dir, "gradcam"))
    ensure_dir(os.path.join(out_dir, "tensorboard_logs"))

    print("Reading CSV (auto-detect header)...")
    df = read_csv_auto(data_path)
    print("Loaded shape:", df.shape)

    label_col = detect_label_col(df)
    print("Detected label column:", label_col)

    # remove metadata columns except label
    for c in ['record','rec','time','timestamp','index','beat_index']:
        if c in df.columns and c != label_col:
            df = df.drop(columns=[c])

    if label_col not in df.columns:
        raise ValueError("Could not find label column automatically. Edit script or pass label_col manually.")

    # labels and features
    y_raw = df[label_col].astype(str).values
    X_df = df.drop(columns=[label_col])
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        X_df = X_df.apply(pd.to_numeric, errors='coerce')
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    X = X_df[numeric_cols].astype(np.float32).values
    print(f"Numeric features used: {X.shape[1]}")

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    print("Classes detected:", classes)
    num_classes = len(classes)
    print("Number of classes:", num_classes)

    # class weights
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = {i: float(w) for i,w in enumerate(cw)}
    print("Class weights:", class_weights)

    # scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # choose reshape and pad
    img_h, img_w = choose_image_shape(X_scaled.shape[1])
    needed = img_h * img_w
    if X_scaled.shape[1] < needed:
        pad = needed - X_scaled.shape[1]
        X_scaled = np.pad(X_scaled, ((0,0),(0,pad)), mode='constant', constant_values=0.0)
    X_imgs = X_scaled.reshape((-1, img_h, img_w, 1))
    print("Image tensor shape:", X_imgs.shape)

    # splits
    X_train, X_tmp, y_train, y_tmp = train_test_split(X_imgs, y, test_size=args.test_size+args.val_size, stratify=y, random_state=42)
    val_rel = args.val_size / (args.test_size + args.val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=42)
    print("Train/Val/Test shapes:", X_train.shape, X_val.shape, X_test.shape)

    # model
    model = build_promax_2d_cnn(input_shape=X_train.shape[1:], num_classes=num_classes)
    model.summary()

    # callbacks
    ckpt_path = os.path.join(out_dir, "best_model.keras")
    cb_checkpoint = callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, save_format='keras')
    cb_early = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    cb_reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
    tb_logdir = os.path.join(out_dir, "tensorboard_logs")
    cb_tb = callbacks.TensorBoard(log_dir=tb_logdir, histogram_freq=1)

    # fit
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=[cb_checkpoint, cb_early, cb_reduce, cb_tb],
        verbose=2
    )

    # save final model (also save weights in keras format)
    final_model_path = os.path.join(out_dir, "final_model.keras")
    model.save(final_model_path, save_format='keras')
    print("Saved final model to", final_model_path)

    # Evaluate
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)

    # classification report & CSV
    report = classification_report(y_test, y_pred, target_names=[str(c) for c in classes], zero_division=0)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    print("\nClassification report:\n", report)

    rows = []
    for i, cname in enumerate(classes):
        rows.append({"class": str(cname), "precision": float(np.round(precision[i],4)), "recall": float(np.round(recall[i],4)), "f1_score": float(np.round(f1[i],4)), "support": int(support[i])})
    rows.append({"class":"accuracy", "precision":"", "recall":"", "f1_score": float(np.round(acc,4)), "support": len(y_test)})
    rows.append({"class":"macro_avg", "precision": float(np.round(p_macro,4)), "recall": float(np.round(r_macro,4)), "f1_score": float(np.round(f1_macro,4)), "support": len(y_test)})
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(os.path.join(out_dir, "results_table.csv"), index=False)
    print("Saved metrics to", os.path.join(out_dir, "results_table.csv"))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(cm, classes, os.path.join(out_dir, "confusion_matrix.png"))
    print("Saved confusion matrix to", os.path.join(out_dir, "confusion_matrix.png"))

    # training curves
    plt.figure()
    plt.plot(history.history.get('loss',[]), label='train_loss')
    plt.plot(history.history.get('val_loss',[]), label='val_loss')
    plt.plot(history.history.get('accuracy',[]), label='train_acc')
    plt.plot(history.history.get('val_accuracy',[]), label='val_acc')
    plt.legend()
    plt.title("Training curves")
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=200)
    plt.close()
    print("Saved training_curves.png")

    # Feature maps (pick up to maps_samples random test indices)
    n_samples = min(args.maps_samples, X_test.shape[0])
    sample_idxs = np.random.choice(range(X_test.shape[0]), size=n_samples, replace=False)
    conv_layer_names = [l.name for l in model.layers if isinstance(l, layers.Conv2D)]
    print("Conv layers:", conv_layer_names)

    for idx in sample_idxs:
        img = X_test[idx:idx+1]
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        base_name = f"sample_{idx}_true-{classes[true_label]}_pred-{classes[pred_label]}"
        # save input map
        inp = img.squeeze()
        plt.imsave(os.path.join(out_dir, "feature_maps", base_name + "_input.png"), inp, cmap='viridis')

        # activations per conv layer
        for lname in conv_layer_names:
            act_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.get_layer(lname).output)
            acts = act_model.predict(img)[0]  # H,W,C
            n_maps = acts.shape[-1]
            cols = min(8, n_maps)
            rows = math.ceil(n_maps/cols)
            fig, axs = plt.subplots(rows, cols, figsize=(cols*1.2, rows*1.2))
            axs = np.array(axs).reshape(-1)
            for i in range(rows*cols):
                ax = axs[i]
                if i < n_maps:
                    fmap = acts[..., i]
                    ax.imshow(fmap, cmap='viridis')
                    ax.axis('off')
                else:
                    ax.axis('off')
            plt.suptitle(f"{base_name} - {lname}")
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            fname = os.path.join(out_dir, "feature_maps", f"{base_name}_{lname}.png")
            fig.savefig(fname, dpi=150)
            plt.close(fig)

    print(f"Saved feature maps for {n_samples} samples under {os.path.join(out_dir,'feature_maps')}")

    # Grad-CAM using last conv layer if exists
    last_conv = conv_layer_names[-1] if conv_layer_names else None
    if last_conv:
        for idx in sample_idxs:
            img = X_test[idx:idx+1]
            base_name = f"sample_{idx}"
            heatmap = make_gradcam_heatmap(img, model, last_conv)
            # normalize heatmap to 0..1
            hmin, hmax = heatmap.min(), heatmap.max()
            heatmap_norm = (heatmap - hmin) / (hmax - hmin + 1e-9)
            # map to color and overlay on input
            cmap = plt.get_cmap('jet')
            heatmap_color = cmap(heatmap_norm)
            img_gray = X_test[idx].squeeze()
            img_norm = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-9)
            blended = 0.6 * heatmap_color[..., :3] + 0.4 * np.stack([img_norm]*3, axis=-1)
            plt.imsave(os.path.join(out_dir, "gradcam", base_name + "_gradcam.png"), blended)
        print(f"Saved Grad-CAMs for {n_samples} samples under {os.path.join(out_dir,'gradcam')}")
    else:
        print("No conv layers found; skipping Grad-CAM.")

    # Save meta info
    meta = {
        "classes": [str(x) for x in classes],
        "image_shape": [img_h, img_w, 1],
        "n_features_used": len(numeric_cols),
        "feature_columns_sample": numeric_cols[:100]
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved meta.json")

    print("\nPro-Max run completed. Results in:", out_dir)

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=r"archive\MIT-BIH Arrhythmia Database.csv", help="Path to CSV")
    parser.add_argument("--out_dir", type=str, default="promax_results", help="Output folder")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs (balanced)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--maps_samples", type=int, default=5, help="How many test samples to save feature maps for")
    args = parser.parse_args()
    main(args)
