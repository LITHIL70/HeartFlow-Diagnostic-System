"""
train_promax_tf215_final_v2.py
- Auto pads/trims numeric features to 36 -> reshapes to 6x6x1
- Fast 2-layer 2D CNN (optimized for speed)
- No cv2, no seaborn, TF2.15-compatible
- Saves model, confusion matrix, training curves, metrics CSV + PNG
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)
tf.get_logger().setLevel('ERROR')

# --------- Helpers ----------
def detect_header(path):
    # return True if header looks like names, else False
    sample = pd.read_csv(path, nrows=1)
    return any(str(c).isalpha() for c in sample.columns)

def load_and_prepare(path, target_cells=36, img_h=6, img_w=6):
    header = 0 if detect_header(path) else None
    df = pd.read_csv(path, header=header, low_memory=False)
    print("Loaded shape:", df.shape)

    # detect label column
    for possible in ["type","label","class","beat_type","target"]:
        if possible in df.columns:
            label_col = possible
            break
    else:
        label_col = df.columns[-1]

    print("Detected label column:", label_col)

    # features / labels
    X_df = df.drop(columns=[label_col])
    y_raw = df[label_col].astype(str).values

    # keep numeric columns only
    X_num = X_df.select_dtypes(include=[np.number])
    # if numeric cols are zero, try coercing
    if X_num.shape[1] == 0:
        X_num = X_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

    X_np = X_num.to_numpy(dtype=np.float32)
    n_features = X_np.shape[1]
    print("Numeric features:", n_features)

    # pad or trim to target_cells
    if n_features < target_cells:
        pad = target_cells - n_features
        X_np = np.pad(X_np, ((0,0),(0,pad)), mode='constant', constant_values=0.0)
        print(f"Padded features from {n_features} -> {target_cells}")
    elif n_features > target_cells:
        X_np = X_np[:, :target_cells]
        print(f"Trimmed features from {n_features} -> {target_cells}")

    # reshape to images
    X_img = X_np.reshape(-1, img_h, img_w, 1)
    print("Image tensor shape:", X_img.shape)

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    print("Classes detected:", list(classes))

    return X_img, y, list(X_num.columns)[:target_cells], classes

# --------- Model ----------
def build_fast_cnn(input_shape, n_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --------- Plots & Save functions ----------
def plot_confusion(cm, classes, outpath):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2. else "black")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close()
    print("Saved confusion matrix ->", outpath)

def save_metrics_csv(y_true, y_pred, classes, out_csv):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    acc = (y_true == y_pred).mean()
    rows = []
    for i, c in enumerate(classes):
        rows.append({
            "class": str(c),
            "precision": float(np.round(precision[i],4)),
            "recall": float(np.round(recall[i],4)),
            "f1_score": float(np.round(f1[i],4)),
            "support": int(support[i])
        })
    # overall rows
    rows.append({"class":"accuracy", "precision":"", "recall":"", "f1_score": float(np.round(acc,4)), "support": len(y_true)})
    dfm = pd.DataFrame(rows)
    dfm.to_csv(out_csv, index=False)
    print("Saved metrics CSV ->", out_csv)
    return dfm

def plot_metrics_table(dfm, out_png):
    # produce a simple table image
    fig, ax = plt.subplots(figsize=(8, max(1.5, 0.35*len(dfm))))
    ax.axis('off')
    table = ax.table(cellText=dfm.values, colLabels=dfm.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved metrics image ->", out_png)

# --------- Main ----------
def main():
    # defaults
    data_path = "archive/MIT-BIH Arrhythmia Database.csv"
    out_dir = "promax_results"
    epochs = 12
    batch_size = 128
    test_size = 0.2
    val_size = 0.1

    # load
    X, y, numeric_cols, classes = load_and_prepare(data_path, target_cells=36, img_h=6, img_w=6)

    # splits
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=(test_size+val_size), stratify=y, random_state=42)
    val_rel = val_size / (test_size+val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=42)
    print("Train/Val/Test shapes:", X_train.shape, X_val.shape, X_test.shape)

    os.makedirs(out_dir, exist_ok=True)

    model = build_fast_cnn(input_shape=X_train.shape[1:], n_classes=len(classes))
    model.summary()

    # callbacks
    ckpt = tf.keras.callbacks.ModelCheckpoint(os.path.join(out_dir, "best_model.keras"), monitor='val_loss', save_best_only=True, save_format='keras')
    early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[ckpt, early, reduce_lr],
        verbose=2
    )

    # save final
    model.save(os.path.join(out_dir, "final_model.keras"))
    print("Saved final model ->", os.path.join(out_dir, "final_model.keras"))

    # predict & metrics
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

    # confusion
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(cm, classes, os.path.join(out_dir, "confusion_matrix.png"))

    # training curves
    plt.figure()
    plt.plot(history.history.get('loss',[]), label='train_loss')
    plt.plot(history.history.get('val_loss',[]), label='val_loss')
    plt.plot(history.history.get('accuracy',[]), label='train_acc')
    plt.plot(history.history.get('val_accuracy',[]), label='val_acc')
    plt.legend(); plt.title("Training Curves")
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=200)
    plt.close()
    print("Saved training_curves.png")

    # classification report
    creport = classification_report(y_test, y_pred, target_names=[str(c) for c in classes], zero_division=0)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(creport)
    print("Saved classification_report.txt")

    # metrics CSV + PNG
    dfm = save_metrics_csv(y_test, y_pred, classes, os.path.join(out_dir, "results_table.csv"))
    plot_metrics_table(dfm, os.path.join(out_dir, "results_table.png"))

    print("\nAll outputs in folder:", out_dir)
    print("Done.")

if __name__ == "__main__":
    main()
