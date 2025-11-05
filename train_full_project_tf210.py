"""
train_full_project_tf210.py
Simple 2D-CNN demo (auto-detect header/label). Saves model, metrics CSV and confusion matrix.
Usage:
  python train_full_project_tf210.py --data_path "archive\MIT-BIH Arrhythmia Database.csv"
"""
import os, math, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def ensure_dir(d): Path(d).mkdir(parents=True, exist_ok=True)

def is_number(x):
    try:
        float(x)
        return True
    except:
        return False

def read_csv_auto(path):
    # Try header=0 first; if the columns are numeric-like, try header=None
    try:
        df = pd.read_csv(path, header=0, low_memory=False)
        cols = list(df.columns)
        # if columns look numeric (0,1,2...) then re-read without header
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
    return df.columns[-1]

def choose_image_shape(n_features):
    # prefer 17x11 for ~187
    if n_features <= 187:
        return (17,11)
    # else find near-rectangle
    best = None
    for h in range(8, int(math.ceil(math.sqrt(n_features)))+8):
        w = math.ceil(n_features / h)
        pad = h*w - n_features
        if best is None or pad < best[0]:
            best = (pad,h,w)
    _,h,w = best
    return (h,w)

def build_simple_2d_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3,3), activation='relu', padding='same'),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPool2D((2,2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_and_save_confusion(cm, classes, outpath):
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

def main(args):
    data_path = args.data_path
    out_dir = args.out_dir
    ensure_dir(out_dir)

    print("Loading CSV (auto-detect header)...")
    df = read_csv_auto(data_path)
    print("Loaded shape:", df.shape)

    label_col = detect_label_col(df)
    print("Detected label column:", label_col)

    # drop common metadata except label
    for c in ['record','rec','time','timestamp','index']:
        if c in df.columns and c != label_col:
            df = df.drop(columns=[c])

    if label_col not in df.columns:
        raise ValueError("Label column not found automatically. Edit script or provide --label_col.")

    y_raw = df[label_col].astype(str).values
    X_df = df.drop(columns=[label_col])
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        X_df = X_df.apply(pd.to_numeric, errors='coerce')
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()

    X = X_df[numeric_cols].astype(np.float32).values
    print("Numeric features used:", X.shape[1])

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_.tolist()
    num_classes = len(classes)
    print("Classes detected:", classes)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    img_h, img_w = choose_image_shape(X_scaled.shape[1])
    needed = img_h * img_w
    if X_scaled.shape[1] < needed:
        pad = needed - X_scaled.shape[1]
        X_scaled = np.pad(X_scaled, ((0,0),(0,pad)), mode='constant', constant_values=0.0)
    X_imgs = X_scaled.reshape((-1, img_h, img_w, 1))
    print("Image shape:", X_imgs.shape)

    # splits
    X_train, X_tmp, y_train, y_tmp = train_test_split(X_imgs, y, test_size=args.test_size+args.val_size, stratify=y, random_state=42)
    val_rel = args.val_size / (args.test_size + args.val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=42)

    print("Train/Val/Test shapes:", X_train.shape, X_val.shape, X_test.shape)

    model = build_simple_2d_cnn(input_shape=X_train.shape[1:], num_classes=num_classes)
    model.summary()

    ckpt = callbacks.ModelCheckpoint(os.path.join(out_dir, "best_model.h5"), monitor='val_loss', save_best_only=True)
    early = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    hist = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=args.epochs, batch_size=args.batch_size, callbacks=[ckpt, early], verbose=2)

    # save model
    model.save(os.path.join(out_dir, "ecg_simple_cnn.h5"))
    print("Model saved to", os.path.join(out_dir, "ecg_simple_cnn.h5"))

    # evaluate
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)

    rows = []
    for i, cname in enumerate(classes):
        rows.append({"class": str(cname), "precision": float(np.round(precision[i],4)), "recall": float(np.round(recall[i],4)), "f1_score": float(np.round(f1[i],4)), "support": int(support[i])})
    rows.append({"class":"accuracy", "precision":"", "recall":"", "f1_score": float(np.round(acc,4)), "support": len(y_test)})
    rows.append({"class":"macro_avg", "precision": float(np.round(p_macro,4)), "recall": float(np.round(r_macro,4)), "f1_score": float(np.round(f1_macro,4)), "support": len(y_test)})
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(os.path.join(out_dir, "results_table.csv"), index=False)
    print("Saved metrics to", os.path.join(out_dir, "results_table.csv"))

    # confusion
    cm = confusion_matrix(y_test, y_pred)
    plot_and_save_confusion(cm, classes, os.path.join(out_dir, "confusion_matrix.png"))

    # training curves
    plt.figure()
    plt.plot(hist.history.get('loss',[]), label='train_loss')
    plt.plot(hist.history.get('val_loss',[]), label='val_loss')
    plt.plot(hist.history.get('accuracy',[]), label='train_acc')
    plt.plot(hist.history.get('val_accuracy',[]), label='val_acc')
    plt.legend()
    plt.title("Training curves")
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=200)
    plt.close()
    print("Saved training_curves.png")

    # save classification report
    report = classification_report(y_test, y_pred, target_names=[str(c) for c in classes], zero_division=0)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    print("Saved classification_report.txt")

    # meta
    meta = {"classes": classes, "image_shape": [img_h,img_w,1], "n_features_used": len(numeric_cols)}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("All results saved under:", out_dir)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=r"archive\MIT-BIH Arrhythmia Database.csv")
    parser.add_argument("--out_dir", type=str, default="feature_map_results_simple")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
