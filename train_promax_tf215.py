import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

##############################################
# 1. LOAD + PREPARE DATA
##############################################
def load_ecg_csv(path):
    print("Loading CSV (header auto-detect)...")
    df = pd.read_csv(path, header=0 if any(str(c).isalpha() for c in pd.read_csv(path, nrows=1).columns) else None)
    print("Loaded shape:", df.shape)

    label_col_candidates = ["type", "label", "class"]
    label_col = None
    for c in df.columns:
        if str(c).lower() in label_col_candidates:
            label_col = c
            break
    if label_col is None:
        label_col = df.columns[-1]

    print("Detected label column:", label_col)

    X = df.drop(columns=[label_col])
    y = df[label_col]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    print("Numeric features used:", len(numeric_cols))

    le = LabelEncoder()
    y = le.fit_transform(y)
    classes = le.classes_
    print("Classes detected:", list(classes))

    img_h, img_w = 17, 11
    X_np = X.to_numpy().astype(np.float32)
    X_img = X_np.reshape(-1, img_h, img_w, 1)
    print("Image shape:", X_img.shape)

    return X_img, y, numeric_cols, classes


##############################################
# 2. BUILD MODEL
##############################################
def build_model(img_h=17, img_w=11, num_classes=5):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(img_h, img_w, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',   # ✅ FIX: Removes TF warning
        metrics=['accuracy']
    )
    return model


##############################################
# 3. TRAIN + SAVE METRICS & IMAGES
##############################################
def plot_confusion(cm, classes, save_path):
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j], ha='center', va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print("Saved:", save_path)


##############################################
# MAIN
##############################################
def main(args):
    X, y, numeric_cols, classes = load_ecg_csv(args.data_path)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)
    val_split = args.val_size / args.test_size
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_split, random_state=42, stratify=y_temp)

    print("Train/Val/Test shapes:", X_train.shape, X_val.shape, X_test.shape)

    os.makedirs(args.out_dir, exist_ok=True)

    model = build_model(num_classes=len(classes))

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2
    )

    model_path = os.path.join(args.out_dir, "ecg_model.keras")   # ✅ NEW FORMAT
    model.save(model_path)
    print("Model saved to", model_path)

    y_pred = model.predict(X_test).argmax(axis=1)

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(cm, classes, os.path.join(args.out_dir, "confusion_matrix.png"))

    # Save training curves
    plt.figure()
    plt.plot(hist.history.get('loss', []), label='train_loss')
    plt.plot(hist.history.get('val_loss', []), label='val_loss')
    plt.plot(hist.history.get('accuracy', []), label='train_acc')
    plt.plot(hist.history.get('val_accuracy', []), label='val_acc')
    plt.legend()
    plt.title("Training Curves")
    plt.savefig(os.path.join(args.out_dir, "training_curves.png"), dpi=200)
    plt.close()
    print("Saved training_curves.png")

    # Save Classification Report
    report = classification_report(y_test, y_pred, target_names=[str(c) for c in classes], zero_division=0)
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # ✅ Generate Accuracy, Precision, Recall, F1 table
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
    acc = (y_pred == y_test).mean()

    df = pd.DataFrame({
        "Class": classes,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })
    df.loc[len(df)] = ["Overall Avg", df["Precision"].mean(), df["Recall"].mean(), df["F1 Score"].mean()]
    df.loc[len(df)] = ["Accuracy", acc, "-", "-"]

    df.to_csv(os.path.join(args.out_dir, "model_scores.csv"), index=False)
    print("Saved model_scores.csv")

    print("\n✅ All results saved under:", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="archive/MIT-BIH Arrhythmia Database.csv")
    parser.add_argument("--out_dir", type=str, default="promax_results")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
