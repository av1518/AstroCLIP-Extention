# In this script, we plot the training and validation loss and learning rate per epoch.
# We use CSV files exported from Weights & Biases to plot the data.
# %%
import pandas as pd
import matplotlib.pyplot as plt


train_file_path = "data/wandb_export_train_loss.csv"
train_data = pd.read_csv(train_file_path)
val_file_path = "data/wandb_export_val_loss_no_temp.csv"
val_data = pd.read_csv(val_file_path)

# Saved every 10 steps during training, need to conver to epochs
total_epochs = 80
train_total_steps = train_data["Step"].max()
train_steps_per_epoch = train_total_steps / total_epochs

# Calculate the approximate epoch number for each step in the training data
train_data["Epoch"] = (train_data["Step"] / train_steps_per_epoch).round().astype(int)

train_loss_column = "[256, 256, 256, 128], lr=0.0005, alt-cropped-validation - train_loss_no_temperature_epoch"
train_epoch_loss = train_data.groupby("Epoch")[train_loss_column].mean().reset_index()

val_total_steps = val_data["Step"].max()
val_steps_per_epoch = val_total_steps / total_epochs

val_data["Epoch"] = (val_data["Step"] / val_steps_per_epoch).round().astype(int)

val_loss_column = "[256, 256, 256, 128], lr=0.0005, alt-cropped-validation - validation_loss_no_temperature"
val_epoch_loss = val_data.groupby("Epoch")[val_loss_column].mean().reset_index()


new_lr_file_path = "data/wandb_export_lr.csv"
new_lr_data = pd.read_csv(new_lr_file_path)

new_lr_total_steps = new_lr_data["Step"].max()
new_lr_steps_per_epoch = new_lr_total_steps / total_epochs

new_lr_data["Epoch"] = (
    (new_lr_data["Step"] / new_lr_steps_per_epoch).round().astype(int)
)

new_lr_column = (
    "[256, 256, 256, 128], lr=0.0005, alt-cropped-validation - learning_rate_step"
)


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
ax1.plot(
    train_epoch_loss["Epoch"],
    train_epoch_loss[train_loss_column],
    marker="o",
    linestyle="-",
    color="blue",
    label="Train Loss",
)
ax1.plot(
    val_epoch_loss["Epoch"],
    val_epoch_loss[val_loss_column],
    marker="o",
    linestyle="-",
    color="orange",
    label="Validation Loss",
)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("InfoNCE Loss")
ax1.legend()
ax1.grid(True)

# Ensure x-axis values are shown on the top plot
ax1.tick_params(axis="x", which="both", labelbottom=True)

ax2.plot(
    new_lr_data["Epoch"],
    new_lr_data[new_lr_column],
    marker="o",
    linestyle="-",
    color="g",
    label="Learning Rate",
)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Learning Rate")
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.savefig("figures/training_plot.png", dpi=500, bbox_inches="tight")
plt.show()
