from tensorflow.keras import backend

smooth = 1e-7  # 用于防止分母为0.


# dice损失函数
def dice_coef(y_true, y_pred):
    y_true_f = backend.flatten(y_true)  # 将 y_true 拉伸为一维.
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (backend.sum(y_true_f * y_true_f) + backend.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)
