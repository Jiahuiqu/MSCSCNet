import logging


class Logger:
    def __init__(self, log_path, log_name='training_logger'):

        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)

    def get_train_logs(self, epoch, max_epoch, lr, train_loss, train_acc, val_loss, val_acc,
                       batch_time, epoch_time, train_epoch_time, val_epoch_time, eta):
        self.logger.info(f"Epoch: [{epoch + 1}/{max_epoch}] || lr: {lr} || Train_loss: {train_loss} || "
                         f"Train_acc: {(train_acc * 100):.2f}% || Val_loss: {val_loss} || "
                         f"Val_acc: {(val_acc * 100):.2f}% || Batch_time: {batch_time:.4f}s || "
                         f"Epoch_time: {epoch_time:.4f}s || Train_epoch_time: {train_epoch_time:.4f}s || "
                         f"Val_epoch_time: {val_epoch_time:.4f}s || ETA: {eta}")

    def get_test_logs(self, infer_total_time, infer_batch_time, oa, kappa, f1, pr, re, aa):
        self.logger.info('Testing Completed!')
        self.logger.info(f"Infer total time: {infer_total_time:.4f}s || Infer batch time: {infer_batch_time:.4f}s ||"
                         f" OA: {(oa * 100):.2f}% || Kappa: {kappa:.4f} || F1: {(f1 * 100):.2f}% ||"
                         f" Pr: {(pr * 100):.2f}% || Re: {(re * 100):.2f}% || AA: {(aa * 100):.2f}%")


if __name__ == '__main__':
    logger = Logger('../logs/demo_logs.txt')
    epoch = 2
    max_epoch = 100
    lr = 4e-4
    train_loss = 0.123456789
    train_acc = 99.49345012
    val_loss = 0.987654321
    val_acc = 98.95641273
    batch_time = 5.512456
    epoch_time = 86.518475
    train_epoch_time = 49.511654
    val_epoch_time = epoch_time - train_epoch_time
    eta = 52146
    logger.get_train_logs(epoch, max_epoch, lr, train_loss, train_acc, val_loss, val_acc, batch_time, epoch_time,
                          train_epoch_time, val_epoch_time, eta)
