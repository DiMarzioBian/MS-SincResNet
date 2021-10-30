def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch <= 5:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5
    else:
        lr = 0.005 * pow(0.5, epoch // 30)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr