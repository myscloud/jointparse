import matplotlib.pyplot as plt

def write_loss_file(log_file_path):
    acc_list = list()
    acc = list()
    with open(log_file_path) as log_file:
        for line in log_file:
            if line[0:5] == 'epoch':
                acc_list.append(acc.copy())
                acc.clear()
            else:
                tokens = line.strip().split(' ')
                loss = float(tokens[-1])
                acc.append(loss)

    acc_list.append(acc)

    with open('loss_log', 'w') as new_file:
        for epoch in acc_list:
            new_file.write('\t'.join([str(loss) for loss in epoch]) + '\n')


def plot_loss(log_file_path):
    acc_list = list()
    with open(log_file_path) as log_file:
        for line in log_file:
            epoch_loss = [float(x) for x in line.strip().split('\t')]
            acc_list.append(epoch_loss)

    train_loss = [x[0] for x in acc_list]
    eval_loss = [x[1] for x in acc_list]

    plt.plot(list(range(1, len(train_loss)+1)), train_loss, 'r-',
             list(range(1, len(train_loss)+1)), eval_loss, 'b-')
    plt.show()

if __name__ == '__main__':
    # write_loss_file('pos_train.log')
    plot_loss('loss_log')
