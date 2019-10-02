import argparse

atas = dict(
    pgd_linf=[91.0, 87.8, 81.6, 71.3, 46.5, 23.1],
    pgd_l2=[90.1, 86.4, 79.6, 67.3, 49.9, 17.3],
    fw_l1=[92.2, 90.0, 83.2, 73.8, 47.4, 35.3],
    jpeg_linf=[89.7, 87.0, 83.1, 78.6, 69.7, 35.4],
    jpeg_l1=[91.4, 88.1, 80.2, 68.9, 56.3, 37.7],
    elastic=[87.4, 81.3, 72.1, 58.2, 45.4, 27.8]
    )


def sum_accs(raw_accs):
    accs = raw_accs.split('|')
    accs = list(map(float, accs))
    return sum(accs)


def calc_uar(sum_accs, attack):
    try:
        sum_atas = sum(atas[attack])
    except:
        raise NotImplementedError
    return 100 * sum_accs / sum_atas


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--accs', type=str, required=True, help='accuracies separated by "|"')
    parser.add_argument('--attack', type=str, required=True, help='attack method name')
    opt = parser.parse_args()

    sum_acc = sum_accs(opt.accs)
    uar = calc_uar(sum_acc, opt.attack)

    print('sum of accs', sum_acc)
    print('UAR:', uar)
