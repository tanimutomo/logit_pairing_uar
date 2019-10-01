import argparse

atas = dict(
    pgd_linf=[84.6, 82.1, 76.2, 66.9, 40.1, 12.9],
    pgd_l2=[85.0, 83.5, 79.6, 72.6, 59.1, 19.9],
    fw_l1=[84.4, 82.7, 76.3, 68.9, 56.4, 36.1],
    jpeg_linf=[85.0, 83.2, 79.3, 72.8, 34.8, 1.1],
    jpeg_l2=[84.8, 82.5, 78.9, 72.3, 47.5, 3.4],
    jpeg_l1=[84.8, 81.8, 76.2, 67.1, 46.4, 41.8],
    elastic=[85.9, 83.2, 78.1, 75.6, 57.0, 22.5],
    fog=[85.8, 83.8, 79.0, 68.4, 67.9, 64.7],
    snow=[84.0, 81.1, 77.7, 65.6, 59.5, 41.2],
    gabor=[84.0, 79.8, 79.8, 66.2, 44.7, 14.6]
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
